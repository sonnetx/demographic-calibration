"""Local HuggingFace VLM implementation for reasoning models."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

import logging

from .base import BaseVLM, LogprobClassificationResult, TokenLogProb, VLMResponse

logger = logging.getLogger(__name__)


def _clear_gpu_cache():
    """Delete CUDA cache to prevent OOM between generation calls."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_cuda_on_error():
    """Reset CUDA state after a device-side assertion.

    A single CUDA assertion (e.g. from sampling with invalid probabilities)
    poisons the entire CUDA context, causing every subsequent kernel to fail.
    Synchronizing and clearing the cache lets the device recover.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass  # synchronize itself may raise if device is in error state
        torch.cuda.empty_cache()


# Models that use Qwen2-VL architecture
QWEN2_VL_MODELS = {
    "Fancy-MLLM/R1-Onevision-7B",
    "Qwen/Qwen2.5-VL-7B-Instruct",
}

# Qwen2-VL models that use <think> reasoning (subset of QWEN2_VL_MODELS)
QWEN2_VL_THINKING_MODELS = {
    "Fancy-MLLM/R1-Onevision-7B",
}

# Models that use GLM architecture
GLM_MODELS = {
    "zai-org/GLM-4.1V-9B-Thinking",
}

# Models that use Kimi architecture
KIMI_MODELS = {
    "moonshotai/Kimi-VL-A3B-Thinking-2506",
}

# Models that use OpenFlamingo architecture
OPENFLAMINGO_MODELS = {
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
}

# Models that use Llama-3.2-Vision architecture
LLAMA_VISION_MODELS = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

# Models that use PaliGemma/MedGemma architecture
MEDGEMMA_MODELS = {
    "google/medgemma-4b-it",
}


class LocalVLM(BaseVLM):
    """Local HuggingFace VLM for reasoning models (R1, GLM, Kimi, OpenFlamingo, Llama Vision)."""

    def __init__(
        self,
        api_key: str,  # Unused for local models, kept for interface compatibility
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        **kwargs,
    ):
        """
        Initialize local VLM.

        Args:
            api_key: Unused (kept for interface compatibility)
            model_name: HuggingFace model ID
            device: Device to load model on ("cuda" or "cpu")
            torch_dtype: Torch dtype ("float16", "bfloat16", or "float32")
            **kwargs: Additional configuration
        """
        super().__init__(api_key="", model_name=model_name, **kwargs)
        self.device = device
        self.dtype = getattr(torch, torch_dtype)

        # Determine model architecture
        self._is_qwen2_vl = model_name in QWEN2_VL_MODELS
        self._is_glm = model_name in GLM_MODELS
        self._is_kimi = model_name in KIMI_MODELS
        self._is_openflamingo = model_name in OPENFLAMINGO_MODELS
        self._is_llama_vision = model_name in LLAMA_VISION_MODELS
        self._is_medgemma = model_name in MEDGEMMA_MODELS
        self._is_thinking = (model_name in QWEN2_VL_THINKING_MODELS) or self._is_glm or self._is_kimi

        # Load model and processor based on architecture
        self._load_model()

    @property
    def is_reasoning_model(self) -> bool:
        return self._is_thinking

    def _load_model(self) -> None:
        """Load model and processor based on architecture."""
        if self._is_openflamingo:
            from huggingface_hub import hf_hub_download
            from open_flamingo import create_model_and_transforms

            self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                cross_attn_every_n_layers=1,
            )
            checkpoint_path = hf_hub_download(self.model_name, "checkpoint.pt")
            self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
            self.model = self.model.to(self.device, dtype=self.dtype)
            self.processor = None  # OpenFlamingo uses image_processor directly
        elif self._is_qwen2_vl:
            # R1-Onevision is fine-tuned Qwen2.5-VL, use Qwen2_5_VL model class
            from transformers import Qwen2_5_VLForConditionalGeneration

            # Add compatibility patch for older PyTorch versions
            if not hasattr(torch.compiler, 'is_compiling'):
                torch.compiler.is_compiling = lambda: False

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            # Processor fallback for R1-Onevision.
            # use_fast=False: transformers >= 4.52 defaults to the fast image
            # processor, which produces much larger vision sequences for
            # high-resolution radiographs (GPU OOM) and differs from the slow
            # processor used in the original experiments.
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False,
                )
            except ValueError as e:
                if "Unrecognized image processor" in str(e):
                    # Fall back to base Qwen2.5-VL processor
                    self.processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        trust_remote_code=True,
                        use_fast=False,
                    )
                else:
                    raise

            self.tokenizer = self.processor.tokenizer
        elif self._is_llama_vision:
            from transformers import MllamaForConditionalGeneration

            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = self.processor.tokenizer
        elif self._is_medgemma:
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = self.processor.tokenizer
        else:
            # GLM and Kimi use AutoModelForCausalLM pattern
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

        self.model.eval()

    def _decode_base64_image(self, image_base64: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data)).convert("RGB")

    def _prepare_qwen2_vl_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare inputs for Qwen2-VL architecture."""
        # Build conversation format for Qwen2-VL
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template and process
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _prepare_glm_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare inputs for GLM architecture."""
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor(
            messages=messages,
            return_tensors="pt",
        )

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _prepare_kimi_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare inputs for Kimi architecture."""
        # Kimi uses similar pattern to GLM
        return self._prepare_glm_inputs(image, prompt)

    def _prepare_openflamingo_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare inputs for OpenFlamingo architecture."""
        # OpenFlamingo native format uses simple <image>text<|endofchunk|> blocks
        # without User:/Assistant: markers (it wasn't trained with chat formatting)
        # Vision input shape: batch x num_media x num_frames x C x H x W
        vision_x = self.image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0)

        prompt_text = f"<image>{prompt}\n"
        lang_x = self.tokenizer(prompt_text, return_tensors="pt")

        return {
            "vision_x": vision_x.to(self.device, dtype=self.dtype),
            "lang_x": lang_x["input_ids"].to(self.device),
            "attention_mask": lang_x["attention_mask"].to(self.device),
        }

    def _prepare_medgemma_inputs(
        self,
        image: Image.Image,
        prompt: str,
        include_system: bool = False,
    ) -> dict:
        """Prepare inputs for MedGemma (Gemma 3 chat-style) architecture.

        Uses apply_chat_template with structured messages containing
        image objects — the processor handles image token insertion.

        Args:
            include_system: If True, include the system instruction. Gemma 3
                models have limited system prompt support, and the 4B model
                works better with direct user messages only.
        """
        messages = []
        if include_system:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_instruction}],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Only cast float tensors (pixel_values) to model dtype;
        # integer tensors (input_ids, attention_mask) must stay as Long.
        return {
            k: v.to(self.model.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.model.device)
            for k, v in inputs.items()
        }

    def _prepare_llama_vision_inputs(
        self,
        image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare inputs for Llama-3.2-Vision architecture."""
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            images=[image],
            text=text,
            return_tensors="pt",
        )

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _extract_logprobs(
        self,
        outputs,
        input_len: int,
        logprobs_top_k: int,
    ) -> list[TokenLogProb]:
        """Extract token logprobs from generation outputs."""
        token_logprobs = []

        if not hasattr(outputs, "scores") or outputs.scores is None:
            return token_logprobs

        # outputs.scores is a tuple of (batch_size, vocab_size) tensors
        for i, scores in enumerate(outputs.scores):
            # Convert scores to log probabilities
            logprobs = F.log_softmax(scores, dim=-1)

            # Get the generated token ID at this position
            token_id = outputs.sequences[0, input_len + i].item()
            token_logprob = logprobs[0, token_id].item()

            # Get token string
            token_str = self.tokenizer.decode([token_id])

            # Get top-k alternatives
            top_k_logprobs, top_k_ids = torch.topk(logprobs[0], k=logprobs_top_k)
            alternatives = [
                (self.tokenizer.decode([tid.item()]), lp.item())
                for tid, lp in zip(top_k_ids, top_k_logprobs)
                if tid.item() != token_id
            ]

            token_logprobs.append(
                TokenLogProb(
                    token=token_str,
                    logprob=token_logprob,
                    top_alternatives=alternatives[:logprobs_top_k],
                )
            )

        return token_logprobs

    async def predict(
        self,
        image_base64: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_logprobs: bool = True,
        logprobs_top_k: int = 5,
    ) -> VLMResponse:
        """
        Make prediction using local model.

        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            return_logprobs: Whether to return log probabilities
            logprobs_top_k: Number of top alternative tokens

        Returns:
            VLMResponse with prediction and logprobs
        """
        # Decode image
        image = self._decode_base64_image(image_base64)

        # Prepare inputs based on architecture
        if self._is_openflamingo:
            inputs = self._prepare_openflamingo_inputs(image, prompt)
        elif self._is_qwen2_vl:
            inputs = self._prepare_qwen2_vl_inputs(image, prompt)
        elif self._is_llama_vision:
            inputs = self._prepare_llama_vision_inputs(image, prompt)
        elif self._is_medgemma:
            inputs = self._prepare_medgemma_inputs(image, prompt)
        elif self._is_glm:
            inputs = self._prepare_glm_inputs(image, prompt)
        else:
            inputs = self._prepare_kimi_inputs(image, prompt)

        # OpenFlamingo has different generation API
        if self._is_openflamingo:
            input_len = inputs["lang_x"].shape[1]

            from transformers import GenerationConfig

            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            if temperature > 0:
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=0,
                    pad_token_id=pad_token_id,
                )
            else:
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            with torch.no_grad():
                outputs = self.model.generate(
                    vision_x=inputs["vision_x"],
                    lang_x=inputs["lang_x"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=gen_config,
                    output_scores=return_logprobs,
                    return_dict_in_generate=True,
                )
        else:
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "output_scores": return_logprobs,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }

            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
                gen_kwargs["top_k"] = 0  # disable top-k to let top_p control diversity
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode generated text (excluding input)
        generated_ids = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract logprobs if requested
        token_logprobs = None
        if return_logprobs:
            token_logprobs = self._extract_logprobs(outputs, input_len, logprobs_top_k)

        del outputs, inputs
        _clear_gpu_cache()

        # Parse diagnosis
        prediction = self.parse_diagnosis(text)

        return VLMResponse(
            text=text,
            prediction=prediction,
            token_logprobs=token_logprobs,
            finish_reason="stop",
            model_name=self.model_name,
        )

    def _extract_ab_logprobs_from_scores(
        self,
        scores: torch.Tensor,
        logprobs_top_k: int,
    ) -> list[TokenLogProb]:
        """Extract A/B logprobs from raw vocabulary scores at position 0.

        Even if the generated token is not A/B (e.g. <think>), the logits
        still contain the probability mass over the full vocabulary.
        """
        logprobs = F.log_softmax(scores, dim=-1)

        # Look up token IDs for "A" and "B"
        a_ids = self.tokenizer.encode("A", add_special_tokens=False)
        b_ids = self.tokenizer.encode("B", add_special_tokens=False)

        # Get the generated token
        top_k_logprobs, top_k_ids = torch.topk(logprobs[0], k=logprobs_top_k)
        top_token_id = top_k_ids[0].item()
        top_token_str = self.tokenizer.decode([top_token_id])
        top_logprob = top_k_logprobs[0].item()

        alternatives = []
        for tid, lp in zip(top_k_ids, top_k_logprobs):
            if tid.item() != top_token_id:
                alternatives.append(
                    (self.tokenizer.decode([tid.item()]), lp.item())
                )

        # Also ensure A and B are in the alternatives if not already in top-k
        for label, ids in [("A", a_ids), ("B", b_ids)]:
            if ids:
                token_id = ids[0]
                lp_val = logprobs[0, token_id].item()
                # Check if already in alternatives
                already = any(
                    tok.strip().upper() == label
                    for tok, _ in alternatives
                )
                if not already and token_id != top_token_id:
                    alternatives.append((label, lp_val))

        return [
            TokenLogProb(
                token=top_token_str,
                logprob=top_logprob,
                top_alternatives=alternatives,
            )
        ]

    def _find_post_think_answer(self, generated_ids: torch.Tensor) -> int | None:
        """Find the token index of the A/B answer relative to </think>.

        For thinking models the expected output is:
            <think>...reasoning...</think>A
        but R1-Onevision often places the answer *inside* the think block:
            <think>...B. Malignant\n</think><|im_end|>

        Strategy:
        1. Look for A/B *after* </think> (ideal case).
        2. Fallback: scan backwards from </think> for the last A/B token
           inside the reasoning block.

        Returns:
            Index into outputs.scores for the answer token, or None.
        """
        ids = generated_ids.tolist()

        # Decode progressively to find </think> boundary
        cumulative = ""
        think_end_idx = None

        for i, token_id in enumerate(ids):
            cumulative += self.tokenizer.decode([token_id])
            # R1 sometimes emits </pre> instead of </think>
            if think_end_idx is None and (
                "</think>" in cumulative or "</pre>" in cumulative
            ):
                think_end_idx = i + 1  # first token after closing tag
                break

        if think_end_idx is None:
            # No </think> found — model may have hit max_new_tokens mid-reasoning.
            # Scan backwards from end for last A/B token as last resort.
            logger.debug(
                "No </think> found in %d generated tokens. "
                "Scanning backwards for last A/B token.",
                len(ids),
            )
            for i in range(len(ids) - 1, max(len(ids) - 40, -1), -1):
                if i < 0:
                    break
                tok_str = self.tokenizer.decode([ids[i]]).strip().upper()
                if tok_str in ("A", "B"):
                    logger.debug(
                        "Found answer token %r at position %d (no </think>, last-resort scan)",
                        tok_str, i,
                    )
                    return i
            return None

        # 1) Scan tokens after </think> for A or B
        for i in range(think_end_idx, len(ids)):
            tok_str = self.tokenizer.decode([ids[i]]).strip().upper()
            if tok_str in ("A", "B"):
                logger.debug(
                    "Found answer token %r after </think> at position %d (of %d total)",
                    tok_str, i, len(ids),
                )
                return i

        # 2) Fallback: scan backwards inside the think block for last A/B
        for i in range(think_end_idx - 2, max(think_end_idx - 50, -1), -1):
            if i < 0:
                break
            tok_str = self.tokenizer.decode([ids[i]]).strip().upper()
            if tok_str in ("A", "B"):
                logger.debug(
                    "Found answer token %r inside think block at position %d (of %d total)",
                    tok_str, i, len(ids),
                )
                return i

        logger.debug(
            "No A/B token found near </think> at position %d. "
            "Remaining tokens: %r",
            think_end_idx,
            self.tokenizer.decode(ids[max(0, think_end_idx - 20):]),
        )
        return None

    async def predict_classification(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """Constrained A/B classification for logprob extraction.

        For thinking models (R1, GLM, Kimi): lets the model reason freely,
        then extracts A/B logprobs from the answer position after </think>.
        The logprobs at that position reflect genuine post-reasoning
        uncertainty, not the near-random A/B distribution at position 0
        where the model expects to generate <think>.

        For OpenFlamingo: max_new_tokens=1 with direct vocab extraction.
        """
        try:
            return await self._predict_classification_impl(
                image_base64, logprobs_top_k, temperature,
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                logger.error(
                    "CUDA error in predict_classification, resetting device: %s", e
                )
                _reset_cuda_on_error()
                raise
            raise

    async def _predict_classification_impl(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """Inner implementation of predict_classification."""
        image = self._decode_base64_image(image_base64)
        classification_prompt = self.get_classification_prompt()
        is_thinking = self._is_thinking

        if self._is_openflamingo or self._is_medgemma:
            # OpenFlamingo-3B (1B LM) and MedGemma-4B cannot follow complex
            # multi-paragraph instructions.  Use a short VQA-style prompt
            # ending with "Answer:" so the first generated token is guided
            # toward A/B rather than <eos>/<|endofchunk|>.
            a_cap = self._task_config.a_label.capitalize()
            b_cap = self._task_config.b_label.capitalize()
            classification_prompt = (
                f"Classify this medical image. "
                f"A. {a_cap} B. {b_cap}\nAnswer:"
            )

            if self._is_openflamingo:
                inputs = self._prepare_openflamingo_inputs(image, classification_prompt)
                input_len = inputs["lang_x"].shape[1]

                from transformers import GenerationConfig

                gen_config_kwargs = {
                    "max_new_tokens": 1,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_config_kwargs["temperature"] = temperature
                    gen_config_kwargs["top_p"] = 0.9
                    gen_config_kwargs["top_k"] = 0
                gen_config = GenerationConfig(**gen_config_kwargs)
                with torch.no_grad():
                    outputs = self.model.generate(
                        vision_x=inputs["vision_x"],
                        lang_x=inputs["lang_x"],
                        attention_mask=inputs["attention_mask"],
                        generation_config=gen_config,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
            else:
                # MedGemma
                inputs = self._prepare_medgemma_inputs(image, classification_prompt)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = {
                    "max_new_tokens": 1,
                    "do_sample": temperature > 0,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9
                    gen_kwargs["top_k"] = 0
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)

            if hasattr(outputs, "scores") and outputs.scores:
                token_logprobs = self._extract_ab_logprobs_from_scores(
                    outputs.scores[0], logprobs_top_k
                )
            else:
                token_logprobs = []

            del outputs, inputs
            _clear_gpu_cache()
            return self._extract_ab_confidence(token_logprobs)

        # Prepare inputs for chat-based models
        if self._is_qwen2_vl:
            inputs = self._prepare_qwen2_vl_inputs(image, classification_prompt)
        elif self._is_llama_vision:
            inputs = self._prepare_llama_vision_inputs(image, classification_prompt)
        elif self._is_glm:
            inputs = self._prepare_glm_inputs(image, classification_prompt)
        else:
            inputs = self._prepare_kimi_inputs(image, classification_prompt)

        input_len = inputs["input_ids"].shape[1]

        if is_thinking:
            # Let thinking models reason freely, then extract logprobs
            # at the answer token position after </think>.
            # 4096 tokens: CheXpert X-rays can trigger very long reasoning
            # chains (500+ tokens) — 2048 was too tight and R1 would hit
            # the limit right at </think> with no room for the answer.
            gen_kwargs = {
                "max_new_tokens": 4096,
                "do_sample": temperature > 0,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # Prevent degenerate repetition loops (e.g. R1 repeating
                # Chinese text for 2048 tokens without producing </think>).
                "repetition_penalty": 1.2,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                # Qwen2-VL ships with top_k=1 and top_p=0.001 in its
                # default generation_config, which makes sampling fully
                # deterministic.  Override explicitly so SC gets real variance.
                gen_kwargs["top_p"] = 0.9
                gen_kwargs["top_k"] = 0

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs.sequences[0, input_len:]
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            answer_idx = self._find_post_think_answer(generated_ids)

            if answer_idx is not None and answer_idx < len(outputs.scores):
                logger.info(
                    "Classification: found answer at token %d/%d",
                    answer_idx, len(outputs.scores),
                )
                token_logprobs = self._extract_ab_logprobs_from_scores(
                    outputs.scores[answer_idx], logprobs_top_k
                )
            else:
                # No A/B token found — fall back to text parsing for the
                # prediction and mark as non-compliant with uninformative
                # confidence.  Using scores[-1] would give garbage logprobs
                # from whatever token ended generation (e.g. <|im_end|> or
                # a repeated Chinese character).
                logger.warning(
                    "Could not find A/B answer after </think> for %s "
                    "(answer_idx=%s, n_scores=%d). "
                    "Falling back to text-based prediction. "
                    "Full text (last 200 chars): %r",
                    self.model_name,
                    answer_idx,
                    len(outputs.scores),
                    full_text[-200:],
                )
                parsed_prediction = self.parse_diagnosis(full_text)
                del outputs, inputs
                _clear_gpu_cache()
                return LogprobClassificationResult(
                    predicted_label=parsed_prediction,
                    confidence=0.5,
                    a_logprob=None,
                    b_logprob=None,
                    raw_first_token="<no_answer>",
                    compliant=False,
                    raw_response=full_text,
                )

            del outputs, inputs
            _clear_gpu_cache()

            result = self._extract_ab_confidence(token_logprobs)
            result.raw_response = full_text

            # Self-consistency fix for thinking models: when temperature > 0,
            # the reasoning phase varies but converges to the same answer,
            # so the argmax prediction is always identical across K samples.
            # Instead, sample from the A/B probability distribution so SC
            # captures the model's genuine uncertainty at the decision point.
            if (
                temperature > 0
                and result.a_logprob is not None
                and result.b_logprob is not None
            ):
                logprobs_pair = np.array([result.a_logprob, result.b_logprob])
                max_lp = np.max(logprobs_pair)
                # Guard against both logprobs being -inf
                if not np.isfinite(max_lp):
                    logger.warning(
                        "SC sampling: both A/B logprobs are -inf, skipping"
                    )
                else:
                    probs = np.exp(logprobs_pair - max_lp)
                    probs = probs / probs.sum()
                    a_label = self._task_config.a_label
                    b_label = self._task_config.b_label
                    sampled = a_label if np.random.random() < probs[0] else b_label
                    logger.info(
                        "SC sampling (T=%.2f): P(A)=%.4f, P(B)=%.4f, sampled=%s",
                        temperature, probs[0], probs[1], sampled,
                    )
                    result.predicted_label = sampled
                    result.confidence = float(probs[0] if sampled == a_label else probs[1])
                    result.compliant = True

            logger.info(
                "Classification result: predicted=%s, confidence=%.3f, "
                "compliant=%s, raw_first_token=%r",
                result.predicted_label, result.confidence,
                result.compliant, result.raw_first_token,
            )
            return result

        # Non-thinking chat model: single-token extraction
        gen_kwargs = {
            "max_new_tokens": 1,
            "do_sample": temperature > 0,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 0
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        if hasattr(outputs, "scores") and outputs.scores:
            token_logprobs = self._extract_ab_logprobs_from_scores(
                outputs.scores[0], logprobs_top_k
            )
        else:
            token_logprobs = self._extract_logprobs(outputs, input_len, logprobs_top_k)

        del outputs, inputs
        _clear_gpu_cache()
        return self._extract_ab_confidence(token_logprobs)

    async def predict_confidence_elicitation(
        self,
        image_base64: str,
        classification_prompt: str,
        predicted_label: str,
    ) -> str:
        """Multi-turn confidence elicitation for local models."""
        image = self._decode_base64_image(image_base64)
        ce_prompt = self.get_confidence_elicitation_prompt()
        # For thinking models, use a richer assistant turn so the model
        # doesn't re-classify from scratch. Just "A" triggers re-reasoning.
        if self._is_thinking:
            label_word = self._task_config.a_label.capitalize() if predicted_label.upper().startswith("A") else self._task_config.b_label.capitalize()
            assistant_content = f"{predicted_label}. {label_word}"
        else:
            assistant_content = predicted_label

        # Build multi-turn messages
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": classification_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ce_prompt},
                ],
            },
        ]

        if self._is_openflamingo:
            # OpenFlamingo doesn't support multi-turn natively.
            # Use a simplified single-turn prompt (the full prompts
            # are too complex for a 1B language model).
            a_cap = self._task_config.a_label.capitalize()
            b_cap = self._task_config.b_label.capitalize()
            combined_prompt = (
                f"Classify this medical image. "
                f"A. {a_cap} B. {b_cap}\n"
                f"Answer: {predicted_label}\n"
                f"How confident are you? Answer as a percentage.\n"
                f"Confidence:"
            )
            inputs = self._prepare_openflamingo_inputs(image, combined_prompt)
            input_len = inputs["lang_x"].shape[1]

            from transformers import GenerationConfig

            gen_config = GenerationConfig(
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    vision_x=inputs["vision_x"],
                    lang_x=inputs["lang_x"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=gen_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                )
            generated_ids = outputs.sequences[0, input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            del outputs, inputs
            _clear_gpu_cache()
            return text

        if self._is_medgemma:
            # MedGemma-4B can't follow complex multi-paragraph prompts.
            # Use a simplified single-turn prompt (same approach as OpenFlamingo).
            a_cap = self._task_config.a_label.capitalize()
            b_cap = self._task_config.b_label.capitalize()
            combined_prompt = (
                f"Classify this medical image. "
                f"A. {a_cap} B. {b_cap}\n"
                f"Answer: {predicted_label}\n"
                f"How confident are you? Answer as a percentage.\n"
                f"Confidence:"
            )
            inputs = self._prepare_medgemma_inputs(image, combined_prompt)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": 64,
                "do_sample": False,
                "output_scores": False,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = outputs.sequences[0, input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            del outputs, inputs
            _clear_gpu_cache()
            return text

        # For chat-based models (Qwen2-VL, Llama Vision, GLM, Kimi)
        if self._is_qwen2_vl:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif self._is_llama_vision:
            # Llama-3.2-Vision: image placeholder in messages is just
            # {"type": "image"}, actual image passed to processor.
            llama_messages = [
                {
                    "role": "system",
                    "content": self.system_instruction,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": classification_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ce_prompt},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                llama_messages,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                images=[image],
                text=text,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            # GLM / Kimi
            inputs = self.processor(
                messages=messages,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        # Thinking models need enough tokens to reason before answering
        max_tokens = 1024 if self._is_thinking else 128

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "output_scores": False,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs.sequences[0, input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        del outputs, inputs
        _clear_gpu_cache()
        logger.info(
            "Confidence elicitation response (last 200 chars): %r",
            response[-200:],
        )
        return response

    async def predict_batch(
        self,
        images_base64: list[str],
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_logprobs: bool = True,
    ) -> list[VLMResponse]:
        """
        Batch prediction (sequential for local models).

        Args:
            images_base64: List of base64-encoded images
            prompt: Text prompt for all images
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            return_logprobs: Whether to request log probabilities

        Returns:
            List of VLMResponse objects
        """
        results = []
        for img in images_base64:
            try:
                result = await self.predict(
                    img, prompt, temperature, max_tokens, return_logprobs
                )
                results.append(result)
            except Exception as e:
                results.append(
                    VLMResponse(
                        text=f"Error: {e}",
                        prediction="unknown",
                        model_name=self.model_name,
                    )
                )
        return results
