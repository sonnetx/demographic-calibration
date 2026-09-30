"""vLLM-served model accessed via OpenAI-compatible API.

Designed for thinking models (R1-Onevision-7B) where inference is served
by a local vLLM server. vLLM's continuous batching handles concurrent
self-consistency requests automatically, providing significant speedup
over sequential HuggingFace Transformers inference.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
import openai
from openai import AsyncOpenAI

from .base import (
    BaseVLM,
    LogprobClassificationResult,
    TokenLogProb,
    VLMResponse,
    retry_with_backoff,
    strip_thinking_tags,
)
from .openai_vlm import _is_retryable_openai

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 2.0
_MAX_DELAY = 120.0


class VLLMModel(BaseVLM):
    """vLLM-served model via OpenAI-compatible API.

    Connects to a running vLLM server (started separately or in the same
    SLURM job). All inference happens on the server side; this class is
    a lightweight HTTP client.

    Designed primarily for thinking models (R1-Onevision-7B) where
    predict_classification() must generate up to 4096 tokens of reasoning
    before the A/B answer, then extract logprobs at the answer position.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        is_thinking: bool = True,
        timeout: float = 300.0,
        max_concurrent: int = 32,
        rate_limit_rpm: int = 999,
        **kwargs,
    ):
        super().__init__(api_key="", model_name=model_name, **kwargs)
        self.base_url = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        self._is_thinking = is_thinking
        self.client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't need a real key
            base_url=self.base_url,
            timeout=timeout,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            "VLLMModel initialized: model=%s, base_url=%s, thinking=%s",
            model_name,
            self.base_url,
            is_thinking,
        )

    @property
    def is_reasoning_model(self) -> bool:
        return self._is_thinking

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    async def predict(
        self,
        image_base64: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_logprobs: bool = True,
        logprobs_top_k: int = 5,
    ) -> VLMResponse:
        async def _call():
            async with self._semaphore:
                kwargs: dict = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_instruction,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "logprobs": return_logprobs,
                    "top_logprobs": logprobs_top_k if return_logprobs else None,
                }
                if self._is_thinking:
                    kwargs["extra_body"] = {"repetition_penalty": 1.2}
                return await self.client.chat.completions.create(**kwargs)

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_openai,
                context=f"VLLMModel.predict({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(f"vLLM API error: {e}") from e

        choice = response.choices[0]
        text = choice.message.content or ""

        # Parse logprobs
        token_logprobs = None
        if return_logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [
                TokenLogProb(
                    token=tlp.token,
                    logprob=tlp.logprob,
                    top_alternatives=[
                        (alt.token, alt.logprob)
                        for alt in (tlp.top_logprobs or [])
                        if alt.token != tlp.token
                    ],
                )
                for tlp in choice.logprobs.content
            ]

        # Strip thinking tags before parsing diagnosis
        clean_text = strip_thinking_tags(text) if self._is_thinking else text
        prediction = self.parse_diagnosis(clean_text)

        return VLMResponse(
            text=text,
            prediction=prediction,
            token_logprobs=token_logprobs,
            finish_reason=choice.finish_reason or "stop",
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # predict_classification()
    # ------------------------------------------------------------------

    def _find_post_think_answer_from_logprobs(
        self,
        logprobs_content: list,
    ) -> int | None:
        """Find the index of the A/B answer token after </think>.

        Mirrors LocalVLM._find_post_think_answer() but operates on
        OpenAI-format ChatCompletionTokenLogprob objects (with .token
        string attributes) instead of tensor token IDs.

        Strategy:
        1. Scan tokens to find </think> boundary (cumulative text).
        2. After </think>, find first token whose stripped text is A or B.
        3. Fallback: scan backwards inside the think block.
        4. Last resort: scan backwards from end of sequence.

        Returns:
            Index into logprobs_content, or None.
        """
        cumulative = ""
        think_end_idx = None

        for i, tlp in enumerate(logprobs_content):
            cumulative += tlp.token
            if think_end_idx is None and (
                "</think>" in cumulative or "</pre>" in cumulative
            ):
                think_end_idx = i + 1
                break

        if think_end_idx is None:
            # No </think> found — scan backwards for last A/B token
            logger.debug(
                "No </think> found in %d tokens. Scanning backwards.",
                len(logprobs_content),
            )
            for i in range(
                len(logprobs_content) - 1,
                max(len(logprobs_content) - 40, -1),
                -1,
            ):
                if i < 0:
                    break
                if logprobs_content[i].token.strip().upper() in ("A", "B"):
                    logger.debug(
                        "Found answer token %r at position %d (no </think>)",
                        logprobs_content[i].token.strip(),
                        i,
                    )
                    return i
            return None

        # 1) Scan tokens after </think> for A or B
        for i in range(think_end_idx, len(logprobs_content)):
            if logprobs_content[i].token.strip().upper() in ("A", "B"):
                logger.debug(
                    "Found answer token %r after </think> at position %d",
                    logprobs_content[i].token.strip(),
                    i,
                )
                return i

        # 2) Fallback: scan backwards inside think block
        for i in range(think_end_idx - 2, max(think_end_idx - 50, -1), -1):
            if i < 0:
                break
            if logprobs_content[i].token.strip().upper() in ("A", "B"):
                logger.debug(
                    "Found answer token %r inside think block at position %d",
                    logprobs_content[i].token.strip(),
                    i,
                )
                return i

        logger.debug(
            "No A/B token found near </think> at position %d.",
            think_end_idx,
        )
        return None

    async def predict_classification(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """Constrained A/B classification with logprob extraction.

        For thinking models: generates full reasoning + answer, then
        extracts logprobs at the A/B answer position after </think>.
        """
        classification_prompt = self.get_classification_prompt()

        if self._is_thinking:
            max_tokens = 4096
        else:
            max_tokens = 1

        async def _call():
            async with self._semaphore:
                kwargs: dict = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_instruction,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                    },
                                },
                                {"type": "text", "text": classification_prompt},
                            ],
                        },
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "logprobs": True,
                    "top_logprobs": logprobs_top_k,
                }
                if self._is_thinking:
                    extra: dict = {"repetition_penalty": 1.2}
                    if temperature > 0:
                        extra["top_p"] = 0.9
                        extra["top_k"] = 0
                    kwargs["extra_body"] = extra
                return await self.client.chat.completions.create(**kwargs)

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_openai,
                context=f"VLLMModel.classify({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(f"vLLM classification API error: {e}") from e

        choice = response.choices[0]
        full_text = choice.message.content or ""
        logprobs_content = (
            choice.logprobs.content
            if choice.logprobs and choice.logprobs.content
            else []
        )

        if not self._is_thinking:
            # Non-thinking model: single-token extraction (like OpenAIVLM)
            token_logprobs = []
            for tlp in logprobs_content:
                token_logprobs.append(
                    TokenLogProb(
                        token=tlp.token,
                        logprob=tlp.logprob,
                        top_alternatives=[
                            (alt.token, alt.logprob)
                            for alt in (tlp.top_logprobs or [])
                            if alt.token != tlp.token
                        ],
                    )
                )
            return self._extract_ab_confidence(token_logprobs)

        # --- Thinking model: find A/B answer after </think> ---
        answer_idx = self._find_post_think_answer_from_logprobs(logprobs_content)

        if answer_idx is not None and answer_idx < len(logprobs_content):
            tlp = logprobs_content[answer_idx]
            token_logprobs = [
                TokenLogProb(
                    token=tlp.token,
                    logprob=tlp.logprob,
                    top_alternatives=[
                        (alt.token, alt.logprob)
                        for alt in (tlp.top_logprobs or [])
                        if alt.token != tlp.token
                    ],
                )
            ]
            logger.info(
                "Classification: found answer at token %d/%d",
                answer_idx,
                len(logprobs_content),
            )
        else:
            # No A/B token found — fall back to text-based prediction
            logger.warning(
                "Could not find A/B answer after </think> for %s "
                "(answer_idx=%s, n_logprobs=%d). "
                "Falling back to text-based prediction. "
                "Full text (last 200 chars): %r",
                self.model_name,
                answer_idx,
                len(logprobs_content),
                full_text[-200:],
            )
            parsed_prediction = self.parse_diagnosis(full_text)
            return LogprobClassificationResult(
                predicted_label=parsed_prediction,
                confidence=0.5,
                a_logprob=None,
                b_logprob=None,
                raw_first_token="<no_answer>",
                compliant=False,
                raw_response=full_text,
            )

        result = self._extract_ab_confidence(token_logprobs)
        result.raw_response = full_text

        # SC sampling: when temperature > 0, sample from A/B distribution
        # instead of argmax so self-consistency captures genuine uncertainty.
        if (
            temperature > 0
            and result.a_logprob is not None
            and result.b_logprob is not None
        ):
            logprobs_pair = np.array([result.a_logprob, result.b_logprob])
            max_lp = np.max(logprobs_pair)
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
                    temperature,
                    probs[0],
                    probs[1],
                    sampled,
                )
                result.predicted_label = sampled
                result.confidence = float(
                    probs[0] if sampled == a_label else probs[1]
                )
                result.compliant = True

        logger.info(
            "Classification result: predicted=%s, confidence=%.3f, "
            "compliant=%s, raw_first_token=%r",
            result.predicted_label,
            result.confidence,
            result.compliant,
            result.raw_first_token,
        )
        return result

    # ------------------------------------------------------------------
    # predict_confidence_elicitation()
    # ------------------------------------------------------------------

    async def predict_confidence_elicitation(
        self,
        image_base64: str,
        classification_prompt: str,
        predicted_label: str,
    ) -> str:
        """Multi-turn confidence elicitation."""
        ce_prompt = self.get_confidence_elicitation_prompt()

        # Richer assistant turn for thinking models
        if self._is_thinking:
            label_word = (
                self._task_config.a_label.capitalize()
                if predicted_label.upper().startswith("A")
                else self._task_config.b_label.capitalize()
            )
            assistant_content = f"{predicted_label}. {label_word}"
        else:
            assistant_content = predicted_label

        max_tokens = 1024 if self._is_thinking else 128

        async def _call():
            async with self._semaphore:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_instruction,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                    },
                                },
                                {"type": "text", "text": classification_prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        },
                        {
                            "role": "user",
                            "content": ce_prompt,
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    logprobs=False,
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_openai,
                context=f"VLLMModel.elicit({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(
                f"vLLM confidence elicitation API error: {e}"
            ) from e

        text = response.choices[0].message.content or ""
        logger.info(
            "Confidence elicitation response (last 200 chars): %r",
            text[-200:],
        )
        return text

    # ------------------------------------------------------------------
    # predict_batch()
    # ------------------------------------------------------------------

    async def predict_batch(
        self,
        images_base64: list[str],
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_logprobs: bool = True,
    ) -> list[VLMResponse]:
        tasks = [
            self.predict(img, prompt, temperature, max_tokens, return_logprobs)
            for img in images_base64
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in results:
            if isinstance(result, Exception):
                processed.append(
                    VLMResponse(
                        text=f"Error: {result}",
                        prediction="unknown",
                        model_name=self.model_name,
                    )
                )
            else:
                processed.append(result)

        return processed
