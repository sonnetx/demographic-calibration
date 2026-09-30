"""Abstract base class for Vision-Language Models."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .task_config import TaskConfig

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    coro_fn,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_check=None,
    context: str = "",
):
    """Retry an async callable with exponential backoff and jitter.

    Args:
        coro_fn: Zero-argument async callable (lambda producing a coroutine).
        max_retries: Maximum number of retries (0 = no retries).
        base_delay: Initial delay in seconds.
        max_delay: Cap on delay between retries.
        retryable_check: Optional callable(exception) -> bool. If provided,
            only retry when it returns True.  Defaults to retrying on any
            Exception.
        context: Short label for log messages (e.g. "predict" or
            "predict_classification").

    Returns:
        The result of coro_fn().

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            last_exc = exc
            if retryable_check and not retryable_check(exc):
                raise
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            wait = delay + jitter
            logger.warning(
                "[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                context or "retry",
                attempt + 1,
                max_retries + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise last_exc  # unreachable, but keeps type checkers happy


def strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> and <pre>...</pre> blocks from reasoning model output.

    If all content is inside thinking tags (nothing meaningful after the closing tag),
    falls back to using the content inside the tags so answers aren't lost.
    R1 sometimes emits </pre> instead of </think>.
    """
    # Handle both <think> and <pre> tags (R1 sometimes uses </pre>)
    stripped = re.sub(r"<(?:think|pre)>.*?</(?:think|pre)>", "", text, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    # All content was inside tags — extract it rather than returning empty
    match = re.search(r"<(?:think|pre)>(.*?)</(?:think|pre)>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # No think/pre tags at all
    return text.strip()


@dataclass
class TokenLogProb:
    """Single token with its log probability."""

    token: str
    logprob: float
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class VLMResponse:
    """Response from a VLM model."""

    text: str
    prediction: str  # predicted label (e.g. "benign", "malignant", "present", "absent")
    token_logprobs: list[TokenLogProb] | None = None
    finish_reason: str = "stop"
    model_name: str = ""

    @property
    def has_logprobs(self) -> bool:
        """Check if log probabilities are available."""
        return self.token_logprobs is not None and len(self.token_logprobs) > 0

    def _find_answer_tokens(self) -> list["TokenLogProb"]:
        """Find all answer tokens after 'Answer:' marker until newline."""
        if not self.has_logprobs:
            return []

        current_text = ""
        answer_tokens = []
        collecting = False

        for i, tlp in enumerate(self.token_logprobs):
            if not collecting:
                current_text += tlp.token
                if "answer:" in current_text.lower():
                    collecting = True
                continue
            if collecting:
                if "\n" in tlp.token:
                    break
                if tlp.token.strip() and tlp.logprob > -100:
                    answer_tokens.append(tlp)

        return answer_tokens

    def _find_alt_token_logprob(
        self,
        first_token: "TokenLogProb",
        a_label: str = "benign",
        b_label: str = "malignant",
    ) -> tuple[float, str]:
        """Find the alternative choice's logprob from top-k alternatives.

        Returns (logprob, token_text) for the alternative label.
        Falls back to the minimum top-k logprob if not found.
        """
        predicted_lower = self.prediction.lower()
        a_prefix = a_label[:3].lower()
        b_prefix = b_label[:3].lower()

        for alt_token, alt_logprob in first_token.top_alternatives:
            alt_lower = alt_token.strip().lower()
            if predicted_lower == a_label and alt_lower.startswith(b_prefix):
                return alt_logprob, alt_token
            elif predicted_lower == b_label and alt_lower.startswith(a_prefix):
                return alt_logprob, alt_token

        # Fallback: use the minimum top-k logprob as bound
        min_alt_logprob = first_token.logprob
        min_alt_token = first_token.token
        for tok, alt_lp in first_token.top_alternatives:
            if alt_lp < min_alt_logprob:
                min_alt_logprob = alt_lp
                min_alt_token = tok
        return min_alt_logprob, min_alt_token

    def _softmax_with_alt(
        self,
        generated_norm_logprob: float,
        alt_norm_logprob: float,
    ) -> float:
        """Compute softmax P(predicted) over two normalized logprobs."""
        logprobs_pair = np.array([generated_norm_logprob, alt_norm_logprob])
        max_lp = np.max(logprobs_pair)
        exp_lps = np.exp(logprobs_pair - max_lp)
        probs = exp_lps / np.sum(exp_lps)
        return float(probs[0])

    @staticmethod
    def _normalize_logprob(
        total_logprob: float,
        text: str,
        n_tokens: int,
        normalization: str,
    ) -> float:
        if normalization == "token":
            return total_logprob / max(n_tokens, 1)
        elif normalization == "byte":
            return total_logprob / max(len(text.encode("utf-8")), 1)
        return total_logprob

    def get_diagnosis_logprob(
        self,
        normalization: str = "byte",
        a_label: str = "benign",
        b_label: str = "malignant",
    ) -> float | None:
        """
        Extract forced-choice probability for the predicted diagnosis.

        Collects all answer tokens after "Answer:" and sums their logprobs,
        then normalizes to be comparable with the alternative choice's
        logprob. At the first answer token, also checks top alternatives
        to find the competing choice's logprob for softmax normalization.

        Args:
            normalization: One of "byte", "token", "none".
            a_label: Label for choice A (e.g. "benign", "present").
            b_label: Label for choice B (e.g. "malignant", "absent").

        Returns:
            Confidence score in [0, 1], or None if logprobs unavailable
        """
        if not self.has_logprobs:
            return None

        answer_tokens = self._find_answer_tokens()
        if not answer_tokens:
            return None

        generated_total_logprob = sum(t.logprob for t in answer_tokens)
        answer_text = "".join(t.token for t in answer_tokens)
        generated_norm = self._normalize_logprob(
            generated_total_logprob, answer_text, len(answer_tokens), normalization,
        )

        first_token = answer_tokens[0]
        alt_logprob, alt_text = self._find_alt_token_logprob(
            first_token, a_label, b_label,
        )
        alt_norm = self._normalize_logprob(alt_logprob, alt_text, 1, normalization)

        return self._softmax_with_alt(generated_norm, alt_norm)

    def get_min_diagnosis_logprob(
        self,
        normalization: str = "byte",
        a_label: str = "benign",
        b_label: str = "malignant",
    ) -> float | None:
        """
        Extract forced-choice probability using the minimum token logprob
        across answer tokens instead of the mean.

        Args:
            normalization: One of "byte", "token", "none".
            a_label: Label for choice A (e.g. "benign", "present").
            b_label: Label for choice B (e.g. "malignant", "absent").

        Returns:
            Confidence score in [0, 1], or None if logprobs unavailable
        """
        if not self.has_logprobs:
            return None

        answer_tokens = self._find_answer_tokens()
        if not answer_tokens:
            return None

        min_token = min(answer_tokens, key=lambda t: t.logprob)
        generated_norm = self._normalize_logprob(
            min_token.logprob, min_token.token, 1, normalization,
        )

        first_token = answer_tokens[0]
        alt_logprob, alt_text = self._find_alt_token_logprob(
            first_token, a_label, b_label,
        )
        alt_norm = self._normalize_logprob(alt_logprob, alt_text, 1, normalization)

        return self._softmax_with_alt(generated_norm, alt_norm)


@dataclass
class LogprobClassificationResult:
    """Result from constrained single-token A/B classification for logprob extraction."""

    predicted_label: str  # "benign" or "malignant"
    confidence: float  # softmax P(predicted) over {A, B}
    a_logprob: float | None  # raw logprob for "A" token
    b_logprob: float | None  # raw logprob for "B" token
    raw_first_token: str  # actual first token generated
    compliant: bool  # whether first token was A or B
    raw_response: str = ""  # full model output (thinking + answer for thinking models)


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""

    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize VLM.

        Args:
            api_key: API key for the model service
            model_name: Name of the model to use
            **kwargs: Additional model configuration
        """
        self.api_key = api_key
        self.model_name = model_name
        self.kwargs = kwargs
        self._task_config = TaskConfig.ddi()
        self.system_instruction = self._task_config.system_instruction

    def set_task_config(self, task_config: TaskConfig) -> None:
        """Configure the model for a specific classification task."""
        self._task_config = task_config
        self.system_instruction = task_config.system_instruction

    @abstractmethod
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
        Make a prediction on an image.

        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt for the model
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum response tokens
            return_logprobs: Whether to request log probabilities
            logprobs_top_k: Number of top alternative tokens to return

        Returns:
            VLMResponse with prediction and optional logprobs
        """
        pass

    @abstractmethod
    async def predict_batch(
        self,
        images_base64: list[str],
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_logprobs: bool = True,
    ) -> list[VLMResponse]:
        """
        Batch prediction with rate limiting.

        Args:
            images_base64: List of base64-encoded images
            prompt: Text prompt for all images
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            return_logprobs: Whether to request log probabilities

        Returns:
            List of VLMResponse objects
        """
        pass

    @property
    def is_reasoning_model(self) -> bool:
        """Whether this model produces chain-of-thought reasoning.

        Reasoning models generate extended thinking traces, making embedding
        consistency meaningful. Non-reasoning models produce short structured
        responses where embedding consistency collapses to label agreement.
        """
        return False

    @abstractmethod
    async def predict_classification(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """
        Constrained single-token classification for logprob confidence extraction.

        Makes a separate API call with a minimal A/B prompt and max_tokens=1,
        so the first generated token is the classification label. Logprobs at
        position 0 reflect genuine uncertainty, not autoregressive conditioning.

        Args:
            image_base64: Base64-encoded image
            logprobs_top_k: Number of top alternative tokens to request
            temperature: Sampling temperature (0.0 for greedy, >0 for SC)

        Returns:
            LogprobClassificationResult with confidence from A/B softmax
        """
        pass

    @abstractmethod
    async def predict_confidence_elicitation(
        self,
        image_base64: str,
        classification_prompt: str,
        predicted_label: str,
    ) -> str:
        """
        Multi-turn confidence elicitation: resubmit question + answer, ask for confidence.

        Note: The verbalized confidence from this call is the model assessing
        confidence for an answer produced via constrained single-token output,
        not through its normal reasoning process. This may measure something
        different than free-form verbalized confidence.

        Args:
            image_base64: Base64-encoded image
            classification_prompt: The classification prompt used in step 1
            predicted_label: The predicted label letter ("A" or "B")

        Returns:
            Raw text from the confidence elicitation response
        """
        pass

    def get_diagnosis_prompt(self) -> str:
        """Freeform diagnosis prompt (delegates to task config)."""
        return self._task_config.diagnosis_prompt

    def get_classification_prompt(self) -> str:
        """Minimal prompt for constrained A/B classification (logprob extraction)."""
        return self._task_config.classification_prompt

    def get_confidence_elicitation_prompt(self) -> str:
        """Follow-up prompt for multi-turn confidence elicitation."""
        return self._task_config.confidence_elicitation_prompt

    def _extract_ab_confidence(
        self,
        token_logprobs: list[TokenLogProb],
    ) -> LogprobClassificationResult:
        """Extract A/B classification confidence from first-token logprobs.

        Scans the first token and its top_alternatives for "A" and "B",
        applies softmax over those two logprobs to produce confidence.
        Maps A→a_label, B→b_label (from task config).
        """
        a_label = self._task_config.a_label
        b_label = self._task_config.b_label

        if not token_logprobs:
            return LogprobClassificationResult(
                predicted_label="unknown",
                confidence=0.5,
                a_logprob=None,
                b_logprob=None,
                raw_first_token="",
                compliant=False,
            )

        first = token_logprobs[0]
        raw_first = first.token.strip()

        # Collect all candidate logprobs: the chosen token + alternatives
        all_candidates = [(first.token.strip(), first.logprob)]
        all_candidates.extend(
            (alt_tok.strip(), alt_lp)
            for alt_tok, alt_lp in first.top_alternatives
        )

        a_lp = None
        b_lp = None
        for tok, lp in all_candidates:
            if tok.upper() == "A" and a_lp is None:
                a_lp = lp
            elif tok.upper() == "B" and b_lp is None:
                b_lp = lp

        # Determine predicted label from the actual first token
        if raw_first.upper() == "A":
            predicted = a_label
            compliant = True
        elif raw_first.upper() == "B":
            predicted = b_label
            compliant = True
        else:
            predicted = "unknown"
            compliant = False

        # Compute confidence via softmax over A and B logprobs
        if a_lp is not None and b_lp is not None:
            logprobs_pair = np.array([a_lp, b_lp])
            max_lp = np.max(logprobs_pair)
            # Guard against both logprobs being -inf (inf - inf = NaN)
            if not np.isfinite(max_lp):
                a_lp = None
                b_lp = None
                confidence = 0.5
            else:
                exp_lps = np.exp(logprobs_pair - max_lp)
                probs = exp_lps / np.sum(exp_lps)
                if predicted == a_label:
                    confidence = float(probs[0])
                elif predicted == b_label:
                    confidence = float(probs[1])
                else:
                    # Non-compliant but both logprobs found: use the higher one
                    if probs[0] >= probs[1]:
                        predicted = a_label
                        confidence = float(probs[0])
                    else:
                        predicted = b_label
                        confidence = float(probs[1])
        elif a_lp is not None:
            predicted = predicted if predicted != "unknown" else a_label
            confidence = 1.0
        elif b_lp is not None:
            predicted = predicted if predicted != "unknown" else b_label
            confidence = 1.0
        else:
            confidence = 0.5

        return LogprobClassificationResult(
            predicted_label=predicted,
            confidence=confidence,
            a_logprob=a_lp,
            b_logprob=b_lp,
            raw_first_token=first.token,
            compliant=compliant,
        )

    def parse_diagnosis(self, text: str) -> str:
        """
        Parse diagnosis from model response.

        Uses a_label/b_label from task config (e.g. "benign"/"malignant"
        or "present"/"absent").

        Args:
            text: Full model response text

        Returns:
            Parsed label or "unknown"
        """
        a_label = self._task_config.a_label
        b_label = self._task_config.b_label

        # Strip thinking blocks from reasoning models (R1, GLM, Kimi)
        text = strip_thinking_tags(text)
        text_lower = text.lower()

        # Look for "Answer:" format
        if "answer:" in text_lower:
            after_answer = text_lower.split("answer:")[-1]
            first_line = after_answer.split("\n")[0].strip()
            if a_label in first_line:
                return a_label
            elif b_label in first_line:
                return b_label
            # Check for choice letters (A / B)
            cleaned = re.sub(r"[\[\]\(\)\{\}]", "", first_line).strip()
            if re.match(r"^a\b", cleaned):
                return a_label
            elif re.match(r"^b\b", cleaned):
                return b_label

        # Look for explicit diagnosis line
        if "diagnosis:" in text_lower:
            after_diagnosis = text_lower.split("diagnosis:")[-1]
            first_line = after_diagnosis.split("\n")[0]
            if b_label in first_line:
                return b_label
            elif a_label in first_line:
                return a_label

        # Check for choice letters anywhere
        if re.search(r"\banswer\b.*\b[Bb]\b", text) and b_label not in text_lower:
            return b_label
        if re.search(r"\banswer\b.*\b[Aa]\b", text) and a_label not in text_lower:
            return a_label

        # Fallback: check which appears more prominently
        b_count = text_lower.count(b_label)
        a_count = text_lower.count(a_label)

        if b_count > a_count:
            return b_label
        elif a_count > b_count:
            return a_label

        # Last resort: scan for medical synonym patterns in the conclusion
        # (last ~500 chars where the conclusion likely is)
        conclusion = text_lower[-500:] if len(text_lower) > 500 else text_lower

        # Negative indicators (absent/benign direction)
        neg_patterns = [
            r"\bno\s+(?:evidence|signs?|findings?|indication)\b",
            r"\bnormal\b", r"\bnormalcy\b", r"\bunremarkable\b",
            r"\bnegative\b", r"\bno\s+pneumonia\b", r"\bnot\s+(?:present|detected|found)\b",
            r"\brule\s*out\b", r"\bbenign\b",
            r"\babsence\b", r"\babsent\b",
        ]
        # Positive indicators (present/malignant direction)
        pos_patterns = [
            r"\bconsistent\s+with\s+(?:pneumonia|malignant|malignancy)\b",
            r"\bpositive\b", r"\bdetected\b", r"\bconfirmed\b",
            r"\bsuggestive\s+of\b", r"\bindicative\s+of\b",
            r"\bpneumonia\s+(?:is\s+)?(?:likely|probable|identified|found|seen)\b",
            r"\bmalignant\b",
            r"\bpresence\b", r"\bpresent\b",
        ]

        neg_score = sum(1 for p in neg_patterns if re.search(p, conclusion))
        pos_score = sum(1 for p in pos_patterns if re.search(p, conclusion))

        if neg_score > pos_score:
            return b_label
        elif pos_score > neg_score:
            return a_label

        return "unknown"
