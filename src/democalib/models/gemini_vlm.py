"""Google Gemini Vision Language Model implementation using the google-genai SDK."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial

from .base import BaseVLM, LogprobClassificationResult, TokenLogProb, VLMResponse, retry_with_backoff

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 2.0
_MAX_DELAY = 120.0


def _is_retryable_gemini(exc: Exception) -> bool:
    """Return True for Gemini errors worth retrying (rate limits, transient server errors)."""
    msg = str(exc).lower()
    # Never retry client errors like 404 (model not found), 400 (bad request)
    if "404" in msg or "not_found" in msg or "400" in msg or "invalid" in msg:
        return False
    exc_type = type(exc).__name__
    if exc_type in (
        "ResourceExhausted",    # 429 rate limit
        "ServiceUnavailable",   # 503
        "InternalServerError",  # 500
        "DeadlineExceeded",     # timeout
        "Aborted",              # transient
    ):
        return True
    if any(kw in msg for kw in ("429", "rate limit", "quota", "resource exhausted", "503", "500")):
        return True
    return False


class GeminiVLM(BaseVLM):
    """Google Gemini Vision Language Model."""

    SUPPORTED_MODELS = [
        "gemini-pro-vision",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        rate_limit_rpm: int = 60,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(api_key, model_name, **kwargs)
        self.rate_limit_rpm = rate_limit_rpm
        self.timeout = timeout
        self._request_semaphore = asyncio.Semaphore(max(1, rate_limit_rpm // 60))

        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        ]

    def _image_part(self, image_base64: str):
        """Create an image Part from base64 data."""
        image_data = base64.b64decode(image_base64)
        return self._types.Part.from_bytes(data=image_data, mime_type="image/jpeg")

    def _parse_logprobs(self, response) -> list[TokenLogProb] | None:
        """Extract token logprobs from a Gemini response."""
        if not (hasattr(response, "candidates") and response.candidates):
            return None
        candidate = response.candidates[0]
        if not (hasattr(candidate, "logprobs_result") and candidate.logprobs_result):
            return None
        logprobs_result = candidate.logprobs_result
        if not (hasattr(logprobs_result, "chosen_candidates") and logprobs_result.chosen_candidates):
            return None

        token_logprobs = []
        for i, chosen in enumerate(logprobs_result.chosen_candidates):
            top_alts = []
            if (
                hasattr(logprobs_result, "top_candidates")
                and i < len(logprobs_result.top_candidates)
            ):
                top_alts = [
                    (alt.token, alt.log_probability)
                    for alt in logprobs_result.top_candidates[i].candidates
                    if alt.token != chosen.token
                ]
            token_logprobs.append(
                TokenLogProb(
                    token=chosen.token,
                    logprob=chosen.log_probability,
                    top_alternatives=top_alts,
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
        image_part = self._image_part(image_base64)
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": self.system_instruction,
            "safety_settings": self._safety_settings,
        }
        if return_logprobs:
            config_kwargs["response_logprobs"] = True
            config_kwargs["logprobs"] = logprobs_top_k
        config = self._types.GenerateContentConfig(**config_kwargs)

        async def _call():
            async with self._request_semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    partial(
                        self._client.models.generate_content,
                        model=self.model_name,
                        contents=[prompt, image_part],
                        config=config,
                    ),
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_gemini,
                context=f"Gemini.predict({self.model_name})",
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

        text = response.text
        token_logprobs = self._parse_logprobs(response) if return_logprobs else None
        prediction = self.parse_diagnosis(text)

        return VLMResponse(
            text=text,
            prediction=prediction,
            token_logprobs=token_logprobs,
            finish_reason="stop",
            model_name=self.model_name,
        )

    async def predict_classification(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """Constrained single-token A/B classification for logprob extraction."""
        image_part = self._image_part(image_base64)
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=logprobs_top_k,
            system_instruction=self.system_instruction,
            safety_settings=self._safety_settings,
        )

        async def _call():
            async with self._request_semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    partial(
                        self._client.models.generate_content,
                        model=self.model_name,
                        contents=[self.get_classification_prompt(), image_part],
                        config=config,
                    ),
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_gemini,
                context=f"Gemini.classify({self.model_name})",
            )
        except Exception as e:
            raise RuntimeError(f"Gemini classification API error: {e}") from e

        token_logprobs = self._parse_logprobs(response) or []
        return self._extract_ab_confidence(token_logprobs)

    async def predict_confidence_elicitation(
        self,
        image_base64: str,
        classification_prompt: str,
        predicted_label: str,
    ) -> str:
        """Multi-turn confidence elicitation via Gemini."""
        image_part = self._image_part(image_base64)
        config = self._types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=64,
            system_instruction=self.system_instruction,
            safety_settings=self._safety_settings,
        )

        # Build multi-turn history
        history = [
            self._types.Content(
                role="user",
                parts=[image_part, self._types.Part.from_text(text=classification_prompt)],
            ),
            self._types.Content(
                role="model",
                parts=[self._types.Part.from_text(text=predicted_label)],
            ),
            self._types.Content(
                role="user",
                parts=[self._types.Part.from_text(text=self.get_confidence_elicitation_prompt())],
            ),
        ]

        async def _call():
            async with self._request_semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    partial(
                        self._client.models.generate_content,
                        model=self.model_name,
                        contents=history,
                        config=config,
                    ),
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_gemini,
                context=f"Gemini.elicit({self.model_name})",
            )
        except Exception as e:
            raise RuntimeError(f"Gemini confidence elicitation API error: {e}") from e

        return response.text

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
