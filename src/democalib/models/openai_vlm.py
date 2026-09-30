"""OpenAI GPT-4V/GPT-4o Vision-Language Model implementation."""

from __future__ import annotations

import asyncio

import openai
from openai import AsyncOpenAI

from .base import BaseVLM, LogprobClassificationResult, TokenLogProb, VLMResponse, retry_with_backoff

import logging

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 2.0
_MAX_DELAY = 120.0


def _is_retryable_openai(exc: Exception) -> bool:
    """Return True for OpenAI errors worth retrying (rate limits, transient server errors)."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (429, 500, 502, 503):
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("429", "rate limit", "quota", "503", "500")):
        return True
    return False


class OpenAIVLM(BaseVLM):
    """OpenAI GPT-4V/GPT-4o Vision-Language Model."""

    SUPPORTED_MODELS = [
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.2-2025-12-11",
    ]

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        rate_limit_rpm: int = 60,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        Initialize OpenAI VLM.

        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., "gpt-4o", "gpt-4.1")
            rate_limit_rpm: Rate limit in requests per minute
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model_name, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.rate_limit_rpm = rate_limit_rpm
        # Semaphore for rate limiting (allow rate_limit_rpm/60 concurrent requests)
        self._request_semaphore = asyncio.Semaphore(max(1, rate_limit_rpm // 60))

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
        Make prediction using OpenAI Vision API.

        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            return_logprobs: Whether to request log probabilities
            logprobs_top_k: Number of top alternative tokens

        Returns:
            VLMResponse with prediction and logprobs
        """
        async def _call():
            async with self._request_semaphore:
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
                                        "detail": "high",
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=return_logprobs,
                    top_logprobs=logprobs_top_k if return_logprobs else None,
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_openai,
                context=f"OpenAI.predict({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

        choice = response.choices[0]
        text = choice.message.content or ""

        # Parse logprobs if available
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

        # Extract diagnosis from response
        prediction = self.parse_diagnosis(text)

        return VLMResponse(
            text=text,
            prediction=prediction,
            token_logprobs=token_logprobs,
            finish_reason=choice.finish_reason or "stop",
            model_name=self.model_name,
        )

    async def predict_classification(
        self,
        image_base64: str,
        logprobs_top_k: int = 20,
        temperature: float = 0.0,
    ) -> LogprobClassificationResult:
        """Constrained single-token A/B classification for logprob extraction."""
        async def _call():
            async with self._request_semaphore:
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
                                        "detail": "high",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": self.get_classification_prompt(),
                                },
                            ],
                        },
                    ],
                    max_tokens=1,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=logprobs_top_k,
                )

        try:
            response = await retry_with_backoff(
                _call,
                max_retries=_MAX_RETRIES,
                base_delay=_BASE_DELAY,
                max_delay=_MAX_DELAY,
                retryable_check=_is_retryable_openai,
                context=f"OpenAI.classify({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI classification API error: {e}") from e

        choice = response.choices[0]

        token_logprobs = []
        if choice.logprobs and choice.logprobs.content:
            for tlp in choice.logprobs.content:
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

    async def predict_confidence_elicitation(
        self,
        image_base64: str,
        classification_prompt: str,
        predicted_label: str,
    ) -> str:
        """Multi-turn confidence elicitation: resubmit Q+A, ask for confidence."""
        async def _call():
            async with self._request_semaphore:
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
                                        "detail": "high",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": classification_prompt,
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": predicted_label,
                        },
                        {
                            "role": "user",
                            "content": self.get_confidence_elicitation_prompt(),
                        },
                    ],
                    max_tokens=64,
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
                context=f"OpenAI.elicit({self.model_name})",
            )
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI confidence elicitation API error: {e}") from e

        return response.choices[0].message.content or ""

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
            List of VLMResponse objects (may include exceptions)
        """
        tasks = [
            self.predict(img, prompt, temperature, max_tokens, return_logprobs)
            for img in images_base64
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed responses
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
