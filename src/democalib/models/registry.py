"""Model registry for dynamic model loading."""

from __future__ import annotations

from typing import Type

from .base import BaseVLM
from .gemini_vlm import GeminiVLM
from .openai_vlm import OpenAIVLM

# Lazy import for LocalVLM to avoid torch dependency when not needed
_LocalVLM = None


def _get_local_vlm():
    """Lazy load LocalVLM to avoid torch dependency at import time."""
    global _LocalVLM
    if _LocalVLM is None:
        from .local_vlm import LocalVLM
        _LocalVLM = LocalVLM
    return _LocalVLM


# Registry mapping short names to model classes
MODEL_REGISTRY: dict[str, Type[BaseVLM] | str] = {
    # API-based models
    "gpt-4v": OpenAIVLM,
    "gpt-4o": OpenAIVLM,
    "gpt-4o-mini": OpenAIVLM,
    "gpt-4.1": OpenAIVLM,
    "gemini-pro-vision": GeminiVLM,
    "gemini-1.5-pro": GeminiVLM,
    "gemini-1.5-flash": GeminiVLM,
    "gemini-2.0-flash": GeminiVLM,
    "gemini-2.5-pro": GeminiVLM,
    "gemini-2.5-flash": GeminiVLM,
    # Local reasoning models (use string marker for lazy loading)
    "r1-onevision-7b": "local",
    "glm-4.1v-9b-thinking": "local",
    "kimi-vl-a3b-thinking": "local",
    "openflamingo-3b": "local",
    "llama-3.2-11b-vision": "local",
    "qwen2.5-vl-7b": "local",
    "medgemma-4b": "local",
    # vLLM-served models
    "r1-onevision-7b-vllm": "vllm",
}

# Mapping from short names to actual API/HuggingFace model names
MODEL_NAME_MAPPING: dict[str, str] = {
    # API models
    "gpt-4v": "gpt-4-vision-preview",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "gemini-pro-vision": "gemini-pro-vision",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    # Local models (HuggingFace IDs)
    "r1-onevision-7b": "Fancy-MLLM/R1-Onevision-7B",
    "glm-4.1v-9b-thinking": "zai-org/GLM-4.1V-9B-Thinking",
    "kimi-vl-a3b-thinking": "moonshotai/Kimi-VL-A3B-Thinking-2506",
    "openflamingo-3b": "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
    "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "medgemma-4b": "google/medgemma-4b-it",
    # vLLM-served
    "r1-onevision-7b-vllm": "Fancy-MLLM/R1-Onevision-7B",
}

# Models that don't require API keys (run locally)
LOCAL_MODELS = {"r1-onevision-7b", "glm-4.1v-9b-thinking", "kimi-vl-a3b-thinking", "openflamingo-3b", "llama-3.2-11b-vision", "qwen2.5-vl-7b", "medgemma-4b"}


def get_model(
    model_name: str,
    api_key: str = "",
    **kwargs,
) -> BaseVLM:
    """
    Get model instance from registry.

    Args:
        model_name: Short model name (e.g., "gpt-4o", "gemini-1.5-pro", "r1-onevision-7b")
        api_key: API key for the model (not required for local models)
        **kwargs: Additional model configuration (rate_limit_rpm, timeout, etc.)

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model name is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    model_class = MODEL_REGISTRY[model_name]
    actual_model_name = MODEL_NAME_MAPPING[model_name]

    # Handle lazy loading for local models
    if model_class == "local":
        model_class = _get_local_vlm()
    elif model_class == "vllm":
        from .vllm_model import VLLMModel
        model_class = VLLMModel

    return model_class(
        api_key=api_key,
        model_name=actual_model_name,
        **kwargs,
    )


def register_model(
    name: str,
    model_class: Type[BaseVLM],
    actual_name: str | None = None,
) -> None:
    """
    Register a new model in the registry.

    Args:
        name: Short name for the model
        model_class: Model class (must inherit from BaseVLM)
        actual_name: Actual API model name (defaults to short name)
    """
    MODEL_REGISTRY[name] = model_class
    MODEL_NAME_MAPPING[name] = actual_name or name


def list_models() -> list[str]:
    """
    List available model names.

    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())
