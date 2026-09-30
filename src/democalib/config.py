"""Configuration schema and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Data configuration."""

    dataset_type: Literal["ddi", "chexpert"] = Field(
        "ddi", description="Dataset type"
    )
    metadata_path: Path = Field(..., description="Path to metadata CSV")
    images_dir: Path = Field(..., description="Path to images directory")
    random_seed: int = Field(42, description="Random seed for reproducibility")

    # DDI-specific (kept for backward compatibility)
    skin_tone_groups: Optional[Dict[str, List[int]]] = Field(
        None, description="Mapping of group names to FST codes (DDI only)"
    )

    # Generic demographic groups
    demographic_groups: Optional[Dict[str, List[Any]]] = Field(
        None, description="Mapping of group names to demographic attribute values"
    )

    # CheXpert-specific
    condition: Optional[str] = Field(
        None, description="Condition column for binary classification (CheXpert)"
    )

    @model_validator(mode="after")
    def resolve_groups(self) -> "DataConfig":
        """Populate demographic_groups from legacy skin_tone_groups if needed."""
        if self.demographic_groups is None and self.skin_tone_groups is not None:
            self.demographic_groups = self.skin_tone_groups
        if self.dataset_type == "ddi" and self.demographic_groups is None:
            self.demographic_groups = {"light": [12], "medium": [34], "dark": [56]}
            self.skin_tone_groups = self.demographic_groups
        if self.dataset_type == "chexpert" and self.demographic_groups is None:
            self.demographic_groups = {"male": ["Male"], "female": ["Female"]}
        if self.dataset_type == "chexpert" and self.condition is None:
            self.condition = "Pneumonia"
        return self


class ModelConfig(BaseModel):
    """Model configuration."""

    name: Literal[
        # API-based models
        "gpt-4v", "gpt-4o", "gpt-4o-mini", "gpt-4.1",
        "gpt-5.1-2025-11-13", "gpt-5.2-2025-12-11",
        "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash",
        "gemini-2.0-flash", "gemini-2.5-pro", "gemini-2.5-flash",
        # Local reasoning models
        "r1-onevision-7b", "glm-4.1v-9b-thinking", "kimi-vl-a3b-thinking",
        "openflamingo-3b",
        "llama-3.2-11b-vision",
        "qwen2.5-vl-7b",
        "medgemma-4b",
        # vLLM-served models
        "r1-onevision-7b-vllm",
    ] = Field(...)
    api_key_env: Optional[str] = Field(
        None, description="Environment variable name for API key (auto-detected from model name if not set)"
    )
    base_url: Optional[str] = Field(
        None, description="Base URL for vLLM/OpenAI-compatible server (e.g., http://localhost:8000/v1)"
    )
    max_tokens: int = Field(256, description="Maximum response tokens")
    timeout: float = Field(60.0, description="API timeout in seconds")
    rate_limit_rpm: int = Field(60, description="Rate limit requests per minute")

    # Local models that don't require API keys
    _LOCAL_MODELS = {"r1-onevision-7b", "glm-4.1v-9b-thinking", "kimi-vl-a3b-thinking", "openflamingo-3b", "llama-3.2-11b-vision", "qwen2.5-vl-7b", "medgemma-4b", "r1-onevision-7b-vllm"}

    def is_local_model(self) -> bool:
        """Check if this model runs locally (no API key needed)."""
        return self.name in self._LOCAL_MODELS

    def get_api_key_env(self) -> Optional[str]:
        """Get the environment variable name for the API key, or None for local models."""
        if self.is_local_model():
            return None
        if self.api_key_env:
            return self.api_key_env
        # Auto-detect based on model name
        if self.name.startswith("gemini"):
            # Prefer GEMINI_API_KEY if set, fall back to GOOGLE_API_KEY
            import os
            if os.environ.get("GEMINI_API_KEY"):
                return "GEMINI_API_KEY"
            return "GOOGLE_API_KEY"
        else:
            return "OPENAI_API_KEY"


class PromptOverrideConfig(BaseModel):
    """Optional overrides for the task prompts (prompt-sensitivity experiments).

    Any field left unset falls back to the default TaskConfig for the dataset.
    If the A/B option order is changed in classification_prompt, a_label and
    b_label must be updated to match.
    """

    system_instruction: Optional[str] = Field(None, description="System instruction override")
    classification_prompt: Optional[str] = Field(None, description="A/B classification prompt override")
    diagnosis_prompt: Optional[str] = Field(None, description="Single-turn diagnosis prompt override")
    confidence_elicitation_prompt: Optional[str] = Field(None, description="CE second-turn prompt override")
    a_label: Optional[str] = Field(None, description="Label meaning of option A")
    b_label: Optional[str] = Field(None, description="Label meaning of option B")


class ConfidenceConfig(BaseModel):
    """Confidence extraction configuration."""

    extract_logprob: bool = Field(True, description="Extract log-probability confidence")
    extract_verbalized: bool = Field(True, description="Extract verbalized confidence")
    extract_self_consistency: bool = Field(True, description="Extract self-consistency confidence")
    extract_embedding_consistency: bool = Field(
        False, description="Extract embedding-based sample consistency (cosine similarity)"
    )
    self_consistency_k: int = Field(
        15, ge=2, le=50, description="Number of samples for self-consistency / embedding consistency"
    )
    self_consistency_temperatures: List[float] = Field(
        [0.5, 1.0], description="Temperatures for stochastic sampling"
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="Embedding model for sample consistency (e.g., text-embedding-3-small, text-embedding-3-large)",
    )
    logprob_top_k: int = Field(20, ge=1, le=20, description="Top-k logprobs to request")
    logprob_normalization: Literal["byte", "token", "none"] = Field(
        "byte",
        description=(
            "Normalization for freeform logprob confidence. "
            "'byte': divide by UTF-8 byte length (tokenization-agnostic, lm-eval-harness style). "
            "'token': divide by token count (geometric mean). "
            "'none': raw sum of logprobs (favours shorter sequences)."
        ),
    )
    extract_logprob_freeform: bool = Field(
        False,
        description="Also extract freeform logprob baseline (degenerate, for comparison)",
    )


class CalibrationConfig(BaseModel):
    """Calibration configuration."""

    n_bins: int = Field(10, ge=5, le=20, description="Number of bins for ECE")
    bootstrap_n_samples: int = Field(1000, ge=100, description="Bootstrap iterations")
    bootstrap_ci_level: float = Field(0.95, ge=0.8, le=0.99, description="Confidence interval level")
    temperature_scaling_lr: float = Field(0.01, description="Learning rate for temperature optimization")
    temperature_scaling_max_iter: int = Field(100, description="Max iterations for temperature optimization")


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: Path = Field(Path("outputs"), description="Output directory")
    save_raw_predictions: bool = Field(True, description="Save raw model predictions")
    generate_figures: bool = Field(True, description="Generate reliability diagrams")
    figure_format: Literal["png", "pdf", "svg"] = Field("png", description="Figure format")
    figure_dpi: int = Field(300, description="Figure DPI")


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    experiment_name: str = Field(..., description="Experiment name for output organization")
    data: DataConfig
    model: ModelConfig
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts: Optional[PromptOverrideConfig] = Field(
        None, description="Optional task-prompt overrides for prompt-sensitivity experiments"
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
