"""Confidence extraction aggregator."""

from __future__ import annotations

from dataclasses import dataclass

import logging

from ..data.dataset import DDISample, Sample
from ..models.base import BaseVLM, VLMResponse
from .embedding_consistency import compute_embedding_consistency
from .logprob import (
    extract_constrained_logprob_confidence,
    extract_logprob_confidence,
    extract_min_logprob_confidence,
)
from .self_consistency import compute_self_consistency_confidence
from .verbalized import extract_verbalized_confidence

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Aggregated confidence results for a single sample."""

    sample_id: str
    prediction: str
    ground_truth: str
    demographic_group: str

    # Confidence signals
    logprob_confidence: float | None  # NEW: constrained A/B logprob
    min_logprob_confidence: float | None  # same as logprob_confidence for single token
    verbalized_confidence: float | None
    # Per-temperature self-consistency: {temp: (prediction, confidence)}
    self_consistency_results: dict[float, tuple[str, float]] | None
    # Embedding-based sample consistency: {temp: cosine_similarity}
    embedding_consistency_results: dict[float, float] | None

    # Raw response for debugging
    raw_response: str

    # Freeform logprob baseline (degenerate, for methodological comparison)
    logprob_confidence_freeform: float | None = None
    min_logprob_confidence_freeform: float | None = None
    # Whether the constrained A/B call produced a compliant first token
    logprob_compliant: bool | None = None
    # Full classification response (thinking + answer for thinking models)
    raw_classification_response: str = ""

    @property
    def is_correct(self) -> bool:
        """Check if prediction matches ground truth."""
        return self.prediction == self.ground_truth

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "demographic_group": self.demographic_group,
            "is_correct": self.is_correct,
            "logprob_confidence": self.logprob_confidence,
            "min_logprob_confidence": self.min_logprob_confidence,
            "verbalized_confidence": self.verbalized_confidence,
            "logprob_confidence_freeform": self.logprob_confidence_freeform,
            "min_logprob_confidence_freeform": self.min_logprob_confidence_freeform,
            "logprob_compliant": self.logprob_compliant,
            "raw_classification_response": self.raw_classification_response,
            "raw_response": self.raw_response,
        }
        if self.self_consistency_results:
            for temp, (pred, conf) in self.self_consistency_results.items():
                d[f"sc_prediction_t{temp}"] = pred
                d[f"sc_confidence_t{temp}"] = conf
        if self.embedding_consistency_results:
            for temp, sim in self.embedding_consistency_results.items():
                d[f"emb_consistency_t{temp}"] = sim
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceResult":
        """Create from dictionary."""
        # Reconstruct per-temperature SC results from flat keys
        sc_results: dict[float, tuple[str, float]] = {}
        for key in data:
            if key.startswith("sc_prediction_t"):
                temp = float(key.removeprefix("sc_prediction_t"))
                pred = data[key]
                conf = data.get(f"sc_confidence_t{temp}")
                if pred is not None and conf is not None:
                    sc_results[temp] = (pred, conf)

        # Handle flat self_consistency_confidence key (from simplified runs)
        if not sc_results and "self_consistency_confidence" in data:
            sc_pred = data.get("self_consistency_prediction", data.get("prediction"))
            sc_conf = data["self_consistency_confidence"]
            if sc_pred is not None and sc_conf is not None:
                sc_results[0.0] = (sc_pred, sc_conf)

        # Reconstruct embedding consistency results
        emb_results: dict[float, float] = {}
        for key in data:
            if key.startswith("emb_consistency_t"):
                temp = float(key.removeprefix("emb_consistency_t"))
                emb_results[temp] = data[key]

        return cls(
            sample_id=data["sample_id"],
            prediction=data["prediction"],
            ground_truth=data["ground_truth"],
            demographic_group=data.get("demographic_group", data.get("skin_tone_group", "unknown")),
            logprob_confidence=data.get("logprob_confidence"),
            min_logprob_confidence=data.get("min_logprob_confidence"),
            verbalized_confidence=data.get("verbalized_confidence"),
            self_consistency_results=sc_results or None,
            embedding_consistency_results=emb_results or None,
            logprob_confidence_freeform=data.get("logprob_confidence_freeform"),
            min_logprob_confidence_freeform=data.get("min_logprob_confidence_freeform"),
            logprob_compliant=data.get("logprob_compliant"),
            raw_classification_response=data.get("raw_classification_response", ""),
            raw_response=data.get("raw_response", ""),
        )


class ConfidenceExtractor:
    """Extract all confidence signals from VLM responses."""

    def __init__(
        self,
        model: BaseVLM,
        extract_logprob: bool = True,
        extract_verbalized: bool = True,
        extract_self_consistency: bool = True,
        extract_embedding_consistency: bool = False,
        self_consistency_k: int = 10,
        self_consistency_temperatures: list[float] | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        logprob_top_k: int = 5,
        extract_logprob_freeform: bool = True,
        logprob_normalization: str = "byte",
    ):
        """
        Initialize confidence extractor.

        Args:
            model: VLM model instance
            extract_logprob: Whether to extract log-probability confidence (TLP)
            extract_verbalized: Whether to extract verbalized confidence (CE)
            extract_self_consistency: Whether to extract self-consistency confidence
            extract_embedding_consistency: Whether to extract embedding-based
                sample consistency (cosine similarity across K embedded responses)
            self_consistency_k: Number of samples for self-consistency / embedding consistency
            self_consistency_temperatures: Temperatures for stochastic sampling
            embedding_model: Embedding model name for embedding consistency
            embedding_api_key: API key for the embedding model
            logprob_top_k: Number of top alternative tokens to request
            extract_logprob_freeform: Also extract freeform logprob baseline (degenerate)
        """
        self.model = model
        self.extract_logprob = extract_logprob
        self.extract_verbalized = extract_verbalized
        self.extract_self_consistency = extract_self_consistency
        self.extract_embedding_consistency = extract_embedding_consistency
        self.sc_k = self_consistency_k
        self.sc_temperatures = self_consistency_temperatures or [0.5, 1.0]
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.logprob_top_k = logprob_top_k
        self.extract_logprob_freeform = extract_logprob_freeform
        self.logprob_normalization = logprob_normalization

    async def extract(
        self,
        sample: Sample,
        prompt: str,
        max_tokens: int = 256,
        max_retries: int = 5,
    ) -> ConfidenceResult:
        """
        Extract all confidence signals for a sample.

        Two-step pipeline:
        1. Constrained A/B classification with logprobs (predict_classification)
        2. Multi-turn confidence elicitation (predict_confidence_elicitation)
        3. (Optional) Freeform logprob baseline for methodological comparison
        4. Self-consistency / embedding consistency (unchanged)

        Args:
            sample: Dataset sample (DDISample, CheXpertSample, etc.)
            prompt: Diagnosis prompt (used for freeform baseline and self-consistency)
            max_tokens: Maximum response tokens
            max_retries: Max retries when prediction is unknown or confidence is null

        Returns:
            ConfidenceResult with all extracted signals
        """
        image_base64 = sample.to_base64()
        classification_prompt = self.model.get_classification_prompt()

        # === Step 1: Constrained A/B classification + logprob confidence ===
        logprob_conf = None
        min_logprob_conf = None
        logprob_compliant = None
        prediction = "unknown"
        raw_cls_response = ""

        if self.extract_logprob:
            try:
                cls_result = await self.model.predict_classification(
                    image_base64=image_base64,
                    logprobs_top_k=self.logprob_top_k,
                )
                logprob_conf = extract_constrained_logprob_confidence(cls_result)
                min_logprob_conf = logprob_conf  # identical for single token
                logprob_compliant = cls_result.compliant
                prediction = cls_result.predicted_label
                raw_cls_response = cls_result.raw_response

                if not cls_result.compliant:
                    logger.warning(
                        "Non-compliant A/B token for %s: got %r",
                        sample.image_id,
                        cls_result.raw_first_token,
                    )
            except Exception as e:
                logger.warning(
                    "Classification logprob failed for %s: %s",
                    sample.image_id,
                    e,
                )

        # If classification didn't produce a valid prediction, fall back to
        # freeform prediction
        if prediction == "unknown":
            response = await self.model.predict(
                image_base64=image_base64,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                return_logprobs=False,
            )
            prediction = response.prediction

        # === Step 2: Multi-turn confidence elicitation ===
        verbalized_conf = None
        raw_response = ""

        if self.extract_verbalized and prediction != "unknown":
            try:
                label_letter = "A" if prediction == self.model._task_config.a_label else "B"
                ce_text = await self.model.predict_confidence_elicitation(
                    image_base64=image_base64,
                    classification_prompt=classification_prompt,
                    predicted_label=label_letter,
                )
                verbalized_conf = extract_verbalized_confidence(ce_text)
                raw_response = ce_text
            except Exception as e:
                logger.warning(
                    "Confidence elicitation failed for %s: %s",
                    sample.image_id,
                    e,
                )

        # === Step 3: (Optional) Freeform logprob baseline ===
        logprob_conf_freeform = None
        min_logprob_conf_freeform = None

        if self.extract_logprob_freeform:
            try:
                freeform_response = await self.model.predict(
                    image_base64=image_base64,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    return_logprobs=True,
                    logprobs_top_k=self.logprob_top_k,
                )
                tc = self.model._task_config
                logprob_conf_freeform = extract_logprob_confidence(
                    freeform_response, normalization=self.logprob_normalization,
                    a_label=tc.a_label, b_label=tc.b_label,
                )
                min_logprob_conf_freeform = extract_min_logprob_confidence(
                    freeform_response, normalization=self.logprob_normalization,
                    a_label=tc.a_label, b_label=tc.b_label,
                )
                if not raw_response:
                    raw_response = freeform_response.text
            except Exception as e:
                logger.warning(
                    "Freeform logprob failed for %s: %s",
                    sample.image_id,
                    e,
                )

        # === Step 4: Self-consistency ===
        sc_results: dict[float, tuple[str, float]] = {}
        if self.extract_self_consistency:
            for temp in self.sc_temperatures:
                try:
                    sc_pred, sc_conf, _ = await compute_self_consistency_confidence(
                        model=self.model,
                        sample=sample,
                        k=self.sc_k,
                        temperature=temp,
                    )
                    sc_results[temp] = (sc_pred, sc_conf)
                except RuntimeError:
                    pass

        # === Step 5: Embedding consistency (reasoning models only) ===
        # EC embeds full-text responses and measures semantic agreement.
        # For non-reasoning models, responses are short structured text
        # (e.g., "Answer: Benign\nConfidence: 75%") where embeddings
        # collapse to near-identical vectors regardless of content,
        # making EC redundant with SC.
        emb_results: dict[float, float] = {}
        if self.extract_embedding_consistency and self.model.is_reasoning_model:
            for temp in self.sc_temperatures:
                try:
                    emb_sim, _ = await compute_embedding_consistency(
                        model=self.model,
                        sample=sample,
                        prompt=prompt,
                        k=self.sc_k,
                        temperature=temp,
                        max_tokens=max_tokens,
                        embedding_model=self.embedding_model,
                        embedding_api_key=self.embedding_api_key,
                    )
                    emb_results[temp] = emb_sim
                except RuntimeError:
                    pass

        return ConfidenceResult(
            sample_id=sample.image_id,
            prediction=prediction,
            ground_truth=sample.label_name,
            demographic_group=sample.demographic_group,
            logprob_confidence=logprob_conf,
            min_logprob_confidence=min_logprob_conf,
            verbalized_confidence=verbalized_conf,
            self_consistency_results=sc_results or None,
            embedding_consistency_results=emb_results or None,
            logprob_confidence_freeform=logprob_conf_freeform,
            min_logprob_confidence_freeform=min_logprob_conf_freeform,
            logprob_compliant=logprob_compliant,
            raw_classification_response=raw_cls_response,
            raw_response=raw_response,
        )

    async def extract_batch(
        self,
        samples: list[Sample],
        prompt: str,
        max_tokens: int = 256,
        progress_callback=None,
        checkpoint_callback=None,
    ) -> list[ConfidenceResult]:
        """
        Extract confidence for multiple samples.

        Args:
            samples: List of dataset samples
            prompt: Diagnosis prompt
            max_tokens: Maximum response tokens
            progress_callback: Optional callback(current, total) for progress updates
            checkpoint_callback: Optional callback(result) to persist each result
                incrementally (e.g. append to JSONL checkpoint file)

        Returns:
            List of ConfidenceResult objects
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                result = await self.extract(sample, prompt, max_tokens)
                results.append(result)
            except Exception as e:
                # Create a failed result
                result = ConfidenceResult(
                    sample_id=sample.image_id,
                    prediction="unknown",
                    ground_truth=sample.label_name,
                    demographic_group=sample.demographic_group,
                    logprob_confidence=None,
                    min_logprob_confidence=None,
                    verbalized_confidence=None,
                    self_consistency_results=None,
                    embedding_consistency_results=None,
                    raw_response=f"Error: {e}",
                )
                results.append(result)
                logger.error("Failed to extract confidence for %s: %s", sample.image_id, e)

            if checkpoint_callback:
                checkpoint_callback(result)

            if progress_callback:
                progress_callback(i + 1, len(samples))

        return results
