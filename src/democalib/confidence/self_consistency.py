"""Self-consistency confidence extraction.

Uses the same constrained A/B classification prompt as TLP, but with
temperature > 0 to generate K stochastic single-token predictions.
Confidence = fraction of samples agreeing with the majority vote.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

from ..data.dataset import Sample
from ..models.base import BaseVLM, LogprobClassificationResult

logger = logging.getLogger(__name__)


async def compute_self_consistency_confidence(
    model: BaseVLM,
    sample: Sample,
    k: int = 10,
    temperature: float = 0.7,
) -> tuple[str, float, list[LogprobClassificationResult]]:
    """
    Compute self-consistency confidence via K stochastic A/B classifications.

    Uses the same constrained single-token A/B prompt as TLP
    (predict_classification), ensuring SC measures the stability of the
    same decision process that produces the primary prediction.

    Args:
        model: VLM model instance
        sample: Data sample
        k: Number of generations
        temperature: Sampling temperature (higher = more diversity)

    Returns:
        Tuple of (majority_prediction, consistency_score, all_results)

    Raises:
        RuntimeError: If all samples fail
    """
    image_base64 = sample.to_base64()

    # Generate K single-token A/B responses at temperature > 0
    tasks = [
        model.predict_classification(
            image_base64=image_base64,
            logprobs_top_k=5,  # Don't need full top-k for SC
            temperature=temperature,
        )
        for _ in range(k)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors
    valid_results = [r for r in responses if isinstance(r, LogprobClassificationResult)]
    n_errors = len(responses) - len(valid_results)

    if n_errors > 0:
        logger.warning(
            "Self-consistency: %d/%d samples failed for %s",
            n_errors, k, sample.image_id,
        )

    if not valid_results:
        raise RuntimeError("All self-consistency samples failed")

    # Count predictions
    predictions = [r.predicted_label for r in valid_results]
    prediction_counts = Counter(predictions)

    # Majority vote
    majority_prediction = prediction_counts.most_common(1)[0][0]

    # Consistency score = fraction matching majority
    consistency_score = prediction_counts[majority_prediction] / len(valid_results)

    logger.info(
        "Self-consistency [%s]: %d/%d valid, distribution=%s, score=%.2f",
        sample.image_id,
        len(valid_results),
        k,
        dict(prediction_counts),
        consistency_score,
    )

    return majority_prediction, consistency_score, valid_results


def aggregate_predictions_weighted(
    predictions: list[str],
    weights: list[float] | None = None,
) -> tuple[str, float]:
    """
    Aggregate multiple predictions with optional weights.

    Args:
        predictions: List of predicted labels
        weights: Optional confidence weights for each prediction

    Returns:
        Tuple of (final_prediction, confidence_score)
    """
    if weights is None:
        weights = [1.0] * len(predictions)

    if len(predictions) != len(weights):
        raise ValueError("Predictions and weights must have same length")

    # Weighted vote
    weighted_counts: dict[str, float] = {}
    for pred, weight in zip(predictions, weights):
        weighted_counts[pred] = weighted_counts.get(pred, 0) + weight

    total_weight = sum(weights)

    if total_weight == 0:
        return "unknown", 0.0

    # Find winner
    winner = max(weighted_counts.items(), key=lambda x: x[1])
    final_prediction = winner[0]
    confidence = winner[1] / total_weight

    return final_prediction, confidence
