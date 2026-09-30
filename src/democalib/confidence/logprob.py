"""Log-probability confidence extraction."""

from __future__ import annotations

import numpy as np

from ..models.base import LogprobClassificationResult, VLMResponse


def extract_logprob_confidence(
    response: VLMResponse,
    normalization: str = "byte",
    a_label: str = "benign",
    b_label: str = "malignant",
) -> float | None:
    """
    Extract normalized log-probability confidence.

    Args:
        response: VLM response with token log probabilities
        normalization: One of "byte", "token", "none".
        a_label: Label for choice A.
        b_label: Label for choice B.

    Returns:
        Confidence score in [0, 1], or None if logprobs unavailable
    """
    if not response.has_logprobs:
        return None

    return response.get_diagnosis_logprob(
        normalization=normalization, a_label=a_label, b_label=b_label,
    )


def extract_min_logprob_confidence(
    response: VLMResponse,
    normalization: str = "byte",
    a_label: str = "benign",
    b_label: str = "malignant",
) -> float | None:
    """
    Extract minimum token probability across diagnosis answer tokens.

    Args:
        response: VLM response with token log probabilities
        normalization: One of "byte", "token", "none".
        a_label: Label for choice A.
        b_label: Label for choice B.

    Returns:
        Min token probability in [0, 1], or None if logprobs unavailable
    """
    if not response.has_logprobs:
        return None

    return response.get_min_diagnosis_logprob(
        normalization=normalization, a_label=a_label, b_label=b_label,
    )


def extract_constrained_logprob_confidence(
    result: LogprobClassificationResult,
) -> float | None:
    """Extract confidence from constrained single-token A/B classification.

    Uses the softmax probability from predict_classification(), which forces
    a single-token A/B output and reads logprobs at position 0 — avoiding
    the near-deterministic logprobs from full-response extraction.

    Returns confidence even if the first token was non-compliant (not A/B),
    as long as both A and B logprobs were found in the top-k alternatives.

    Args:
        result: Result from predict_classification()

    Returns:
        Confidence score in [0, 1], or None if extraction failed
    """
    if result is None:
        return None
    if result.a_logprob is not None and result.b_logprob is not None:
        return result.confidence
    return None


def compute_sequence_probability(
    token_logprobs: list[float],
    normalize: str = "byte",
    token_texts: list[str] | None = None,
) -> float:
    """
    Compute sequence probability from token log probabilities.

    Args:
        token_logprobs: List of log probabilities
        normalize: Normalization mode — "byte", "token", or "none".
        token_texts: Token strings (required for "byte" normalization).

    Returns:
        Probability score
    """
    if not token_logprobs:
        return 0.0

    # Filter out invalid logprobs (and matching texts)
    if token_texts is not None:
        pairs = [
            (lp, t)
            for lp, t in zip(token_logprobs, token_texts)
            if lp > -100
        ]
        if not pairs:
            return 0.0
        valid_logprobs = [lp for lp, _ in pairs]
        valid_texts = [t for _, t in pairs]
    else:
        valid_logprobs = [lp for lp in token_logprobs if lp > -100]
        valid_texts = None
        if not valid_logprobs:
            return 0.0

    if normalize == "byte":
        if valid_texts is not None:
            byte_len = sum(len(t.encode("utf-8")) for t in valid_texts)
        else:
            byte_len = len(valid_logprobs)  # fallback to token count
        return float(np.exp(np.sum(valid_logprobs) / max(byte_len, 1)))
    elif normalize == "token":
        # Geometric mean = exp(mean(log_probs))
        return float(np.exp(np.mean(valid_logprobs)))
    else:  # "none"
        # Raw product = exp(sum(log_probs))
        return float(np.exp(np.sum(valid_logprobs)))


def compute_entropy_from_logprobs(token_logprobs: list[float]) -> float:
    """
    Compute entropy from token log probabilities.

    Higher entropy indicates more uncertainty.

    Args:
        token_logprobs: List of log probabilities

    Returns:
        Entropy value
    """
    if not token_logprobs:
        return 0.0

    # Filter out invalid logprobs
    valid_logprobs = [lp for lp in token_logprobs if lp > -100]

    if not valid_logprobs:
        return 0.0

    # Entropy = -sum(p * log(p)) = -sum(exp(logp) * logp)
    probs = np.exp(valid_logprobs)
    entropy = -np.sum(probs * valid_logprobs)

    return float(entropy)
