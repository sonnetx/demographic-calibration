"""Calibration and discrimination metrics: ECE, NLL, Brier score, ROC AUC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class CalibrationMetrics:
    """Container for calibration and discrimination metrics."""

    ece: float
    nll: float
    brier: float
    accuracy: float
    auroc: float | None
    n_samples: int

    # For reliability diagram
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray

    balanced_accuracy: float | None = None


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error.

    ECE = sum_{m=1}^{M} (|B_m|/n) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: Predicted confidence scores in [0, 1]
        correctness: Binary correctness indicators (1 = correct, 0 = incorrect)
        n_bins: Number of bins

    Returns:
        Tuple of (ece, bin_confidences, bin_accuracies, bin_counts)
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_confidences[b] = confidences[mask].mean()
            bin_accuracies[b] = correctness[mask].mean()
            bin_counts[b] = mask.sum()

    # ECE: weighted average of |accuracy - confidence|
    weights = bin_counts / len(confidences)
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))

    return float(ece), bin_confidences, bin_accuracies, bin_counts


def compute_nll(
    confidences: np.ndarray,
    correctness: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Compute Negative Log-Likelihood (cross-entropy loss).

    For calibration, confidence represents P(correct), so:
    NLL = -mean(correct * log(conf) + (1-correct) * log(1-conf))

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators
        eps: Small value for numerical stability

    Returns:
        NLL value
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    confidences = np.clip(confidences, eps, 1 - eps)
    nll = -np.mean(
        correctness * np.log(confidences) + (1 - correctness) * np.log(1 - confidences)
    )
    return float(nll)


def compute_brier_score(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> float:
    """
    Compute Brier Score.

    Brier = mean((confidence - correctness)^2)

    Lower is better. Range: [0, 1]

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators

    Returns:
        Brier score
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    return float(np.mean((confidences - correctness) ** 2))


def compute_auroc(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> float | None:
    """
    Compute ROC AUC for discrimination.

    Measures how well the confidence proxy separates correct from incorrect
    predictions. Higher is better. Range: [0, 1], 0.5 = random.

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators

    Returns:
        AUROC value, or None if undefined (e.g., all same class)
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    # AUROC is undefined if only one class present
    if len(np.unique(correctness)) < 2:
        return None

    return float(roc_auc_score(correctness, confidences))


def compute_balanced_accuracy(
    correctness: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """
    Compute balanced accuracy (average per-class recall).

    balanced_accuracy = mean over classes c of: mean(correctness[labels == c])

    Args:
        correctness: Binary correctness indicators (1 = correct, 0 = incorrect)
        labels: Ground-truth class labels for each sample

    Returns:
        Balanced accuracy, or None if any class has zero samples
    """
    correctness = np.asarray(correctness)
    labels = np.asarray(labels)

    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        return None

    per_class_recall = []
    for c in unique_classes:
        mask = labels == c
        if mask.sum() == 0:
            return None
        per_class_recall.append(float(correctness[mask].mean()))

    return float(np.mean(per_class_recall))


def compute_all_metrics(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    labels: np.ndarray | None = None,
) -> CalibrationMetrics:
    """
    Compute all calibration metrics.

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators
        n_bins: Number of bins for ECE
        labels: Ground-truth class labels (needed for balanced accuracy)

    Returns:
        CalibrationMetrics dataclass
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    ece, bin_conf, bin_acc, bin_counts = compute_ece(confidences, correctness, n_bins)
    nll = compute_nll(confidences, correctness)
    brier = compute_brier_score(confidences, correctness)
    accuracy = float(correctness.mean())
    auroc = compute_auroc(confidences, correctness)

    balanced_acc = None
    if labels is not None:
        balanced_acc = compute_balanced_accuracy(correctness, labels)

    return CalibrationMetrics(
        ece=ece,
        nll=nll,
        brier=brier,
        accuracy=accuracy,
        auroc=auroc,
        n_samples=len(confidences),
        bin_confidences=bin_conf,
        bin_accuracies=bin_acc,
        bin_counts=bin_counts,
        balanced_accuracy=balanced_acc,
    )


def compute_ece_quantile(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error using quantile (equal-count) bins.

    Uses quantile-based binning so each bin has approximately the same
    number of samples. This is the approach used in Rivera et al.

    Args:
        confidences: Predicted confidence scores in [0, 1]
        correctness: Binary correctness indicators
        n_bins: Number of quantile bins

    Returns:
        Tuple of (ece, bin_confidences, bin_accuracies, bin_counts)
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    # Compute quantile bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(confidences, quantiles)
    # Use digitize with quantile boundaries
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_confidences[b] = confidences[mask].mean()
            bin_accuracies[b] = correctness[mask].mean()
            bin_counts[b] = mask.sum()

    weights = bin_counts / len(confidences)
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))

    return float(ece), bin_confidences, bin_accuracies, bin_counts


def compute_mce(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error.

    MCE = max_{m} |acc(B_m) - conf(B_m)|

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators
        n_bins: Number of bins

    Returns:
        MCE value
    """
    _, bin_conf, bin_acc, bin_counts = compute_ece(confidences, correctness, n_bins)

    # Only consider non-empty bins
    non_empty = bin_counts > 0
    if not non_empty.any():
        return 0.0

    gaps = np.abs(bin_acc[non_empty] - bin_conf[non_empty])
    return float(np.max(gaps))
