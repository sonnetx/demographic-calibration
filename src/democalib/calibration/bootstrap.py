"""Bootstrap confidence intervals for calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import (
    compute_auroc,
    compute_balanced_accuracy,
    compute_brier_score,
    compute_ece,
    compute_nll,
)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    std_error: float
    bootstrap_samples: np.ndarray


def bootstrap_metric(
    confidences: np.ndarray,
    correctness: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int | None = None,
    extra_arrays: list[np.ndarray] | None = None,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a metric.

    Uses the percentile method for CI construction.

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators
        metric_fn: Function that computes metric(confidences, correctness, *extras)
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        extra_arrays: Additional arrays to resample in parallel (e.g., labels)

    Returns:
        BootstrapResult with point estimate and CI
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)
    extras = [np.asarray(a) for a in extra_arrays] if extra_arrays else []

    rng = np.random.default_rng(random_seed)
    n_samples = len(confidences)

    # Point estimate
    point_estimate = metric_fn(confidences, correctness, *extras)

    # Bootstrap samples
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_conf = confidences[indices]
        boot_correct = correctness[indices]
        boot_extras = [a[indices] for a in extras]
        bootstrap_estimates[i] = metric_fn(boot_conf, boot_correct, *boot_extras)

    # Compute confidence interval (percentile method)
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_width=float(ci_upper - ci_lower),
        std_error=float(np.std(bootstrap_estimates)),
        bootstrap_samples=bootstrap_estimates,
    )


def bootstrap_all_metrics(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, BootstrapResult]:
    """
    Compute bootstrap CIs for all calibration metrics.

    Args:
        confidences: Predicted confidence scores
        correctness: Binary correctness indicators
        n_bins: Number of bins for ECE
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence interval level
        random_seed: Random seed
        labels: Ground-truth class labels (needed for balanced accuracy)

    Returns:
        Dictionary mapping metric names to BootstrapResult
    """

    def ece_fn(conf, corr):
        return compute_ece(conf, corr, n_bins)[0]

    def accuracy_fn(conf, corr):
        return float(corr.mean())

    def auroc_fn(conf, corr):
        result = compute_auroc(conf, corr)
        return result if result is not None else float("nan")

    results = {
        "ece": bootstrap_metric(
            confidences, correctness, ece_fn, n_bootstrap, ci_level, random_seed
        ),
        "nll": bootstrap_metric(
            confidences, correctness, compute_nll, n_bootstrap, ci_level, random_seed
        ),
        "brier": bootstrap_metric(
            confidences, correctness, compute_brier_score, n_bootstrap, ci_level, random_seed
        ),
        "auroc": bootstrap_metric(
            confidences, correctness, auroc_fn, n_bootstrap, ci_level, random_seed
        ),
        "accuracy": bootstrap_metric(
            confidences, correctness, accuracy_fn, n_bootstrap, ci_level, random_seed
        ),
    }

    if labels is not None:
        labels = np.asarray(labels)

        def balanced_acc_fn(conf, corr, lbls):
            result = compute_balanced_accuracy(corr, lbls)
            return result if result is not None else float("nan")

        results["balanced_accuracy"] = bootstrap_metric(
            confidences, correctness, balanced_acc_fn,
            n_bootstrap, ci_level, random_seed,
            extra_arrays=[labels],
        )

    return results


def compare_demographic_intervals(
    results_by_group: dict[str, dict[str, BootstrapResult]],
) -> dict[str, dict]:
    """
    Compare bootstrap intervals across demographic groups.

    Analyzes coverage and width differences.

    Args:
        results_by_group: Nested dict of {group: {metric: BootstrapResult}}

    Returns:
        Comparison statistics
    """
    comparisons = {}

    groups = list(results_by_group.keys())
    if not groups:
        return comparisons

    metrics = list(results_by_group[groups[0]].keys())

    for metric in metrics:
        metric_comparison = {
            "widths": {},
            "point_estimates": {},
            "ci_lowers": {},
            "ci_uppers": {},
            "max_width_diff": 0.0,
            "max_estimate_diff": 0.0,
        }

        for group in groups:
            result = results_by_group[group][metric]
            metric_comparison["widths"][group] = result.ci_width
            metric_comparison["point_estimates"][group] = result.point_estimate
            metric_comparison["ci_lowers"][group] = result.ci_lower
            metric_comparison["ci_uppers"][group] = result.ci_upper

        # Compute differences
        widths = list(metric_comparison["widths"].values())
        estimates = list(metric_comparison["point_estimates"].values())
        metric_comparison["max_width_diff"] = max(widths) - min(widths)
        metric_comparison["max_estimate_diff"] = max(estimates) - min(estimates)

        comparisons[metric] = metric_comparison

    return comparisons


def test_calibration_difference(
    results_group1: dict[str, BootstrapResult],
    results_group2: dict[str, BootstrapResult],
    metric: str = "ece",
) -> dict:
    """
    Test if calibration differs significantly between two groups.

    Uses bootstrap distributions to compute p-value for the null hypothesis
    that the two groups have equal calibration.

    Args:
        results_group1: Bootstrap results for group 1
        results_group2: Bootstrap results for group 2
        metric: Metric to compare

    Returns:
        Dictionary with test statistics
    """
    samples1 = results_group1[metric].bootstrap_samples
    samples2 = results_group2[metric].bootstrap_samples

    # Observed difference
    observed_diff = (
        results_group1[metric].point_estimate - results_group2[metric].point_estimate
    )

    # Bootstrap differences
    diff_samples = samples1 - samples2

    # Two-tailed p-value
    p_value = np.mean(np.abs(diff_samples) >= np.abs(observed_diff))

    return {
        "observed_difference": observed_diff,
        "mean_difference": float(np.mean(diff_samples)),
        "std_difference": float(np.std(diff_samples)),
        "p_value": float(p_value),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }
