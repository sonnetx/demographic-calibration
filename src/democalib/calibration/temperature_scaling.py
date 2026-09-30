"""Temperature scaling for post-hoc calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class TemperatureScalingResult:
    """Result of temperature scaling optimization."""

    temperature: float
    nll_before: float
    nll_after: float
    improvement: float


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    scaled_logits = logits / temperature
    confidence = sigmoid(scaled_logits) for binary case

    Args:
        logits: Raw logits (log-odds)
        temperature: Temperature parameter (T > 1 increases entropy)

    Returns:
        Temperature-scaled probabilities
    """
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def logits_from_confidence(
    confidences: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Convert confidence scores to logits.

    logit = log(p / (1-p))

    Args:
        confidences: Confidence scores in (0, 1)
        eps: Small value for numerical stability

    Returns:
        Logits
    """
    confidences = np.clip(confidences, eps, 1 - eps)
    return np.log(confidences / (1 - confidences))


def compute_nll_from_logits(
    logits: np.ndarray,
    correctness: np.ndarray,
    temperature: float,
    eps: float = 1e-15,
) -> float:
    """
    Compute NLL given logits and temperature.

    Args:
        logits: Raw logits
        correctness: Binary correctness indicators
        temperature: Temperature parameter
        eps: Small value for numerical stability

    Returns:
        NLL value
    """
    scaled_probs = apply_temperature_scaling(logits, temperature)
    scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
    nll = -np.mean(
        correctness * np.log(scaled_probs) + (1 - correctness) * np.log(1 - scaled_probs)
    )
    return float(nll)


def fit_temperature(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bounds: tuple[float, float] = (0.1, 10.0),
    max_iter: int = 100,
) -> TemperatureScalingResult:
    """
    Fit optimal temperature by minimizing NLL on validation data.

    Following Guo et al. 2017, we find T* = argmin_T NLL(scaled_conf, correctness)

    Args:
        confidences: Predicted confidence scores on validation set
        correctness: Binary correctness indicators on validation set
        bounds: Bounds for temperature search
        max_iter: Maximum iterations

    Returns:
        TemperatureScalingResult
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    logits = logits_from_confidence(confidences)

    # NLL before scaling (temperature = 1)
    nll_before = compute_nll_from_logits(logits, correctness, temperature=1.0)

    def objective(temperature):
        return compute_nll_from_logits(logits, correctness, temperature)

    # Optimize temperature using bounded scalar optimization
    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"maxiter": max_iter},
    )

    optimal_temp = result.x
    nll_after = result.fun

    return TemperatureScalingResult(
        temperature=optimal_temp,
        nll_before=nll_before,
        nll_after=nll_after,
        improvement=nll_before - nll_after,
    )


def apply_fitted_temperature(
    confidences: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply a fitted temperature to confidence scores.

    Args:
        confidences: Original confidence scores
        temperature: Fitted temperature parameter

    Returns:
        Temperature-scaled confidence scores
    """
    logits = logits_from_confidence(confidences)
    return apply_temperature_scaling(logits, temperature)


class DemographicTemperatureScaler:
    """
    Demographic-conditional temperature scaling.

    Fits separate temperature parameters for each demographic group.
    """

    def __init__(self):
        """Initialize scaler."""
        self.temperatures: dict[str, float] = {}
        self.results: dict[str, TemperatureScalingResult] = {}

    def fit(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        groups: np.ndarray,
        **kwargs,
    ) -> dict[str, TemperatureScalingResult]:
        """
        Fit group-specific temperatures.

        Args:
            confidences: All confidence scores
            correctness: All correctness indicators
            groups: Group labels for each sample
            **kwargs: Passed to fit_temperature

        Returns:
            Dictionary mapping group names to results
        """
        confidences = np.asarray(confidences)
        correctness = np.asarray(correctness)
        groups = np.asarray(groups)

        unique_groups = np.unique(groups)

        for group in unique_groups:
            mask = groups == group
            group_conf = confidences[mask]
            group_correct = correctness[mask]

            if len(group_conf) < 5:
                # Not enough samples for reliable fitting
                self.temperatures[str(group)] = 1.0
                self.results[str(group)] = TemperatureScalingResult(
                    temperature=1.0,
                    nll_before=0.0,
                    nll_after=0.0,
                    improvement=0.0,
                )
                continue

            result = fit_temperature(group_conf, group_correct, **kwargs)
            self.temperatures[str(group)] = result.temperature
            self.results[str(group)] = result

        return self.results

    def transform(
        self,
        confidences: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Apply group-specific temperature scaling.

        Args:
            confidences: Confidence scores to scale
            groups: Group labels for each sample

        Returns:
            Temperature-scaled confidence scores
        """
        confidences = np.asarray(confidences)
        groups = np.asarray(groups)

        logits = logits_from_confidence(confidences)
        scaled = np.zeros_like(confidences)

        for group, temp in self.temperatures.items():
            mask = groups == group
            if mask.any():
                scaled[mask] = apply_temperature_scaling(logits[mask], temp)

        # Handle unknown groups (use temperature = 1)
        unknown_mask = ~np.isin(groups, list(self.temperatures.keys()))
        if unknown_mask.any():
            scaled[unknown_mask] = apply_temperature_scaling(logits[unknown_mask], 1.0)

        return scaled

    def fit_transform(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        groups: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            confidences: Confidence scores
            correctness: Correctness indicators
            groups: Group labels
            **kwargs: Passed to fit_temperature

        Returns:
            Temperature-scaled confidence scores
        """
        self.fit(confidences, correctness, groups, **kwargs)
        return self.transform(confidences, groups)

    def fit_transform_cv(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        groups: np.ndarray,
        n_folds: int = 5,
        random_seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        K-fold cross-validated temperature scaling.

        Fits temperature on K-1 folds and applies to the held-out fold,
        so every sample gets a calibrated confidence without data leakage.
        Also stores the average per-group temperature in self.temperatures.

        Args:
            confidences: All confidence scores
            correctness: All correctness indicators
            groups: Group labels for each sample
            n_folds: Number of CV folds
            random_seed: Random seed for fold assignment
            **kwargs: Passed to fit_temperature

        Returns:
            Temperature-scaled confidence scores for all samples
        """
        from sklearn.model_selection import StratifiedKFold

        confidences = np.asarray(confidences, dtype=float)
        correctness = np.asarray(correctness)
        groups = np.asarray(groups)

        scaled = np.zeros_like(confidences)
        # Track per-group temperatures across folds for reporting
        group_temps: dict[str, list[float]] = {}

        # Stratify by group + correctness to keep folds balanced
        stratify_keys = np.array([f"{g}_{c}" for g, c in zip(groups, correctness)])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

        for train_idx, test_idx in skf.split(confidences, stratify_keys):
            fold_scaler = DemographicTemperatureScaler()
            fold_scaler.fit(
                confidences[train_idx],
                correctness[train_idx],
                groups[train_idx],
                **kwargs,
            )
            scaled[test_idx] = fold_scaler.transform(confidences[test_idx], groups[test_idx])

            for g, t in fold_scaler.temperatures.items():
                group_temps.setdefault(g, []).append(t)

        # Store average temperatures for reporting
        for g, temps in group_temps.items():
            avg_t = float(np.mean(temps))
            self.temperatures[g] = avg_t
            nll_before = compute_nll_from_logits(
                logits_from_confidence(confidences[groups == g]),
                correctness[groups == g],
                temperature=1.0,
            )
            nll_after = compute_nll_from_logits(
                logits_from_confidence(confidences[groups == g]),
                correctness[groups == g],
                temperature=avg_t,
            )
            self.results[g] = TemperatureScalingResult(
                temperature=avg_t,
                nll_before=nll_before,
                nll_after=nll_after,
                improvement=nll_before - nll_after,
            )

        return scaled
