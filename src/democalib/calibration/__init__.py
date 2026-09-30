"""Calibration metrics and methods."""

from .metrics import compute_ece, compute_nll, compute_brier_score, compute_balanced_accuracy, compute_all_metrics, CalibrationMetrics
from .temperature_scaling import fit_temperature, DemographicTemperatureScaler, TemperatureScalingResult
from .bootstrap import bootstrap_metric, bootstrap_all_metrics, BootstrapResult

__all__ = [
    "compute_ece",
    "compute_nll",
    "compute_brier_score",
    "compute_balanced_accuracy",
    "compute_all_metrics",
    "CalibrationMetrics",
    "fit_temperature",
    "DemographicTemperatureScaler",
    "TemperatureScalingResult",
    "bootstrap_metric",
    "bootstrap_all_metrics",
    "BootstrapResult",
]
