"""Demographic-conditional calibration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration.bootstrap import BootstrapResult, bootstrap_all_metrics
from ..calibration.metrics import CalibrationMetrics, compute_all_metrics
from ..calibration.temperature_scaling import DemographicTemperatureScaler
from ..confidence.aggregator import ConfidenceResult


class DemographicAnalyzer:
    """Analyze calibration across demographic groups."""

    def __init__(self, results: list[ConfidenceResult] | list[dict]):
        """
        Initialize analyzer.

        Args:
            results: List of ConfidenceResult objects or dictionaries
        """
        if results and isinstance(results[0], dict):
            self.results = [ConfidenceResult.from_dict(r) for r in results]
        else:
            self.results = results

        self.df = pd.DataFrame([r.to_dict() for r in self.results])

    @staticmethod
    def _get_correctness(
        df: pd.DataFrame,
        confidence_column: str,
    ) -> pd.Series:
        """
        Get the correctness labels appropriate for a given confidence column.

        For self-consistency columns (sc_confidence_t*), correctness is based
        on the SC majority-vote prediction, not the primary (temp=0) prediction.
        For all other confidence types, uses the primary prediction's correctness.

        Args:
            df: DataFrame with result columns
            confidence_column: The confidence column being evaluated

        Returns:
            Boolean Series indicating correctness
        """
        if confidence_column.startswith("sc_confidence_t"):
            temp_suffix = confidence_column.removeprefix("sc_confidence_")
            pred_col = f"sc_prediction_{temp_suffix}"
            if pred_col in df.columns:
                return df[pred_col] == df["ground_truth"]
        return df["is_correct"]

    def compute_metrics_by_group(
        self,
        confidence_column: str,
        n_bins: int = 10,
    ) -> dict[str, CalibrationMetrics]:
        """
        Compute calibration metrics for each demographic group.

        Args:
            confidence_column: Column name for confidence scores
            n_bins: Number of bins for ECE

        Returns:
            Dict mapping group name to CalibrationMetrics
        """
        metrics_by_group = {}

        for group in self.df["demographic_group"].unique():
            if group == "unknown":
                continue

            group_df = self.df[self.df["demographic_group"] == group]

            # Filter out None values
            valid_mask = group_df[confidence_column].notna()
            confidences = group_df.loc[valid_mask, confidence_column].values.astype(float)
            correct_series = self._get_correctness(group_df, confidence_column)
            correctness = correct_series.loc[valid_mask].values.astype(int)

            if len(confidences) > 0:
                labels = group_df.loc[valid_mask, "ground_truth"].values
                metrics = compute_all_metrics(confidences, correctness, n_bins, labels=labels)
                metrics_by_group[group] = metrics

        return metrics_by_group

    def compute_bootstrap_by_group(
        self,
        confidence_column: str,
        n_bins: int = 10,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        random_seed: int | None = None,
    ) -> dict[str, dict[str, BootstrapResult]]:
        """
        Compute bootstrap confidence intervals for each group.

        Args:
            confidence_column: Column name for confidence scores
            n_bins: Number of bins for ECE
            n_bootstrap: Bootstrap iterations
            ci_level: CI level
            random_seed: Random seed

        Returns:
            Nested dict of {group: {metric: BootstrapResult}}
        """
        results_by_group = {}

        for group in self.df["demographic_group"].unique():
            if group == "unknown":
                continue

            group_df = self.df[self.df["demographic_group"] == group]

            valid_mask = group_df[confidence_column].notna()
            confidences = group_df.loc[valid_mask, confidence_column].values.astype(float)
            correct_series = self._get_correctness(group_df, confidence_column)
            correctness = correct_series.loc[valid_mask].values.astype(int)

            if len(confidences) >= 10:  # Need enough samples for bootstrap
                labels = group_df.loc[valid_mask, "ground_truth"].values
                results_by_group[group] = bootstrap_all_metrics(
                    confidences, correctness, n_bins, n_bootstrap, ci_level, random_seed,
                    labels=labels,
                )

        return results_by_group

    def fit_demographic_temperature_scaling(
        self,
        confidence_column: str,
        val_indices: np.ndarray | list[int] | None = None,
    ) -> DemographicTemperatureScaler:
        """
        Fit demographic-conditional temperature scaling on validation set.

        Args:
            confidence_column: Column name for confidence scores
            val_indices: Indices of validation samples (uses all if None)

        Returns:
            Fitted DemographicTemperatureScaler
        """
        if val_indices is not None:
            val_df = self.df.iloc[val_indices]
        else:
            val_df = self.df

        valid_mask = val_df[confidence_column].notna()
        confidences = val_df.loc[valid_mask, confidence_column].values.astype(float)
        correct_series = self._get_correctness(val_df, confidence_column)
        correctness = correct_series.loc[valid_mask].values.astype(int)
        groups = val_df.loc[valid_mask, "demographic_group"].values

        scaler = DemographicTemperatureScaler()
        scaler.fit(confidences, correctness, groups)

        return scaler

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary statistics table.

        Returns:
            DataFrame with metrics by confidence type and group
        """
        summary_rows = []

        sc_cols = [c for c in self.df.columns if c.startswith("sc_confidence_t")]
        # Also detect the single self_consistency_confidence column
        if "self_consistency_confidence" in self.df.columns and not sc_cols:
            sc_cols = ["self_consistency_confidence"]
        # Detect embedding consistency columns
        emb_cols = [c for c in self.df.columns if c.startswith("emb_consistency_t")]
        conf_types = ["logprob_confidence", "verbalized_confidence"] + sc_cols + emb_cols
        # Include freeform logprob baseline if present
        if "logprob_confidence_freeform" in self.df.columns:
            conf_types.append("logprob_confidence_freeform")
        for conf_type in conf_types:
            for group in sorted(self.df["demographic_group"].unique()):
                if group == "unknown":
                    continue

                group_df = self.df[self.df["demographic_group"] == group]
                valid_mask = group_df[conf_type].notna()

                if valid_mask.sum() > 0:
                    confidences = group_df.loc[valid_mask, conf_type].values.astype(float)
                    correct_series = self._get_correctness(group_df, conf_type)
                    correctness = correct_series.loc[valid_mask].values.astype(int)
                    labels = group_df.loc[valid_mask, "ground_truth"].values

                    metrics = compute_all_metrics(
                        confidences, correctness, labels=labels
                    )

                    row = {
                        "confidence_type": conf_type.replace("_confidence", ""),
                        "demographic_group": group,
                        "n_samples": metrics.n_samples,
                        "accuracy": metrics.accuracy,
                        "balanced_accuracy": metrics.balanced_accuracy,
                        "ece": metrics.ece,
                        "nll": metrics.nll,
                        "brier": metrics.brier,
                    }
                    summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def get_confidence_arrays(
        self,
        confidence_column: str,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Get confidence and correctness arrays by group.

        Args:
            confidence_column: Column name for confidence scores

        Returns:
            Dict mapping group to (confidences, correctness) tuples
        """
        arrays = {}

        for group in self.df["demographic_group"].unique():
            if group == "unknown":
                continue

            group_df = self.df[self.df["demographic_group"] == group]
            valid_mask = group_df[confidence_column].notna()

            confidences = group_df.loc[valid_mask, confidence_column].values.astype(float)
            correct_series = self._get_correctness(group_df, confidence_column)
            correctness = correct_series.loc[valid_mask].values.astype(int)

            if len(confidences) > 0:
                arrays[group] = (confidences, correctness)

        return arrays

    def compute_calibration_gap(
        self,
        confidence_column: str,
    ) -> dict[str, float]:
        """
        Compute calibration gap between demographic groups.

        Returns the difference in ECE between groups.

        Args:
            confidence_column: Column name for confidence scores

        Returns:
            Dict with gap statistics
        """
        metrics = self.compute_metrics_by_group(confidence_column)

        if len(metrics) < 2:
            return {"gap": 0.0}

        eces = [m.ece for m in metrics.values()]
        groups = list(metrics.keys())

        return {
            "max_gap": max(eces) - min(eces),
            "groups": {g: m.ece for g, m in metrics.items()},
            "worst_calibrated": groups[np.argmax(eces)],
            "best_calibrated": groups[np.argmin(eces)],
        }

    def compute_equity_summary(
        self,
        confidence_columns: list[str],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compare calibration equity across confidence signals.

        For each confidence signal, computes per-group ECE and summary
        disparity metrics. Sorted by max_ece_gap ascending (most equitable first).

        Args:
            confidence_columns: Confidence column names to compare.
            n_bins: Number of bins for ECE.

        Returns:
            DataFrame with columns: confidence_type, ece_<group>..., mean_ece,
            max_ece_gap, ece_range_ratio, worst_group, best_group.
        """
        rows = []

        for col in confidence_columns:
            if col not in self.df.columns:
                continue

            metrics_by_group = self.compute_metrics_by_group(col, n_bins)
            if len(metrics_by_group) < 2:
                continue

            group_eces = {g: m.ece for g, m in metrics_by_group.items()}
            ece_values = list(group_eces.values())
            min_ece = min(ece_values)
            max_ece = max(ece_values)

            row = {"confidence_type": col}
            for g, e in sorted(group_eces.items()):
                row[f"ece_{g}"] = e

            row["mean_ece"] = float(np.mean(ece_values))
            row["max_ece_gap"] = max_ece - min_ece
            row["ece_range_ratio"] = max_ece / min_ece if min_ece > 1e-10 else float("inf")

            groups_list = list(group_eces.keys())
            ece_arr = np.array(ece_values)
            row["worst_group"] = groups_list[int(np.argmax(ece_arr))]
            row["best_group"] = groups_list[int(np.argmin(ece_arr))]

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("max_ece_gap").reset_index(drop=True)
        return df

    def compute_compliance_rate(self) -> dict[str, float]:
        """
        Compute A/B compliance rate per demographic group.

        Returns:
            Dict mapping group name to compliance rate (0-1).
            Includes an "overall" key for the full dataset.
        """
        if "logprob_compliant" not in self.df.columns:
            return {}

        valid = self.df[self.df["logprob_compliant"].notna()]
        if valid.empty:
            return {}

        rates: dict[str, float] = {}
        rates["overall"] = float(valid["logprob_compliant"].mean())

        for group in sorted(valid["demographic_group"].unique()):
            if group == "unknown":
                continue
            group_df = valid[valid["demographic_group"] == group]
            if not group_df.empty:
                rates[group] = float(group_df["logprob_compliant"].mean())

        return rates
