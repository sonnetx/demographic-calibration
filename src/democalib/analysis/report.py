"""Structured Markdown report generation for demographic calibration analysis."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, spearmanr

from ..calibration.bootstrap import bootstrap_all_metrics, test_calibration_difference
from ..calibration.metrics import compute_all_metrics, compute_auroc
from .demographic_analysis import DemographicAnalyzer


class ReportGenerator:
    """Generate a structured Markdown report from calibration analysis results."""

    def __init__(
        self,
        analyzer: DemographicAnalyzer,
        model_name: str = "Unknown Model",
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        random_seed: int = 42,
    ):
        self.analyzer = analyzer
        self.model_name = model_name
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_seed = random_seed

        self._df = analyzer.df
        self._conf_cols = self._detect_confidence_columns()
        self._degenerate = self._detect_degenerate_signals()
        self._groups = sorted(
            [g for g in self._df["demographic_group"].unique() if g != "unknown"]
        )

    def generate(self, output_path: Path) -> str:
        """Generate the full Markdown report and write to output_path."""
        sections = [
            f"# Demographic Calibration Report: {self.model_name}",
            f"**Generated:** {datetime.date.today().isoformat()} | "
            f"**Total Samples:** {len(self._df)}",
            "",
            "---",
            "",
            self._section_executive_summary(),
            self._section_confidence_signal_comparison(),
            self._section_overconfidence_analysis(),
            self._section_demographic_disparity(),
            self._section_failure_case_analysis(),
            self._section_recommendations(),
        ]

        report = "\n".join(sections)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        return report

    # --- Private section generators ---

    def _section_executive_summary(self) -> str:
        lines = ["## 1. Executive Summary", ""]

        from ..calibration.metrics import compute_balanced_accuracy

        overall_acc = self._df["is_correct"].mean()
        overall_bal_acc = compute_balanced_accuracy(
            self._df["is_correct"].values.astype(int),
            self._df["ground_truth"].values,
        )
        bal_acc_str = f" | **Balanced:** {overall_bal_acc:.1%}" if overall_bal_acc is not None else ""
        lines.append(f"- **Overall accuracy:** {overall_acc:.1%}{bal_acc_str}")

        for group in self._groups:
            g_df = self._df[self._df["demographic_group"] == group]
            acc = g_df["is_correct"].mean()
            bal_acc = compute_balanced_accuracy(
                g_df["is_correct"].values.astype(int),
                g_df["ground_truth"].values,
            )
            bal_str = f", balanced: {bal_acc:.1%}" if bal_acc is not None else ""
            lines.append(
                f"- **{group.capitalize()} samples:** {len(g_df)} "
                f"(accuracy: {acc:.1%}{bal_str})"
            )

        lines.append("")
        lines.append("### Confidence Signal Status")
        lines.append("")

        headers = ["Signal", "Mean", "Std", "Range", "Status"]
        rows = []
        for col in self._conf_cols:
            vals = self._df[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            label = col.replace("_confidence", "").replace("_", " ")
            status = "DEGENERATE" if self._degenerate.get(col, False) else "INFORMATIVE"
            rows.append([
                label,
                f"{vals.mean():.4f}",
                f"{vals.std():.4f}",
                f"{vals.min():.4f}-{vals.max():.4f}",
                f"**{status}**",
            ])

        lines.append(self._format_table(headers, rows))

        informative = [
            c for c in self._conf_cols if not self._degenerate.get(c, False)
        ]
        if informative:
            labels = [c.replace("_confidence", "").replace("_", " ") for c in informative]
            lines.append("")
            lines.append(
                f"**Key finding:** Only {', '.join(labels)} provide(s) meaningful "
                f"variance for calibration analysis. Subsequent demographic comparisons "
                f"use the most informative signal."
            )
        else:
            lines.append("")
            lines.append(
                "**Warning:** All confidence signals are degenerate (near-constant). "
                "Calibration analysis may not be meaningful."
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _section_confidence_signal_comparison(self) -> str:
        lines = ["## 2. Confidence Signal Comparison", ""]

        # Distribution statistics
        lines.append("### Distribution Statistics")
        lines.append("")
        headers = ["Signal", "Mean", "Median", "Std", "IQR", "Min", "Max"]
        rows = []
        for col in self._conf_cols:
            vals = self._df[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            label = col.replace("_confidence", "").replace("_", " ")
            q25, q75 = np.percentile(vals, [25, 75])
            rows.append([
                label,
                f"{vals.mean():.4f}",
                f"{np.median(vals):.4f}",
                f"{vals.std():.4f}",
                f"{q25:.4f}-{q75:.4f}",
                f"{vals.min():.4f}",
                f"{vals.max():.4f}",
            ])
        lines.append(self._format_table(headers, rows))

        # Discrimination power
        lines.append("")
        lines.append("### Discrimination Power")
        lines.append("")
        headers = ["Signal", "Spearman rho", "p-value", "AUROC"]
        rows = []
        for col in self._conf_cols:
            valid = self._df[col].notna()
            vals = self._df.loc[valid, col].values.astype(float)
            correct = self._df.loc[valid, "is_correct"].values.astype(int)
            if len(vals) < 3:
                continue
            label = col.replace("_confidence", "").replace("_", " ")

            rho, p = spearmanr(vals, correct)
            auroc = compute_auroc(vals, correct)
            auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A"

            rows.append([label, f"{rho:.3f}", f"{p:.3g}", auroc_str])
        lines.append(self._format_table(headers, rows))

        # Signal disconnect
        if (
            "logprob_confidence" in self._df.columns
            and "verbalized_confidence" in self._df.columns
        ):
            both_valid = (
                self._df["logprob_confidence"].notna()
                & self._df["verbalized_confidence"].notna()
            )
            if both_valid.sum() > 0:
                lp = self._df.loc[both_valid, "logprob_confidence"].values.astype(float)
                vb = self._df.loc[both_valid, "verbalized_confidence"].values.astype(float)
                correct = self._df.loc[both_valid, "is_correct"].values.astype(bool)

                lines.append("")
                lines.append("### Signal Disconnect")
                lines.append("")
                lines.append(
                    f"- Mean |verbalized - logprob| per sample: "
                    f"**{np.mean(np.abs(vb - lp)):.3f}**"
                )
                if (~correct).sum() > 0:
                    lines.append(
                        f"- Among incorrect predictions: mean verbalized = "
                        f"{vb[~correct].mean():.3f}, mean logprob = "
                        f"{lp[~correct].mean():.6f}"
                    )
                if correct.sum() > 0:
                    lines.append(
                        f"- Among correct predictions: mean verbalized = "
                        f"{vb[correct].mean():.3f}, mean logprob = "
                        f"{lp[correct].mean():.6f}"
                    )
                lines.append(
                    "- The model's token probabilities do not reflect its "
                    "stated uncertainty."
                )

        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _section_overconfidence_analysis(self) -> str:
        lines = ["## 3. Overconfidence Analysis", ""]

        # Logprob overconfidence
        if "logprob_confidence" in self._df.columns:
            valid = self._df["logprob_confidence"].notna()
            confs = self._df.loc[valid, "logprob_confidence"].values.astype(float)
            correct = self._df.loc[valid, "is_correct"].values.astype(int)

            if len(confs) > 0:
                lines.append("### Logprob Confidence")
                lines.append("")
                for thresh in [0.99, 0.999, 0.9999]:
                    mask = confs > thresh
                    pct = mask.mean()
                    err = 1.0 - correct[mask].mean() if mask.sum() > 0 else 0.0
                    lines.append(
                        f"- {pct:.1%} of samples have logprob > {thresh}"
                    )
                    if mask.sum() > 0:
                        lines.append(
                            f"  - **Error rate among these: {err:.1%}** "
                            f"({int((1 - correct[mask]).sum())}/{mask.sum()} wrong)"
                        )
                lines.append("")

        # Verbalized confidence binned error rates
        if "verbalized_confidence" in self._df.columns:
            valid = self._df["verbalized_confidence"].notna()
            if valid.sum() > 0:
                lines.append("### Verbalized Confidence Error Rates by Bin")
                lines.append("")

                bin_data = self._compute_binned_error_rates(
                    "verbalized_confidence", [0.5, 0.7, 0.8, 0.9, 1.01]
                )
                headers = ["Confidence Range", "N Samples", "Error Rate"]
                rows = [
                    [d["bin_label"], str(d["n_samples"]), f"{d['error_rate']:.1%}"]
                    for d in bin_data
                    if d["n_samples"] > 0
                ]
                lines.append(self._format_table(headers, rows))

                # Check if there's meaningful calibration
                if len(bin_data) >= 2:
                    err_rates = [d["error_rate"] for d in bin_data if d["n_samples"] > 0]
                    if len(err_rates) >= 2 and err_rates[0] > err_rates[-1]:
                        lines.append("")
                        lines.append(
                            "Verbalized confidence shows meaningful calibration: "
                            "higher confidence correlates with lower error rates."
                        )
                lines.append("")

        # Cross-signal analysis
        if (
            "logprob_confidence" in self._df.columns
            and "verbalized_confidence" in self._df.columns
        ):
            both = (
                self._df["logprob_confidence"].notna()
                & self._df["verbalized_confidence"].notna()
            )
            if both.sum() > 0:
                lp = self._df.loc[both, "logprob_confidence"].values.astype(float)
                vb = self._df.loc[both, "verbalized_confidence"].values.astype(float)

                low_verbal = vb < 0.7
                if low_verbal.sum() > 0:
                    lines.append("### Cross-Signal Analysis")
                    lines.append("")
                    lines.append(
                        f"- When verbalized confidence < 0.7 "
                        f"(n={low_verbal.sum()}, model states uncertainty): "
                        f"mean logprob = {lp[low_verbal].mean():.6f} "
                        f"(no corresponding uncertainty in token probs)"
                    )
                    lines.append("")

        # Self-consistency analysis
        sc_cols = [c for c in self._conf_cols if c.startswith("sc_confidence")]
        for col in sc_cols:
            valid = self._df[col].notna()
            confs = self._df.loc[valid, col].values.astype(float)
            correct = self._df.loc[valid, "is_correct"].values.astype(int)

            if len(confs) == 0:
                continue

            label = col.replace("_confidence", "").replace("_", " ")
            at_one = confs >= 1.0
            pct_one = at_one.mean()
            lines.append(f"### Self-Consistency ({label})")
            lines.append("")
            lines.append(f"- {pct_one:.1%} of samples have score = 1.0")

            if at_one.sum() > 0:
                err_at_one = 1.0 - correct[at_one].mean()
                lines.append(f"- Error rate at 1.0: {err_at_one:.1%}")

            if (~at_one).sum() > 0:
                err_below = 1.0 - correct[~at_one].mean()
                lines.append(f"- Error rate below 1.0: {err_below:.1%}")
            lines.append("")

        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _section_demographic_disparity(self) -> str:
        lines = ["## 4. Demographic Disparity Analysis", ""]

        if len(self._groups) < 2:
            lines.append("Insufficient demographic groups for comparison.")
            lines.append("")
            return "\n".join(lines)

        # Accuracy table with bootstrap CIs
        lines.append("### Accuracy")
        lines.append("")
        headers = ["Group", "Accuracy", "95% CI", "Balanced Acc", "95% CI", "N"]
        rows = []
        bootstrap_by_group = {}

        best_col = self._get_best_confidence_column()

        for group in self._groups:
            g_df = self._df[self._df["demographic_group"] == group]
            valid = g_df[best_col].notna()
            confs = g_df.loc[valid, best_col].values.astype(float)
            correct = g_df.loc[valid, "is_correct"].values.astype(int)
            labels = g_df.loc[valid, "ground_truth"].values
            n = len(correct)

            if n >= 10:
                bs = bootstrap_all_metrics(
                    confs, correct,
                    n_bootstrap=self.n_bootstrap,
                    ci_level=self.ci_level,
                    random_seed=self.random_seed,
                    labels=labels,
                )
                bootstrap_by_group[group] = bs
                acc = bs["accuracy"]
                row = [
                    group.capitalize(),
                    f"{acc.point_estimate:.1%}",
                    f"[{acc.ci_lower:.1%}, {acc.ci_upper:.1%}]",
                ]
                if "balanced_accuracy" in bs:
                    bal = bs["balanced_accuracy"]
                    row.append(f"{bal.point_estimate:.1%}")
                    row.append(f"[{bal.ci_lower:.1%}, {bal.ci_upper:.1%}]")
                else:
                    row.extend(["N/A", "N/A"])
                row.append(str(n))
                rows.append(row)
            else:
                acc_val = correct.mean() if n > 0 else 0
                rows.append([
                    group.capitalize(), f"{acc_val:.1%}", "N/A (too few)",
                    "N/A", "N/A", str(n),
                ])

        lines.append(self._format_table(headers, rows))

        # Statistical test for accuracy differences across groups
        if len(self._groups) >= 2:
            p_val, test_name = self._test_group_difference_accuracy()
            lines.append("")
            sig = "significant" if p_val < 0.05 else "not significant"
            lines.append(
                f"{test_name}: p = {p_val:.3f} ({sig} at alpha=0.05)"
            )

        # Calibration table using best confidence signal
        lines.append("")
        lines.append(
            f"### Calibration ({best_col.replace('_confidence', '').replace('_', ' ')})"
        )
        lines.append("")
        headers = ["Group", "ECE", "Brier", "AUROC"]
        rows = []
        for group in self._groups:
            if group in bootstrap_by_group:
                bs = bootstrap_by_group[group]
                ece = bs["ece"]
                brier = bs["brier"]
                auroc = bs["auroc"]
                auroc_str = (
                    f"{auroc.point_estimate:.3f}"
                    if not np.isnan(auroc.point_estimate)
                    else "N/A"
                )
                rows.append([
                    group.capitalize(),
                    f"{ece.point_estimate:.3f} [{ece.ci_lower:.3f}, {ece.ci_upper:.3f}]",
                    f"{brier.point_estimate:.3f} [{brier.ci_lower:.3f}, {brier.ci_upper:.3f}]",
                    auroc_str,
                ])
        lines.append(self._format_table(headers, rows))

        # Bootstrap calibration difference test
        group_keys = list(bootstrap_by_group.keys())
        if len(group_keys) >= 2:
            diff_test = test_calibration_difference(
                bootstrap_by_group[group_keys[0]],
                bootstrap_by_group[group_keys[1]],
                metric="ece",
            )
            lines.append("")
            lines.append(
                f"Bootstrap calibration difference test (ECE): "
                f"p = {diff_test['p_value']:.3f}"
            )

        # Calibration gap
        gap = self.analyzer.compute_calibration_gap(best_col)
        if "max_gap" in gap:
            lines.append("")
            lines.append(f"### Calibration Gap")
            lines.append("")
            lines.append(f"Max ECE gap: {gap['max_gap']:.4f}")
            if "worst_calibrated" in gap:
                lines.append(
                    f"Worst calibrated group: {gap['worst_calibrated']}"
                )

        # Equity summary across confidence signals
        if len(self._conf_cols) >= 2:
            equity_df = self.analyzer.compute_equity_summary(self._conf_cols)
            if not equity_df.empty:
                lines.append("")
                lines.append("### Calibration Equity Across Confidence Signals")
                lines.append("")
                lines.append(
                    "Comparison of how uniformly each confidence signal is "
                    "calibrated across demographic groups (sorted by most "
                    "equitable first):"
                )
                lines.append("")

                # Build table
                ece_group_cols = [
                    c for c in equity_df.columns if c.startswith("ece_")
                ]
                eq_headers = ["Signal"] + [
                    c.replace("ece_", "ECE ").capitalize()
                    for c in ece_group_cols
                ] + ["Mean ECE", "Max Gap", "Worst Group"]
                eq_rows = []
                for _, row in equity_df.iterrows():
                    label = (
                        row["confidence_type"]
                        .replace("_confidence", "")
                        .replace("_", " ")
                    )
                    eq_row = [label]
                    for c in ece_group_cols:
                        eq_row.append(f"{row[c]:.4f}")
                    eq_row.append(f"{row['mean_ece']:.4f}")
                    eq_row.append(f"{row['max_ece_gap']:.4f}")
                    eq_row.append(str(row["worst_group"]))
                    eq_rows.append(eq_row)

                lines.append(self._format_table(eq_headers, eq_rows))

                most_eq = equity_df.iloc[0]
                least_eq = equity_df.iloc[-1]
                lines.append("")
                lines.append(
                    f"**Most equitable signal:** "
                    f"{most_eq['confidence_type'].replace('_confidence', '').replace('_', ' ')} "
                    f"(max ECE gap: {most_eq['max_ece_gap']:.4f}). "
                    f"**Least equitable:** "
                    f"{least_eq['confidence_type'].replace('_confidence', '').replace('_', ' ')} "
                    f"(max ECE gap: {least_eq['max_ece_gap']:.4f})."
                )

        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _section_failure_case_analysis(self) -> str:
        lines = ["## 5. Failure Case Analysis", ""]

        best_col = self._get_best_confidence_column()
        col_label = best_col.replace("_confidence", "").replace("_", " ")

        headers = [
            "Group", "Total Wrong",
            f"Mean {col_label} conf (wrong)",
            "High-Conf Failures (>0.8)",
            "High-Conf Failure Rate",
        ]
        rows = []

        for group in self._groups:
            g_df = self._df[self._df["demographic_group"] == group]
            wrong = g_df[~g_df["is_correct"]]
            total_wrong = len(wrong)

            valid_wrong = wrong[best_col].notna()
            if valid_wrong.sum() > 0:
                mean_conf = wrong.loc[valid_wrong, best_col].mean()
            else:
                mean_conf = float("nan")

            # High confidence failures
            valid_all = g_df[best_col].notna()
            high_conf_wrong = (
                (~g_df["is_correct"])
                & valid_all
                & (g_df[best_col] > 0.8)
            ).sum()
            rate = high_conf_wrong / len(g_df) if len(g_df) > 0 else 0

            rows.append([
                group.capitalize(),
                str(total_wrong),
                f"{mean_conf:.3f}" if not np.isnan(mean_conf) else "N/A",
                str(int(high_conf_wrong)),
                f"{rate:.1%}",
            ])

        lines.append(self._format_table(headers, rows))

        # Test if overconfidence failures are disproportionate
        if len(self._groups) >= 2:
            counts = []
            for group in self._groups:
                g_df = self._df[self._df["demographic_group"] == group]
                valid = g_df[best_col].notna()
                hc_wrong = int(((~g_df["is_correct"]) & valid & (g_df[best_col] > 0.8)).sum())
                hc_right_or_low = int(len(g_df) - hc_wrong)
                counts.append([hc_wrong, hc_right_or_low])

            if all(c[0] > 0 for c in counts):
                table = np.array(counts)
                if len(self._groups) == 2:
                    _, p_val = fisher_exact(table)
                    test_name = "Fisher's exact test"
                else:
                    chi2, p_val, _, _ = chi2_contingency(table)
                    test_name = "Chi-squared test"
                lines.append("")
                lines.append(
                    f"{test_name} for disproportionate high-confidence "
                    f"failures across groups: p = {p_val:.3f}"
                )

        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _section_recommendations(self) -> str:
        lines = ["## 6. Recommendations", ""]

        best_col = self._get_best_confidence_column()
        best_label = best_col.replace("_confidence", "").replace("_", " ")

        # Signal recommendation
        degenerate_list = [
            c.replace("_confidence", "").replace("_", " ")
            for c, is_deg in self._degenerate.items()
            if is_deg
        ]
        lines.append(
            f"1. **Use {best_label} confidence** as the primary calibration "
            f"signal for this model. It provides the best discrimination "
            f"between correct and incorrect predictions."
        )

        if degenerate_list:
            lines.append(
                f"2. **{', '.join(d.capitalize() for d in degenerate_list)} "
                f"confidence {'is' if len(degenerate_list) == 1 else 'are'} "
                f"not informative** for this model — "
                f"{'it produces' if len(degenerate_list) == 1 else 'they produce'} "
                f"near-constant values regardless of prediction correctness."
            )

        # Demographic disparity
        if len(self._groups) >= 2:
            p_val, test_name = self._test_group_difference_accuracy()
            if p_val < 0.05:
                lines.append(
                    f"3. **Demographic disparity is statistically significant** "
                    f"({test_name}, p={p_val:.3f}). The model performs differently "
                    f"across skin tone groups."
                )
            else:
                lines.append(
                    f"3. **Demographic disparity is not statistically significant** "
                    f"({test_name}, p={p_val:.3f}) at the 0.05 level, though the "
                    f"sample size may limit power to detect real differences."
                )

        # Equity recommendation
        if len(self._groups) >= 2 and len(self._conf_cols) >= 2:
            equity_df = self.analyzer.compute_equity_summary(self._conf_cols)
            if not equity_df.empty:
                most_equitable = equity_df.iloc[0]["confidence_type"]
                gap = equity_df.iloc[0]["max_ece_gap"]
                label = most_equitable.replace("_confidence", "").replace("_", " ")
                rec_num = 4 if degenerate_list else 3
                lines.append(
                    f"{rec_num + 1}. **Most equitable confidence signal:** "
                    f"{label} (max ECE gap across groups: {gap:.4f}). "
                    f"Consider prioritizing this signal for fairness-sensitive "
                    f"applications."
                )

        # Temperature scaling
        lines.append(
            f"4. **Temperature scaling** on {best_label} confidence could "
            f"potentially reduce ECE. Consider fitting per-group temperature "
            f"parameters using the existing `fit_demographic_temperature_scaling` "
            f"functionality."
        )

        lines.append("")
        return "\n".join(lines)

    # --- Utility methods ---

    def _detect_confidence_columns(self) -> list[str]:
        base = ["logprob_confidence", "verbalized_confidence"]
        sc = [c for c in self._df.columns if c.startswith("sc_confidence_t")]
        emb = [c for c in self._df.columns if c.startswith("emb_consistency_t")]
        return [c for c in base + sc + emb if c in self._df.columns]

    def _detect_degenerate_signals(self) -> dict[str, bool]:
        result = {}
        for col in self._conf_cols:
            vals = self._df[col].dropna().values.astype(float)
            if len(vals) == 0:
                result[col] = True
            else:
                result[col] = vals.std() < 0.01 or (vals.max() - vals.min()) < 0.05
        return result

    def _get_best_confidence_column(self) -> str:
        best_col = "verbalized_confidence"
        best_rho = -1.0

        for col in self._conf_cols:
            if self._degenerate.get(col, False):
                continue
            valid = self._df[col].notna()
            vals = self._df.loc[valid, col].values.astype(float)
            correct = self._df.loc[valid, "is_correct"].values.astype(int)
            if len(vals) < 3:
                continue
            rho, _ = spearmanr(vals, correct)
            if abs(rho) > best_rho:
                best_rho = abs(rho)
                best_col = col

        if best_col not in self._df.columns:
            # Fallback to first available
            for col in self._conf_cols:
                if col in self._df.columns:
                    return col
        return best_col

    def _compute_binned_error_rates(
        self,
        confidence_col: str,
        bin_edges: list[float],
    ) -> list[dict]:
        valid = self._df[confidence_col].notna()
        confs = self._df.loc[valid, confidence_col].values.astype(float)
        correct = self._df.loc[valid, "is_correct"].values.astype(int)

        results = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= lo) & (confs < hi)
            n = mask.sum()
            hi_display = min(hi, 1.0)
            results.append({
                "bin_label": f"{lo:.2f}-{hi_display:.2f}",
                "n_samples": int(n),
                "n_errors": int((1 - correct[mask]).sum()) if n > 0 else 0,
                "error_rate": float(1.0 - correct[mask].mean()) if n > 0 else 0.0,
                "mean_confidence": float(confs[mask].mean()) if n > 0 else 0.0,
            })
        return results

    def _test_group_difference_accuracy(self) -> tuple[float, str]:
        """Run Fisher's exact (2 groups) or chi-squared (3+) on accuracy."""
        table = []
        for group in self._groups:
            g_df = self._df[self._df["demographic_group"] == group]
            table.append(
                [int(g_df["is_correct"].sum()), int((~g_df["is_correct"]).sum())]
            )
        table = np.array(table)
        if len(self._groups) == 2:
            _, p_val = fisher_exact(table)
            return p_val, "Fisher's exact test"
        else:
            _, p_val, _, _ = chi2_contingency(table)
            return p_val, "Chi-squared test"

    def _format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
