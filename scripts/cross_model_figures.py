"""Cross-model and cross-modality aggregate figures for the paper.

Loads multiple `results.json` files (one per model x dataset combination) and
produces publication-quality aggregate figures that the existing per-model
analysis pipeline cannot. Run after generating per-model results JSONs.

Usage:
    python scripts/cross_model_figures.py \
        --results gpt4o:ddi=outputs/gpt4o_ddi_calibration/results.json \
        --results gpt4o:pneumonia=outputs/chexpert_pneumonia_gpt4o/results.json \
        --results gpt4o:effusion=outputs/chexpert_effusion_gpt4o/results.json \
        --results r1:ddi=outputs/r1_ddi_calibration/results.json \
        --results r1:pneumonia=outputs/chexpert_pneumonia_r1/results.json \
        --results r1:effusion=outputs/chexpert_effusion_r1/results.json \
        --results qwen:ddi=outputs/qwen_vl_ddi_calibration/results.json \
        --results qwen:pneumonia=outputs/chexpert_pneumonia_qwen_vl/results.json \
        --results qwen:effusion=outputs/chexpert_effusion_qwen_vl/results.json \
        -o outputs/cross_model_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make the package importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from democalib.analysis.demographic_analysis import DemographicAnalyzer
from democalib.utils.io import load_results

# -----------------------------------------------------------------------------
# Visual style — colorblind-friendly, paper-ready
# -----------------------------------------------------------------------------
# Embed TrueType (Type 42) fonts instead of matplotlib's default Type 3;
# AAAI camera-ready compliance checks reject PDFs containing Type 3 fonts.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_COLORS = {
    "gpt4o": "#1f77b4",   # blue
    "r1":    "#d62728",   # red
    "qwen":  "#2ca02c",   # green
}
MODEL_LABELS = {"gpt4o": "GPT-4o", "r1": "R1-Onevision", "qwen": "Qwen2.5-VL"}
DATASET_LABELS = {"ddi": "DDI", "pneumonia": "Pneumonia", "effusion": "Effusion"}
SIGNAL_LABELS = {
    "logprob_confidence": "TLP",
    "verbalized_confidence": "CE",
    "sc_confidence_t0.5": "SC$_{0.5}$",
    "sc_confidence_t1.0": "SC$_{1.0}$",
}
SIGNAL_MARKERS = {"TLP": "o", "CE": "s", "SC$_{0.5}$": "^", "SC$_{1.0}$": "D"}


# -----------------------------------------------------------------------------
# Aggregator: build one long-form DataFrame from many results.json files
# -----------------------------------------------------------------------------
def build_aggregate_df(results_specs: dict[tuple[str, str], Path]) -> pd.DataFrame:
    """Compute per-(model, dataset, signal, group) calibration metrics.

    Returns long-form DataFrame with columns:
        model, dataset, signal, group, ece, brier, auroc, n
    """
    rows = []

    for (model, dataset), results_path in results_specs.items():
        if not results_path.exists():
            print(f"  [skip] {model}/{dataset}: {results_path} not found")
            continue

        results = load_results(results_path)
        analyzer = DemographicAnalyzer(results)
        df = analyzer.df

        # Detect available confidence columns.
        conf_cols = [
            c for c in [
                "logprob_confidence",
                "verbalized_confidence",
                "sc_confidence_t0.5",
                "sc_confidence_t1.0",
            ]
            if c in df.columns and df[c].notna().any()
        ]

        for conf_col in conf_cols:
            try:
                metrics_by_group = analyzer.compute_metrics_by_group(conf_col)
            except Exception as e:
                print(f"  [warn] {model}/{dataset}/{conf_col}: {e}")
                continue

            for group, m in metrics_by_group.items():
                if group == "unknown":
                    continue
                rows.append({
                    "model":   model,
                    "dataset": dataset,
                    "signal":  SIGNAL_LABELS.get(conf_col, conf_col),
                    "group":   group,
                    "ece":     getattr(m, "ece", float("nan")),
                    "brier":   getattr(m, "brier", float("nan")),
                    "auroc":   getattr(m, "auroc", float("nan")),
                    "n":       getattr(m, "n_samples", 0),
                })

    return pd.DataFrame(rows)


def compute_equity_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, dataset, signal): mean ECE, max ECE gap, ECE ratio across groups."""
    rows = []
    for (model, dataset, signal), grp in agg_df.groupby(["model", "dataset", "signal"]):
        eces = grp["ece"].dropna().values
        aurocs = grp["auroc"].dropna().values
        if len(eces) < 2:
            continue
        rows.append({
            "model":    model,
            "dataset":  dataset,
            "signal":   signal,
            "mean_ece": float(np.mean(eces)),
            "max_ece":  float(np.max(eces)),
            "min_ece":  float(np.min(eces)),
            "ece_gap":  float(np.max(eces) - np.min(eces)),
            "ece_ratio": float(np.max(eces) / max(np.min(eces), 1e-6)),
            "mean_auroc": float(np.mean(aurocs)) if len(aurocs) else float("nan"),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure 1: Calibration-Equity Tradeoff (HERO)
# -----------------------------------------------------------------------------
def plot_calibration_equity_scatter(
    equity_df: pd.DataFrame,
    save_path: Path,
    figsize: tuple[float, float] = (7, 5),
) -> None:
    """Small-multiples paired bars: mean ECE (aggregate) vs max ECE gap (equity).

    Grid of (model x dataset) panels. Within each panel, two bars per signal
    side by side — light = mean ECE, dark = max gap. Communicates the tradeoff
    by showing both metrics for the same (model, dataset, signal) condition.
    """
    models = ["gpt4o", "r1", "qwen"]
    datasets = ["ddi", "pneumonia", "effusion"]
    signal_order = ["TLP", "CE", "SC$_{0.5}$", "SC$_{1.0}$"]

    fig, axes = plt.subplots(len(models), len(datasets), figsize=figsize, sharey=True)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            sub = equity_df[(equity_df["model"] == model) &
                            (equity_df["dataset"] == dataset)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sub = sub.set_index("signal").reindex(
                [s for s in signal_order if s in sub["signal"].values]
            ).reset_index()

            x = np.arange(len(sub))
            bw = 0.38
            color = MODEL_COLORS.get(model, "gray")
            ax.bar(x - bw/2, sub["mean_ece"], bw,
                   color=color, alpha=0.45, edgecolor="black", linewidth=0.4,
                   label="Mean ECE")
            ax.bar(x + bw/2, sub["ece_gap"], bw,
                   color=color, alpha=1.0, edgecolor="black", linewidth=0.4,
                   label="Max ECE gap")

            ax.set_xticks(x)
            ax.set_xticklabels(sub["signal"], fontsize=9, rotation=0)
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=9)

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset),
                             fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model), fontsize=10,
                              fontweight="bold")
            if i == 0 and j == len(datasets) - 1:
                ax.legend(fontsize=9, loc="upper right",
                          framealpha=0.95)

    fig.suptitle("Mean ECE (light) vs Max ECE Gap (dark), by Model and Dataset",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 2: Cross-Domain AUROC Heatmap
# -----------------------------------------------------------------------------
def plot_cross_domain_auroc_heatmap(
    agg_df: pd.DataFrame,
    save_path: Path,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Small-multiples bar chart: AUROC by signal, faceted (model x dataset).

    Bars below the chance line (AUROC = 0.5) are colored red to flag
    anti-discriminative signals. Same small-multiples layout as figs 3 and 5.
    """
    agg = (agg_df.groupby(["model", "dataset", "signal"])["auroc"]
                  .mean().reset_index())

    models = ["gpt4o", "r1", "qwen"]
    datasets = ["ddi", "pneumonia", "effusion"]
    signal_order = ["TLP", "CE", "SC$_{0.5}$", "SC$_{1.0}$"]

    fig, axes = plt.subplots(len(models), len(datasets), figsize=figsize, sharey=True)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            sub = agg[(agg["model"] == model) & (agg["dataset"] == dataset)]
            if sub.empty:
                ax.set_axis_off()
                continue
            sub = sub.set_index("signal").reindex(
                [s for s in signal_order if s in sub["signal"].values]
            ).reset_index()

            x = np.arange(len(sub))
            colors = [
                "#9e9e9e" if a < 0.5 else MODEL_COLORS.get(model, "gray")
                for a in sub["auroc"]
            ]
            ax.bar(x, sub["auroc"], 0.7,
                   color=colors, edgecolor="black", linewidth=0.5)
            # Chance line.
            ax.axhline(0.5, color="black", linestyle="--", lw=0.8, alpha=0.7)
            # Annotate values above each bar.
            for k, v in enumerate(sub["auroc"]):
                ax.text(k, v + 0.02, f"{v:.2f}", ha="center",
                        fontsize=8, color="#333")

            ax.set_xticks(x)
            ax.set_xticklabels(sub["signal"], fontsize=8)
            ax.set_ylim(0.3, 0.9)
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=8)

            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset),
                             fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nAUROC", fontsize=9)

    fig.suptitle("AUROC by Confidence Signal, Model, and Dataset "
                 "(dashed line = chance; gray = anti-discriminative)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 3: Confidence Distribution Grid (3 models x 4 signals)
# -----------------------------------------------------------------------------
def plot_signal_distribution_grid(
    results_specs: dict[tuple[str, str], Path],
    save_path: Path,
    dataset_filter: str = "ddi",
    figsize: tuple[float, float] = (12, 9),
) -> None:
    """Grid of confidence distributions: rows = models, columns = signals.

    Uses KDE + rug for non-degenerate signals (handles edge spikes gracefully),
    and a clear bar+annotation for degenerate ones (Qwen's SC = 1.0 everywhere).
    """
    from scipy.stats import gaussian_kde

    models = ["gpt4o", "r1", "qwen"]
    signal_cols = [
        "logprob_confidence",
        "verbalized_confidence",
        "sc_confidence_t1.0",
    ]
    n_rows, n_cols = len(models), len(signal_cols)
    # Important: do NOT share y-axis. Each panel has its own scale.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False)

    # Slightly wider x-range so spikes at 0 or 1 are visible (avoids edge clipping).
    XLIM = (-0.03, 1.03)

    for i, model in enumerate(models):
        path = results_specs.get((model, dataset_filter))
        if not path or not path.exists():
            for j in range(n_cols):
                axes[i, j].text(0.5, 0.5, f"No data\n{model}/{dataset_filter}",
                                ha="center", va="center", transform=axes[i, j].transAxes)
                axes[i, j].set_axis_off()
            continue

        results = load_results(path)
        analyzer = DemographicAnalyzer(results)
        df = analyzer.df

        for j, conf_col in enumerate(signal_cols):
            ax = axes[i, j]
            if conf_col not in df.columns or not df[conf_col].notna().any():
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=11, fontstyle="italic")
                ax.set_xlim(*XLIM)
                continue

            sub = df[[conf_col, "is_correct"]].dropna()
            v_std = sub[conf_col].std()

            if v_std < 1e-3:
                # Degenerate: show a visible vertical bar at the spike + clear annotation.
                v = sub[conf_col].iloc[0]
                ax.axvspan(v - 0.02, v + 0.02, color="lightgray", alpha=0.7)
                ax.axvline(v, color="black", lw=2)
                ax.text(0.5, 0.55,
                        f"DEGENERATE\nall {len(sub)} samples\nspike at c = {v:.2f}",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#444", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="white",
                                  edgecolor="gray", alpha=0.9))
                ax.set_ylim(0, 1)
                ax.set_yticks([])
            else:
                correct = sub[sub["is_correct"]][conf_col].values
                wrong = sub[~sub["is_correct"]][conf_col].values

                xs = np.linspace(0, 1, 200)
                # Robust KDE — fall back to histogram if too few samples.
                try:
                    if len(correct) >= 5:
                        kde_c = gaussian_kde(correct, bw_method=0.15)
                        ax.fill_between(xs, kde_c(xs), alpha=0.5, color="#2ca02c",
                                        label=f"Correct (n={len(correct)})")
                    if len(wrong) >= 5:
                        kde_w = gaussian_kde(wrong, bw_method=0.15)
                        ax.fill_between(xs, kde_w(xs), alpha=0.5, color="#d62728",
                                        label=f"Wrong (n={len(wrong)})")
                except Exception:
                    bins = np.linspace(0, 1, 20)
                    ax.hist(correct, bins=bins, alpha=0.5, color="#2ca02c",
                            label=f"Correct (n={len(correct)})", density=True)
                    ax.hist(wrong, bins=bins, alpha=0.5, color="#d62728",
                            label=f"Wrong (n={len(wrong)})", density=True)

                # Rug plot at the bottom showing actual sample positions.
                ymin, ymax = ax.get_ylim() if ax.get_ylim()[1] > 0 else (0, 1)
                rug_y = -0.03 * (ymax - ymin)
                ax.scatter(correct, [rug_y] * len(correct), marker="|",
                           color="#2ca02c", s=20, alpha=0.4, clip_on=False)
                ax.scatter(wrong, [rug_y] * len(wrong), marker="|",
                           color="#d62728", s=20, alpha=0.4, clip_on=False)

                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc="upper left")

            if i == 0:
                ax.set_title(SIGNAL_LABELS[conf_col], fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nDensity", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Confidence", fontsize=10)
            ax.set_xlim(*XLIM)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Confidence Distributions by Model and Signal on {DATASET_LABELS.get(dataset_filter, dataset_filter)}",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 4: Class Imbalance vs ECE
# -----------------------------------------------------------------------------
def plot_class_imbalance_effect(
    agg_df: pd.DataFrame,
    save_path: Path,
    figsize: tuple[float, float] = (12, 4.5),
) -> None:
    """Small-multiples: ECE across datasets, one panel per model.

    Each panel: bars for each dataset (ordered by positive prevalence) showing
    TLP ECE. Same layout language as figs 3 and 5.
    """
    PREVALENCE = {"pneumonia": 3.4, "effusion": 27.7, "ddi": 47.7}
    DATASET_ORDER = ["pneumonia", "effusion", "ddi"]
    models = ["gpt4o", "r1", "qwen"]

    tlp = agg_df[agg_df["signal"] == "TLP"].copy()
    tlp_summary = (tlp.groupby(["model", "dataset"])["ece"]
                       .mean().reset_index())

    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)

    ymax = max(tlp_summary["ece"].max() * 1.2, 0.5)

    for ax, model in zip(axes, models):
        sub = tlp_summary[tlp_summary["model"] == model].set_index("dataset")
        eces = [sub.loc[d, "ece"] if d in sub.index else 0 for d in DATASET_ORDER]
        x = np.arange(len(DATASET_ORDER))
        ax.bar(x, eces, 0.65,
               color=MODEL_COLORS.get(model, "gray"),
               alpha=0.85, edgecolor="black", linewidth=0.5)
        for k, v in enumerate(eces):
            ax.text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{DATASET_LABELS[d]}\n({PREVALENCE[d]}% pos.)" for d in DATASET_ORDER
        ], fontsize=9)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("TLP ECE (mean across groups)", fontsize=10)

    fig.suptitle("TLP ECE by Dataset, One Panel per Model "
                 "(datasets ordered by positive class prevalence)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 5: Equity Ranking Across Models (small multiples)
# -----------------------------------------------------------------------------
def plot_equity_ranking_grid(
    equity_df: pd.DataFrame,
    save_path: Path,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Small-multiples bar chart of max ECE gap per signal, faceted by (model, dataset).

    Shows that the "most equitable" signal differs across (model, dataset) — there
    is no universal winner, supporting the deployment guidance.
    """
    datasets = ["ddi", "pneumonia", "effusion"]
    models = ["gpt4o", "r1", "qwen"]
    fig, axes = plt.subplots(len(models), len(datasets), figsize=figsize, sharex=True)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            sub = equity_df[(equity_df["model"] == model) &
                            (equity_df["dataset"] == dataset)]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
                continue

            sub_sorted = sub.sort_values("ece_gap")
            best_sig = sub_sorted.iloc[0]["signal"]
            colors = ["#2ca02c" if s == best_sig else "#9e9e9e"
                      for s in sub_sorted["signal"]]
            ax.barh(sub_sorted["signal"], sub_sorted["ece_gap"],
                    color=colors, edgecolor="black", linewidth=0.5)

            for k, v in enumerate(sub_sorted["ece_gap"].values):
                ax.text(v + 0.002, k, f"{v:.3f}", va="center", fontsize=8)

            if i == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=11)
            if j == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel("Max ECE gap", fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Max ECE Gap by Confidence Signal, Model, and Dataset",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 6: Reasoning vs Standard (R1 vs Qwen — same backbone, different CoT)
# -----------------------------------------------------------------------------
def plot_reasoning_vs_standard_panel(
    results_specs: dict[tuple[str, str], Path],
    save_path: Path,
    dataset_filter: str = "ddi",
    figsize: tuple[float, float] = (7, 4.5),
) -> None:
    """2x3 grid: Qwen (top) vs R1 (bottom) x (TLP, CE, SC1.0) on DDI.

    R1 and Qwen share the same Qwen2.5-VL backbone — only difference is CoT.
    The visual contrast between rows isolates the effect of reasoning training
    on confidence behavior. Designed as the single sharpest figure for the
    "reasoning models are different" headline finding.
    """
    from scipy.stats import gaussian_kde

    pair = [("qwen", "Qwen2.5-VL\n(no reasoning)"),
            ("r1",   "R1-Onevision\n(with reasoning)")]
    signal_cols = [
        ("logprob_confidence",    "TLP"),
        ("verbalized_confidence", "CE"),
        ("sc_confidence_t1.0",    "SC$_{1.0}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=False)

    XLIM = (-0.03, 1.03)

    for i, (model, model_label) in enumerate(pair):
        path = results_specs.get((model, dataset_filter))
        if not path or not path.exists():
            for j in range(len(signal_cols)):
                axes[i, j].set_axis_off()
            continue
        results = load_results(path)
        analyzer = DemographicAnalyzer(results)
        df = analyzer.df

        for j, (conf_col, sig_label) in enumerate(signal_cols):
            ax = axes[i, j]
            if conf_col not in df.columns or not df[conf_col].notna().any():
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xlim(*XLIM)
                continue

            sub = df[[conf_col, "is_correct"]].dropna()
            v_std = sub[conf_col].std()

            if v_std < 1e-3:
                v = sub[conf_col].iloc[0]
                ax.axvspan(v - 0.02, v + 0.02, color="lightgray", alpha=0.7)
                ax.axvline(v, color="black", lw=2)
                ax.text(0.5, 0.55,
                        f"DEGENERATE\nall {len(sub)} samples\nspike at c = {v:.2f}",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#444", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="white",
                                  edgecolor="gray", alpha=0.9))
                ax.set_ylim(0, 1)
                ax.set_yticks([])
            else:
                correct = sub[sub["is_correct"]][conf_col].values
                wrong = sub[~sub["is_correct"]][conf_col].values
                xs = np.linspace(0, 1, 200)
                try:
                    if len(correct) >= 5:
                        ax.fill_between(
                            xs, gaussian_kde(correct, bw_method=0.15)(xs),
                            alpha=0.5, color="#2ca02c",
                            label=f"Correct (n={len(correct)})",
                        )
                    if len(wrong) >= 5:
                        ax.fill_between(
                            xs, gaussian_kde(wrong, bw_method=0.15)(xs),
                            alpha=0.5, color="#d62728",
                            label=f"Wrong (n={len(wrong)})",
                        )
                except Exception:
                    pass

                if i == 0 and j == 0:
                    ax.legend(fontsize=9, loc="upper left")

            if i == 0:
                ax.set_title(sig_label, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(model_label, fontsize=10, fontweight="bold")
            if i == 1:
                ax.set_xlabel("Confidence", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.set_xlim(*XLIM)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Confidence Distributions: Qwen2.5-VL vs R1-Onevision on "
        f"{DATASET_LABELS.get(dataset_filter, dataset_filter)} "
        "(same backbone, R1 adds chain-of-thought)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 7: Reliability Diagrams — all signals, all models
# -----------------------------------------------------------------------------
GROUP_COLORS_DDI = {
    "light":  "#9467bd",   # purple
    "medium": "#ff7f0e",   # orange
    "dark":   "#8c564b",   # brown
}
GROUP_COLORS_CHEXPERT = {
    "male":   "#17becf",   # teal
    "female": "#e377c2",   # pink
}
GROUP_LABELS_DDI = {"light": "Light", "medium": "Medium", "dark": "Dark"}
GROUP_LABELS_CHEXPERT = {"male": "Male", "female": "Female"}


def _group_colors(dataset: str) -> dict[str, str]:
    if dataset == "ddi":
        return GROUP_COLORS_DDI
    return GROUP_COLORS_CHEXPERT


def _group_labels(dataset: str) -> dict[str, str]:
    if dataset == "ddi":
        return GROUP_LABELS_DDI
    return GROUP_LABELS_CHEXPERT


def _reliability_curve(confs, correct, bin_edges):
    """Return (bin_confs, bin_accs, bin_counts) for non-empty bins."""
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() >= 3:  # need enough samples for a stable estimate
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_counts.append(mask.sum())
    return (np.array(bin_accs), np.array(bin_confs), np.array(bin_counts))


def plot_reliability_diagrams(
    results_specs: dict[tuple[str, str], Path],
    save_path: Path,
    dataset_filter: str = "ddi",
    n_bins: int = 10,
    figsize: tuple[float, float] = (7, 7),
) -> None:
    """3x3 reliability diagrams stratified by demographic group.

    Rows = models, columns = signals (TLP, CE, SC1.0).
    Each panel shows one line per demographic group so calibration
    differences across skin tone (DDI) or sex (CheXpert) are visible.
    """
    models = ["gpt4o", "r1", "qwen"]
    signals = [
        ("logprob_confidence",    "TLP"),
        ("verbalized_confidence", "CE"),
        ("sc_confidence_t1.0",    "SC$_{1.0}$"),
    ]
    g_colors = _group_colors(dataset_filter)
    g_labels = _group_labels(dataset_filter)

    fig, axes = plt.subplots(len(models), len(signals), figsize=figsize,
                             sharex=True, sharey=True)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i, model in enumerate(models):
        path = results_specs.get((model, dataset_filter))

        df = None
        if path and path.exists():
            results = load_results(path)
            df = DemographicAnalyzer(results).df

        for j, (conf_col, sig_label) in enumerate(signals):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.45, zorder=1)

            if df is None:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
            elif conf_col not in df.columns or not df[conf_col].notna().any():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
            else:
                # correctness column: SC uses majority-vote prediction
                if conf_col.startswith("sc_confidence_t"):
                    temp = conf_col.removeprefix("sc_confidence_")
                    pred_col = f"sc_prediction_{temp}"
                    correct_col = "sc_correct"
                    if pred_col in df.columns:
                        df = df.copy()
                        df["sc_correct"] = (df[pred_col] == df["ground_truth"]).astype(float)
                    else:
                        correct_col = "is_correct"
                else:
                    correct_col = "is_correct"

                groups = sorted(
                    [g for g in df["demographic_group"].unique() if g != "unknown"],
                    key=lambda g: list(g_colors.keys()).index(g) if g in g_colors else 99,
                )

                overall_degenerate = df[conf_col].std() < 1e-3

                ece_lines = []
                for group in groups:
                    gdf = df[df["demographic_group"] == group][[conf_col, correct_col]].dropna()
                    if len(gdf) < 5 or overall_degenerate:
                        continue
                    confs = gdf[conf_col].values
                    correct = gdf[correct_col].astype(float).values
                    if confs.std() < 1e-3:
                        continue

                    ba, bc, bct = _reliability_curve(confs, correct, bin_edges)
                    if len(bc) == 0:
                        continue

                    gc = g_colors.get(group, "gray")
                    gl = g_labels.get(group, group.capitalize())
                    ax.plot(bc, ba, "-o", color=gc, linewidth=1.4,
                            markersize=3.5, zorder=3, label=gl)

                    ece = float(np.sum(bct / bct.sum() * np.abs(ba - bc)))
                    ece_lines.append(f"{gl}: {ece:.3f}")

                if overall_degenerate:
                    v = df[conf_col].iloc[0]
                    ax.text(0.5, 0.5, f"DEGENERATE\nc={v:.2f}",
                            ha="center", va="center", transform=ax.transAxes,
                            color="gray", fontsize=8)
                elif ece_lines:
                    ax.text(0.04, 0.97, "\n".join(ece_lines),
                            transform=ax.transAxes, fontsize=6.5,
                            va="top", color="black",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            if i == 0:
                ax.set_title(sig_label, fontsize=10, fontweight="bold")
            model_color = MODEL_COLORS.get(model, "gray")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nAccuracy",
                              fontsize=8, fontweight="bold", color=model_color)
            if i == len(models) - 1:
                ax.set_xlabel("Confidence", fontsize=8)

    # Shared legend for demographic groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1,
               label="Perfect calibration"),
    ]
    for g, gc in g_colors.items():
        gl = g_labels.get(g, g.capitalize())
        legend_elements.append(
            Line2D([0], [0], color=gc, marker="o", markersize=4,
                   linewidth=1.2, label=gl)
        )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), bbox_to_anchor=(0.5, -0.02),
               fontsize=8, framealpha=0.9)

    dataset_label = DATASET_LABELS.get(dataset_filter, dataset_filter)
    fig.suptitle(
        f"Reliability diagrams on {dataset_label} stratified by demographic group",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# -----------------------------------------------------------------------------
# Figure 8: Signal correlation analysis (TLP vs CE vs SC)
# -----------------------------------------------------------------------------
def plot_signal_correlations(
    results_specs: dict[tuple[str, str], Path],
    save_path: Path,
    dataset_filter: str = "ddi",
    figsize: tuple[float, float] = (7, 7),
) -> pd.DataFrame:
    """3x3 scatter plots: rows = models, cols = signal pairs (TLP-CE, TLP-SC, CE-SC).

    Each panel shows sample-level confidence values for the two signals.
    Points are colored by correctness. Diagonal = perfect agreement.
    Annotates Spearman ρ and Pearson r.

    Returns a DataFrame of correlations for the paper table.
    """
    from scipy import stats

    models = ["gpt4o", "r1", "qwen"]
    signal_cols = [
        ("logprob_confidence",    "TLP"),
        ("verbalized_confidence", "CE"),
        ("sc_confidence_t1.0",    "SC$_{1.0}$"),
    ]
    pairs = [
        (signal_cols[0], signal_cols[1]),   # TLP vs CE
        (signal_cols[0], signal_cols[2]),   # TLP vs SC
        (signal_cols[1], signal_cols[2]),   # CE vs SC
    ]

    fig, axes = plt.subplots(len(models), len(pairs), figsize=figsize,
                             sharex=True, sharey=True)

    corr_rows = []

    for i, model in enumerate(models):
        path = results_specs.get((model, dataset_filter))
        color = MODEL_COLORS.get(model, "gray")

        df = None
        if path and path.exists():
            results = load_results(path)
            df = DemographicAnalyzer(results).df

        for j, ((col_a, lbl_a), (col_b, lbl_b)) in enumerate(pairs):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, zorder=1)

            if df is None or col_a not in df.columns or col_b not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
            else:
                sub = df[[col_a, col_b, "is_correct"]].dropna()
                if len(sub) < 5:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=9)
                else:
                    x = sub[col_a].values
                    y = sub[col_b].values
                    correct = sub["is_correct"].astype(bool).values

                    # Scatter colored by correctness
                    ax.scatter(x[correct],  y[correct],  c="#2ca02c", alpha=0.25,
                               s=6, linewidths=0, zorder=2, label="Correct")
                    ax.scatter(x[~correct], y[~correct], c="#d62728", alpha=0.25,
                               s=6, linewidths=0, zorder=2, label="Incorrect")

                    # Correlations
                    r_pearson, _ = stats.pearsonr(x, y)
                    r_spearman, _ = stats.spearmanr(x, y)

                    ax.text(0.04, 0.92,
                            f"ρ={r_spearman:.2f}  r={r_pearson:.2f}",
                            transform=ax.transAxes, fontsize=7,
                            color=color, fontweight="bold")

                    corr_rows.append({
                        "model":     MODEL_LABELS.get(model, model),
                        "dataset":   dataset_filter,
                        "signal_a":  lbl_a.replace("$", "").replace("_{", "").replace("}", ""),
                        "signal_b":  lbl_b.replace("$", "").replace("_{", "").replace("}", ""),
                        "pearson_r": round(r_pearson, 3),
                        "spearman_rho": round(r_spearman, 3),
                        "n":         len(sub),
                    })

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            if i == 0:
                ax.set_title(f"{lbl_a} vs {lbl_b}", fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\n{lbl_b}",
                              fontsize=8, fontweight="bold", color=color)
            if i == len(models) - 1:
                ax.set_xlabel(lbl_a, fontsize=8)

    # Legend (show once)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=5, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=5, label="Incorrect"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1,
               label="Perfect agreement"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Confidence signal agreement on {DATASET_LABELS.get(dataset_filter, dataset_filter)} "
        f"(Spearman ρ, Pearson r per panel)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")

    corr_df = pd.DataFrame(corr_rows)
    return corr_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_results_args(specs: list[str]) -> dict[tuple[str, str], Path]:
    """Parse '--results model:dataset=path' arguments into a dict."""
    out = {}
    for s in specs:
        try:
            key, path = s.split("=", 1)
            model, dataset = key.split(":", 1)
            out[(model.strip(), dataset.strip())] = Path(path.strip())
        except ValueError:
            print(f"  [warn] bad spec: {s} (expected 'model:dataset=path')")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", action="append", default=[],
                    help="Repeated: 'model:dataset=path/to/results.json'")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/cross_model_figures"))
    args = ap.parse_args()

    results_specs = parse_results_args(args.results)
    if not results_specs:
        ap.error("No --results specified.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(results_specs)} result files...")
    agg_df = build_aggregate_df(results_specs)
    if agg_df.empty:
        print("No valid results loaded — aborting.")
        return

    equity_df = compute_equity_table(agg_df)

    # Save raw aggregate tables for inspection / paper tables.
    agg_df.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    equity_df.to_csv(args.output_dir / "equity_summary.csv", index=False)
    print(f"  [saved] {args.output_dir / 'aggregate_metrics.csv'}")
    print(f"  [saved] {args.output_dir / 'equity_summary.csv'}")

    print("\nGenerating figures...")
    plot_calibration_equity_scatter(equity_df, args.output_dir / "fig1_calibration_equity_tradeoff.pdf")
    plot_cross_domain_auroc_heatmap(agg_df, args.output_dir / "fig2_cross_domain_auroc_heatmap.pdf")
    plot_signal_distribution_grid(results_specs, args.output_dir / "fig3_signal_distribution_grid.pdf")
    plot_class_imbalance_effect(agg_df, args.output_dir / "fig4_class_imbalance_effect.pdf")
    plot_equity_ranking_grid(equity_df, args.output_dir / "fig5_equity_ranking_grid.pdf")
    plot_reasoning_vs_standard_panel(results_specs, args.output_dir / "fig6_reasoning_vs_standard.pdf")
    plot_reliability_diagrams(results_specs, args.output_dir / "fig7a_reliability_ddi.pdf",
                              dataset_filter="ddi")
    plot_reliability_diagrams(results_specs, args.output_dir / "fig7b_reliability_effusion.pdf",
                              dataset_filter="effusion")

    corr_dfs = []
    for ds in ["ddi", "pneumonia", "effusion"]:
        cdf = plot_signal_correlations(
            results_specs,
            args.output_dir / f"fig8_signal_correlations_{ds}.pdf",
            dataset_filter=ds,
        )
        corr_dfs.append(cdf)
    corr_df = pd.concat([d for d in corr_dfs if not d.empty], ignore_index=True)
    if not corr_df.empty:
        corr_path = args.output_dir / "signal_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"  [saved] {corr_path}")
        print("\nSignal correlations (Spearman ρ):")
        pivot = corr_df.pivot_table(
            index=["model", "dataset"],
            columns=["signal_a", "signal_b"],
            values="spearman_rho",
        )
        print(pivot.to_string())
        # fig8 scatter plots are intentionally not included in paper —
        # correlation data is fully reported in tab:signal_corr

    print(f"\nAll figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
