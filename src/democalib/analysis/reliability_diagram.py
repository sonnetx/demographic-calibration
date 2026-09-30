"""Reliability diagram visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import fisher_exact, spearmanr

from ..calibration.metrics import CalibrationMetrics, compute_ece

# Canonical demographic group color palette (colorblind-friendly).
DEMOGRAPHIC_PALETTE: dict[str, str] = {
    # Skin tone groups
    "light": "#FF9800",   # orange
    "medium": "#2196F3",  # blue
    "dark": "#7B1FA2",    # purple
    # Sex groups
    "male": "#2196F3",    # blue
    "female": "#E91E63",  # pink
}

_FALLBACK_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))


def get_group_color(group: str, index: int = 0) -> str:
    """Return color for a demographic group, with fallback for unknown groups."""
    if group in DEMOGRAPHIC_PALETTE:
        return DEMOGRAPHIC_PALETTE[group]
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    title: str = "Reliability Diagram",
    ax: plt.Axes | None = None,
    show_histogram: bool = True,
    color: str = "steelblue",
    show_gap: bool = True,
) -> plt.Figure:
    """
    Plot reliability diagram with optional histogram.

    Args:
        metrics: CalibrationMetrics with bin data
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        show_histogram: Whether to show confidence histogram
        color: Bar color
        show_gap: Whether to show calibration gap visualization

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    n_bins = len(metrics.bin_confidences)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1 / n_bins

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    # Bar chart of accuracy per bin
    non_empty = metrics.bin_counts > 0
    ax.bar(
        bin_centers[non_empty],
        metrics.bin_accuracies[non_empty],
        width=bin_width * 0.9,
        alpha=0.7,
        color=color,
        edgecolor="black",
        label="Observed accuracy",
    )

    # Gap visualization
    if show_gap:
        for i in range(n_bins):
            if metrics.bin_counts[i] > 0:
                gap = metrics.bin_accuracies[i] - metrics.bin_confidences[i]
                if gap > 0:  # Underconfident
                    ax.bar(
                        bin_centers[i],
                        gap,
                        bottom=metrics.bin_confidences[i],
                        width=bin_width * 0.9,
                        alpha=0.3,
                        color="green",
                        edgecolor="none",
                    )
                else:  # Overconfident
                    ax.bar(
                        bin_centers[i],
                        -gap,
                        bottom=metrics.bin_accuracies[i],
                        width=bin_width * 0.9,
                        alpha=0.3,
                        color="red",
                        edgecolor="none",
                    )

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE = {metrics.ece:.4f}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add histogram as secondary axis if requested
    if show_histogram:
        ax2 = ax.twinx()
        ax2.bar(
            bin_centers[non_empty],
            metrics.bin_counts[non_empty] / metrics.n_samples,
            width=bin_width * 0.3,
            alpha=0.3,
            color="gray",
            label="Sample fraction",
        )
        ax2.set_ylabel("Fraction of samples", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_demographic_comparison(
    metrics_by_group: dict[str, CalibrationMetrics],
    confidence_type: str = "logprob",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot side-by-side reliability diagrams for demographic groups.

    Args:
        metrics_by_group: Dict mapping group name to CalibrationMetrics
        confidence_type: Name of confidence type for title
        save_path: Optional path to save figure
        figsize: Figure size (auto-scales with group count if None)
        title: Optional overall title

    Returns:
        Matplotlib figure
    """
    n_groups = len(metrics_by_group)
    if figsize is None:
        figsize = (max(14, 6 * n_groups), 6)
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)

    if n_groups == 1:
        axes = [axes]

    bar_color = "steelblue"

    for ax, (group_name, metrics) in zip(axes, metrics_by_group.items()):
        plot_reliability_diagram(
            metrics=metrics,
            title=f"{group_name.capitalize()} Skin\n(n={metrics.n_samples})",
            ax=ax,
            color=bar_color,
            show_gap=False,
        )

    if title is None:
        title = f"Calibration Comparison: {confidence_type.replace('_', ' ').title()} Confidence"
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_distributions(
    confidences_by_group: dict[str, np.ndarray],
    correctness_by_group: dict[str, np.ndarray],
    confidence_type: str = "logprob",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot confidence distributions for different groups.

    Args:
        confidences_by_group: Dict mapping group name to confidence arrays
        correctness_by_group: Dict mapping group name to correctness arrays
        confidence_type: Name of confidence type for title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = DEMOGRAPHIC_PALETTE
    default_colors = plt.cm.Set2(np.linspace(0, 1, len(confidences_by_group)))

    # Left plot: confidence distributions
    for idx, (group, confs) in enumerate(confidences_by_group.items()):
        color = colors.get(group, default_colors[idx])
        axes[0].hist(
            confs,
            bins=20,
            alpha=0.5,
            label=f"{group.capitalize()} (n={len(confs)})",
            color=color,
            density=True,
        )
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{confidence_type.replace('_', ' ').title()} Confidence Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right plot: confidence vs correctness
    for idx, (group, confs) in enumerate(confidences_by_group.items()):
        corr = correctness_by_group[group]
        color = colors.get(group, default_colors[idx])

        # Separate correct and incorrect
        correct_confs = confs[corr == 1]
        incorrect_confs = confs[corr == 0]

        axes[1].hist(
            correct_confs,
            bins=20,
            alpha=0.5,
            label=f"{group.capitalize()} correct",
            color=color,
            density=True,
        )
        axes[1].hist(
            incorrect_confs,
            bins=20,
            alpha=0.3,
            color=color,
            density=True,
            linestyle="--",
            edgecolor=color,
            linewidth=2,
            histtype="step",
        )

    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Correct vs Incorrect Predictions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_calibration_lines(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Calibration Plot",
    n_bins: int = 10,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    colors: dict[str, str] | None = None,
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Plot line-style calibration diagram with multiple series.

    Each series is binned into quantile deciles, then mean confidence
    is plotted against observed accuracy on a 0-100% scale.

    Args:
        series: Dict mapping label to (confidences, correctness) arrays.
            Confidences should be in [0, 1] and correctness binary 0/1.
        title: Plot title
        n_bins: Number of quantile bins (default 10 = deciles)
        save_path: Optional path to save figure
        figsize: Figure size
        colors: Optional dict mapping label to color string
        font_family: Font family for plot text

    Returns:
        Matplotlib figure
    """
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, len(series)))

    for idx, (label, (confidences, correctness)) in enumerate(series.items()):
        confidences = np.asarray(confidences, dtype=float)
        correctness = np.asarray(correctness, dtype=float)

        # Filter valid values
        valid = np.isfinite(confidences) & (confidences <= 1.0)
        confidences = confidences[valid]
        correctness = correctness[valid]

        if len(confidences) == 0:
            continue

        # Quantile binning — reduce bins if fewer samples than bins
        effective_bins = min(n_bins, len(confidences))
        if effective_bins < 2:
            continue
        try:
            bin_indices = pd.qcut(confidences, effective_bins, labels=False, duplicates="drop")
        except ValueError:
            continue

        bin_conf = []
        bin_acc = []
        for b in range(bin_indices.max() + 1):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_conf.append(confidences[mask].mean() * 100)
                bin_acc.append(correctness[mask].mean() * 100)

        color = (colors or {}).get(label, default_colors[idx])
        ax.plot(bin_conf, bin_acc, marker="o", linestyle="-", color=color,
                label=label, linewidth=5)

    ax.plot([0, 100], [0, 100], color="black", lw=2, linestyle="--",
            label="Perfect Calibration")
    ax.set_xlabel("Response Agreement (%)", fontsize=20)
    ax.set_ylabel("Observed Accuracy (%)", fontsize=20)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=28)
    ax.legend(fontsize=16, loc="upper left")
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_forest_with_table(
    data: pd.DataFrame,
    value_col: str = "AUC Total",
    error_col: str = "CI Error",
    method_col: str = "Method",
    group_col: str = "Group",
    extra_cols: list[str] | None = None,
    title: str = "ROC AUC Forest Plot",
    palette: dict[str, str] | None = None,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    cmap_range: tuple[float, float] = (0.6, 0.8),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Plot horizontal forest plot with error bars and a color-coded table.

    Args:
        data: DataFrame with columns for method, value, error, and group.
        value_col: Column name for the point estimate (float).
        error_col: Column name for the symmetric error (half CI width).
        method_col: Column name for method labels.
        group_col: Column name for grouping (used for color).
        extra_cols: Additional columns to include in the table.
        title: Plot title.
        palette: Dict mapping group name to color string.
        save_path: Optional path to save figure.
        figsize: Figure size.
        cmap_range: (vmin, vmax) for the orange colormap normalization.
        font_family: Font family.

    Returns:
        Matplotlib figure
    """
    plt.rcParams["font.family"] = font_family

    if palette is None:
        palette = {
            "CE Group": "tab:blue",
            "TLP Group": "tab:orange",
            "Other Group": "tab:green",
        }

    if extra_cols is None:
        extra_cols = []

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = range(len(data))
    colors_series = data[group_col].map(palette)

    for i, position in enumerate(y_positions):
        row = data.iloc[i]
        color = colors_series.iloc[i]
        ax.errorbar(
            row[value_col], position, xerr=row[error_col],
            fmt="o", linestyle="None", marker="s", markersize=5,
            ecolor=color, color=color, capsize=5,
        )

    ax.set_yticklabels(data[method_col], color="black")
    ax.set_yticks(list(y_positions))
    ax.set_xlabel("ROC AUC")
    ax.set_title(title, fontsize=18)

    # Color-coded table
    cmap = LinearSegmentedColormap.from_list("blue_orange", ["white", "darkorange"])
    norm = mcolors.Normalize(vmin=cmap_range[0], vmax=cmap_range[1])

    def color_map(val):
        try:
            return mcolors.rgb2hex(cmap(norm(float(val))))
        except (ValueError, TypeError):
            return "none"

    table_cols = [method_col, value_col] + extra_cols
    table_data = [
        [row[c] for c in table_cols]
        for _, row in data.iloc[::-1].iterrows()
    ]
    cell_colors = [
        ["none"] + [color_map(row[c]) for c in [value_col] + extra_cols]
        for _, row in data.iloc[::-1].iterrows()
    ]

    col_labels = table_cols
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="right",
        bbox=[1.05, 0, 1.0, 1.0],
        colWidths=[3.5] + [2] * (len(table_cols) - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0.45, 0.85)
    plt.subplots_adjust(right=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    data: pd.DataFrame,
    metric_col: str,
    method_col: str = "Method",
    group_col: str = "Group",
    title: str = "",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
    xlim: tuple[float, float] = (0.05, 0.85),
) -> plt.Figure:
    """
    Plot horizontal strip plot comparing a metric across methods.

    Args:
        data: DataFrame with method, group, and metric columns.
        metric_col: Column name for the metric to plot.
        method_col: Column name for method labels.
        group_col: Column name for grouping (determines color).
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size. If None, auto-scales height with row count so
            y-tick labels don't overlap (common when comparing many confidence
            signals x demographic groups).
        xlim: X-axis limits.

    Returns:
        Matplotlib figure
    """
    plot_data = data.dropna(subset=[metric_col])
    plot_data = plot_data.iloc[::-1].reset_index(drop=True)

    unique_groups = plot_data[group_col].unique()
    palette = sns.color_palette("husl", len(unique_groups))

    # Auto-scale height: give each row ~0.35 inches so labels fit.
    n_rows = max(len(plot_data), 1)
    if figsize is None:
        figsize = (7, max(2.5, 0.35 * n_rows + 1.0))

    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(
        y=method_col, x=metric_col, hue=group_col,
        data=plot_data, orient="h", palette=palette, size=8, ax=ax,
    )

    if ax.get_legend():
        ax.get_legend().remove()

    ax.set_xlim(*xlim)

    for i, (method, value) in enumerate(
        zip(plot_data[method_col], plot_data[metric_col])
    ):
        text_x = max(value, xlim[0]) + 0.02
        ax.text(text_x, i, f"{value:.2f}", color="black", va="center", fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_temperature_scaling_comparison(
    confidences_by_group: dict[str, np.ndarray],
    correctness_by_group: dict[str, np.ndarray],
    scaler,  # DemographicTemperatureScaler
    confidence_type: str = "logprob",
    n_bins: int = 10,
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot before/after temperature scaling reliability diagrams.

    Shows side-by-side reliability diagrams for each demographic group,
    with pre- and post-temperature scaling calibration curves overlaid.

    Args:
        confidences_by_group: Dict mapping group to confidence arrays.
        correctness_by_group: Dict mapping group to correctness arrays.
        scaler: Fitted DemographicTemperatureScaler with .temperatures dict.
        confidence_type: Confidence type name (e.g. "logprob", "verbalized").
        n_bins: Number of bins for reliability diagram.
        save_path: Optional path to save figure.
        figsize: Figure size (auto-scales with group count if None).

    Returns:
        Matplotlib figure
    """
    from ..calibration.temperature_scaling import apply_fitted_temperature

    group_names = sorted(confidences_by_group.keys())
    n_groups = len(group_names)

    if figsize is None:
        figsize = (max(14, 6 * n_groups), 6)

    if n_groups == 0:
        fig, _ = plt.subplots(1, 1, figsize=figsize)
        plt.close(fig)
        return fig

    fig, axes = plt.subplots(1, n_groups, figsize=figsize)
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, group_names):
        confs = np.asarray(confidences_by_group[group], dtype=float)
        corr = np.asarray(correctness_by_group[group], dtype=float)

        if len(confs) == 0:
            continue

        effective_bins = min(n_bins, len(confs))
        if effective_bins < 2:
            continue

        bin_edges = np.linspace(0, 1, effective_bins + 1)

        def _bin_stats(c, y, edges):
            indices = np.digitize(c, edges[1:-1])
            centers, accs = [], []
            for b in range(len(edges) - 1):
                mask = indices == b
                if mask.sum() > 0:
                    centers.append(c[mask].mean())
                    accs.append(y[mask].mean())
            return np.array(centers), np.array(accs)

        orig_centers, orig_accs = _bin_stats(confs, corr, bin_edges)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect")

        # Original (before scaling)
        ax.plot(orig_centers, orig_accs, "o-", color="steelblue", linewidth=2,
                markersize=6, label="Before")

        # After temperature scaling
        if scaler and group in scaler.temperatures:
            temp = scaler.temperatures[group]
            scaled_confs = apply_fitted_temperature(confs, temp)
            sc_centers, sc_accs = _bin_stats(scaled_confs, corr, bin_edges)
            ax.plot(sc_centers, sc_accs, "s-", color="darkorange", linewidth=2,
                    markersize=6, label=f"After (T={temp:.2f})")

        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{group.capitalize()} Skin (n={len(confs)})", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Temperature Scaling: {confidence_type.replace('_', ' ').title()} Confidence",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curve",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    colors: dict[str, str] | None = None,
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Plot ROC curves for multiple confidence methods.

    Args:
        curves: Dict mapping label to (correctness, confidences) arrays.
            correctness is binary 0/1, confidences are scores in [0, 1].
        title: Plot title.
        save_path: Optional path to save figure.
        figsize: Figure size.
        colors: Optional dict mapping label to color string.
        font_family: Font family.

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import auc, roc_curve

    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))

    for idx, (label, (correctness, confidences)) in enumerate(curves.items()):
        correctness = np.asarray(correctness, dtype=int)
        confidences = np.asarray(confidences, dtype=float)

        # Filter valid
        valid = np.isfinite(confidences)
        correctness = correctness[valid]
        confidences = confidences[valid]

        if len(correctness) < 2 or len(np.unique(correctness)) < 2:
            continue

        fpr, tpr, _ = roc_curve(correctness, confidences)
        roc_auc = auc(fpr, tpr)

        color = (colors or {}).get(label, default_colors[idx])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_by_demographic(
    curves_by_group: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    title: str = "ROC Curves by Demographic",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot side-by-side ROC curves for each demographic group.

    Args:
        curves_by_group: Nested dict of {group: {conf_type: (correctness, confidences)}}.
        title: Overall title.
        save_path: Optional path to save figure.
        figsize: Figure size (auto-scales with group count if None).

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import auc, roc_curve

    n_groups = len(curves_by_group)
    if figsize is None:
        figsize = (max(14, 6 * n_groups), 6)
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)
    if n_groups == 1:
        axes = [axes]

    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (group_name, curves) in zip(axes, curves_by_group.items()):
        for idx, (label, (correctness, confidences)) in enumerate(curves.items()):
            correctness = np.asarray(correctness, dtype=int)
            confidences = np.asarray(confidences, dtype=float)

            valid = np.isfinite(confidences)
            correctness = correctness[valid]
            confidences = confidences[valid]

            if len(correctness) < 2 or len(np.unique(correctness)) < 2:
                continue

            fpr, tpr, _ = roc_curve(correctness, confidences)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=default_colors[idx], lw=2,
                    label=f"{label} ({roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{group_name.capitalize()} Skin")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_violin(
    analyzer_df: pd.DataFrame,
    confidence_columns: list[str],
    save_path: Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Violin/box plot comparing confidence distributions across signal types,
    split by prediction correctness.

    Args:
        analyzer_df: DataFrame with confidence columns and 'is_correct'.
        confidence_columns: Column names to plot.
        save_path: Optional path to save figure.
        figsize: Figure size.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    plt.rcParams["font.family"] = font_family

    # Build long-format data
    rows = []
    for col in confidence_columns:
        if col not in analyzer_df.columns:
            continue
        valid = analyzer_df[col].notna()
        for _, r in analyzer_df.loc[valid, [col, "is_correct"]].iterrows():
            label = col.replace("_confidence", "").replace("_", " ")
            rows.append({
                "Confidence Type": label,
                "Confidence": float(r[col]),
                "Outcome": "Correct" if r["is_correct"] else "Incorrect",
            })

    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    palette = {"Correct": "#4CAF50", "Incorrect": "#F44336"}

    # Check for degenerate signals
    degenerate_types = set()
    for col in confidence_columns:
        if col not in analyzer_df.columns:
            continue
        vals = analyzer_df[col].dropna()
        if len(vals) > 0 and vals.std() < 0.001:
            label = col.replace("_confidence", "").replace("_", " ")
            degenerate_types.add(label)

    sns.violinplot(
        data=long_df,
        x="Confidence Type",
        y="Confidence",
        hue="Outcome",
        split=True,
        inner="box",
        palette=palette,
        ax=ax,
        cut=0,
    )

    # Annotate degenerate signals
    type_labels = long_df["Confidence Type"].unique()
    for i, t in enumerate(type_labels):
        if t in degenerate_types:
            ax.annotate(
                "DEGENERATE\n(no variance)",
                xy=(i, 0.5),
                ha="center",
                fontsize=9,
                color="gray",
                fontstyle="italic",
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Confidence Distributions by Signal Type", fontsize=14)
    ax.legend(title="Prediction", loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_overconfidence_bar(
    analyzer_df: pd.DataFrame,
    confidence_columns: list[str],
    bin_edges: list[float] | None = None,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Grouped bar chart showing error rate within confidence bins for each
    confidence type.

    Args:
        analyzer_df: DataFrame with confidence columns and 'is_correct'.
        confidence_columns: Confidence column names.
        bin_edges: Custom bin edges. Defaults to [0.5, 0.7, 0.8, 0.9, 0.95, 1.01].
        save_path: Optional save path.
        figsize: Figure size.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    plt.rcParams["font.family"] = font_family

    if bin_edges is None:
        bin_edges = [0.5, 0.7, 0.8, 0.9, 0.95, 1.01]

    valid_cols = [c for c in confidence_columns if c in analyzer_df.columns]
    if not valid_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_cols)))
    n_bins = len(bin_edges) - 1
    bar_width = 0.8 / len(valid_cols)

    fig, ax = plt.subplots(figsize=figsize)
    bin_labels = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        hi_str = f"{min(hi, 1.0):.2f}"
        bin_labels.append(f"{lo:.2f}-{hi_str}")

    x = np.arange(n_bins)

    for col_idx, col in enumerate(valid_cols):
        valid = analyzer_df[col].notna()
        confs = analyzer_df.loc[valid, col].values.astype(float)
        correct = analyzer_df.loc[valid, "is_correct"].values.astype(int)

        error_rates = []
        counts = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= lo) & (confs < hi)
            n = mask.sum()
            counts.append(n)
            if n > 0:
                error_rates.append(1.0 - correct[mask].mean())
            else:
                error_rates.append(0.0)

        offset = (col_idx - len(valid_cols) / 2 + 0.5) * bar_width
        label = col.replace("_confidence", "").replace("_", " ")
        bars = ax.bar(
            x + offset,
            error_rates,
            width=bar_width * 0.9,
            color=colors[col_idx],
            label=label,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Annotate with sample counts
        for bar, n in zip(bars, counts):
            if n > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Confidence Range", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("Error Rate by Confidence Bin", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_scatter(
    analyzer_df: pd.DataFrame,
    x_column: str = "verbalized_confidence",
    y_column: str = "logprob_confidence",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (8, 8),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Scatter plot of one confidence type vs another, colored by correctness,
    with marginal histograms.

    Args:
        analyzer_df: DataFrame with both confidence columns and 'is_correct'.
        x_column: Column for x-axis.
        y_column: Column for y-axis.
        save_path: Optional save path.
        figsize: Figure size.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    plt.rcParams["font.family"] = font_family

    valid = analyzer_df[x_column].notna() & analyzer_df[y_column].notna()
    df = analyzer_df.loc[valid].copy()

    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    x_vals = df[x_column].values.astype(float)
    y_vals = df[y_column].values.astype(float)
    correct = df["is_correct"].values.astype(bool)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    ax_main.scatter(x_vals[correct], y_vals[correct], c="#4CAF50", alpha=0.6,
                    s=40, label="Correct", edgecolors="none")
    ax_main.scatter(x_vals[~correct], y_vals[~correct], c="#F44336", alpha=0.7,
                    s=40, label="Incorrect", marker="x")

    # Diagonal reference
    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

    # Spearman correlation (undefined if either input is constant)
    if np.std(x_vals) < 1e-12 or np.std(y_vals) < 1e-12:
        rho, p = float("nan"), float("nan")
    else:
        rho, p = spearmanr(x_vals, y_vals)
    ax_main.text(0.05, 0.05, f"Spearman r={rho:.3f}\np={p:.3g}",
                 transform=ax_main.transAxes, fontsize=10,
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    x_label = x_column.replace("_confidence", "").replace("_", " ").title()
    y_label = y_column.replace("_confidence", "").replace("_", " ").title()
    ax_main.set_xlabel(f"{x_label} Confidence", fontsize=12)
    ax_main.set_ylabel(f"{y_label} Confidence", fontsize=12)
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.set_xlim(-0.02, 1.02)

    # Zoom y-axis if near-constant
    y_std = np.std(y_vals)
    if y_std < 0.01:
        y_min = max(0, np.min(y_vals) - 0.005)
        y_max = min(1.001, np.max(y_vals) + 0.005)
        ax_main.set_ylim(y_min, y_max)
        ax_main.text(0.95, 0.05, f"Y-axis zoomed\n(std={y_std:.6f})",
                     transform=ax_main.transAxes, fontsize=8, ha="right",
                     color="gray", fontstyle="italic")
    else:
        ax_main.set_ylim(-0.02, 1.02)

    # Marginal histograms
    bins = 20
    ax_top.hist(x_vals[correct], bins=bins, alpha=0.5, color="#4CAF50", density=True)
    ax_top.hist(x_vals[~correct], bins=bins, alpha=0.5, color="#F44336", density=True)
    ax_top.set_ylabel("Density", fontsize=9)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(y_vals[correct], bins=bins, alpha=0.5, color="#4CAF50",
                  density=True, orientation="horizontal")
    ax_right.hist(y_vals[~correct], bins=bins, alpha=0.5, color="#F44336",
                  density=True, orientation="horizontal")
    ax_right.set_xlabel("Density", fontsize=9)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle(f"{x_label} vs {y_label} Confidence", fontsize=14, y=0.98)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_demographic_accuracy_bar(
    analyzer_df: pd.DataFrame,
    confidence_column: str = "verbalized_confidence",
    bootstrap_n: int = 1000,
    ci_level: float = 0.95,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 5),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Two-panel bar chart: accuracy by demographic group (left) and ECE (right),
    with bootstrap error bars.

    Args:
        analyzer_df: DataFrame with 'demographic_group', 'is_correct', and
            the confidence column.
        confidence_column: Which confidence to use for ECE.
        bootstrap_n: Number of bootstrap samples.
        ci_level: Confidence interval level.
        save_path: Optional save path.
        figsize: Figure size.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    from ..calibration.bootstrap import bootstrap_all_metrics

    plt.rcParams["font.family"] = font_family

    groups = sorted([g for g in analyzer_df["demographic_group"].unique() if g != "unknown"])
    group_colors = DEMOGRAPHIC_PALETTE
    default_colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    accuracies = []
    acc_cis = []
    eces = []
    ece_cis = []
    group_sizes = []

    for group in groups:
        g_df = analyzer_df[analyzer_df["demographic_group"] == group]
        valid = g_df[confidence_column].notna()
        confs = g_df.loc[valid, confidence_column].values.astype(float)
        correct = g_df.loc[valid, "is_correct"].values.astype(int)
        group_sizes.append(len(correct))

        labels = g_df.loc[valid, "ground_truth"].values if "ground_truth" in g_df.columns else None
        bs = bootstrap_all_metrics(confs, correct, n_bootstrap=bootstrap_n,
                                   ci_level=ci_level, random_seed=42, labels=labels)
        acc_r = bs["accuracy"]
        ece_r = bs["ece"]

        accuracies.append(acc_r.point_estimate)
        acc_cis.append((acc_r.point_estimate - acc_r.ci_lower,
                        acc_r.ci_upper - acc_r.point_estimate))
        eces.append(ece_r.point_estimate)
        ece_cis.append((ece_r.point_estimate - ece_r.ci_lower,
                        ece_r.ci_upper - ece_r.point_estimate))

    x = np.arange(len(groups))
    colors = [group_colors.get(g, default_colors[i]) for i, g in enumerate(groups)]

    # Left panel: accuracy
    ax1.bar(x, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.errorbar(x, accuracies, yerr=np.array(acc_cis).T, fmt="none",
                 ecolor="black", capsize=5, linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g.capitalize()}\n(n={n})" for g, n in zip(groups, group_sizes)])
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy by Demographic Group", fontsize=13)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Fisher's exact test annotation if exactly 2 groups
    if len(groups) == 2:
        g1 = analyzer_df[analyzer_df["demographic_group"] == groups[0]]
        g2 = analyzer_df[analyzer_df["demographic_group"] == groups[1]]
        table = [
            [int(g1["is_correct"].sum()), int((~g1["is_correct"]).sum())],
            [int(g2["is_correct"].sum()), int((~g2["is_correct"]).sum())],
        ]
        _, p_val = fisher_exact(table)
        ax1.text(0.5, 0.95, f"Fisher's exact p={p_val:.3f}",
                 transform=ax1.transAxes, ha="center", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Right panel: ECE
    conf_label = confidence_column.replace("_confidence", "").replace("_", " ")
    ax2.bar(x, eces, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.errorbar(x, eces, yerr=np.array(ece_cis).T, fmt="none",
                 ecolor="black", capsize=5, linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{g.capitalize()}\n(n={n})" for g, n in zip(groups, group_sizes)])
    ax2.set_ylabel("ECE", fontsize=12)
    ax2.set_title(f"ECE by Group ({conf_label})", fontsize=13)
    ax2.set_ylim(0, max(0.1, max(eces) * 1.5))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_violin_by_group(
    analyzer_df: pd.DataFrame,
    confidence_columns: list[str],
    group_column: str = "demographic_group",
    save_path: Path | None = None,
    figsize_per_group: tuple[int, int] = (10, 6),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Violin plots of confidence distributions split by correctness, one subplot
    per demographic group.

    Args:
        analyzer_df: DataFrame with confidence columns, 'is_correct', and group column.
        confidence_columns: Confidence column names to plot.
        group_column: Column for demographic grouping.
        save_path: Optional path to save figure.
        figsize_per_group: Size of each group's subplot.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    plt.rcParams["font.family"] = font_family

    groups = sorted(
        [g for g in analyzer_df[group_column].unique() if g != "unknown"]
    )
    if not groups:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    n_groups = len(groups)
    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(figsize_per_group[0] * n_groups, figsize_per_group[1]),
        sharey=True,
    )
    if n_groups == 1:
        axes = [axes]

    palette = {"Correct": "#4CAF50", "Incorrect": "#F44336"}

    # Detect degenerate signals (globally)
    degenerate_types = set()
    for col in confidence_columns:
        if col not in analyzer_df.columns:
            continue
        vals = analyzer_df[col].dropna()
        if len(vals) > 0 and vals.std() < 0.001:
            degenerate_types.add(col.replace("_confidence", "").replace("_", " "))

    for ax, group in zip(axes, groups):
        g_df = analyzer_df[analyzer_df[group_column] == group]

        rows = []
        for col in confidence_columns:
            if col not in g_df.columns:
                continue
            valid = g_df[col].notna()
            for _, r in g_df.loc[valid, [col, "is_correct"]].iterrows():
                label = col.replace("_confidence", "").replace("_", " ")
                rows.append({
                    "Confidence Type": label,
                    "Confidence": float(r[col]),
                    "Outcome": "Correct" if r["is_correct"] else "Incorrect",
                })

        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(f"{group.capitalize()} Skin")
            continue

        long_df = pd.DataFrame(rows)
        sns.violinplot(
            data=long_df,
            x="Confidence Type",
            y="Confidence",
            hue="Outcome",
            split=True,
            inner="box",
            palette=palette,
            ax=ax,
            cut=0,
        )

        # Annotate degenerate signals
        type_labels = long_df["Confidence Type"].unique()
        for i, t in enumerate(type_labels):
            if t in degenerate_types:
                ax.annotate(
                    "DEGENERATE\n(no variance)",
                    xy=(i, 0.5), ha="center", fontsize=8,
                    color="gray", fontstyle="italic",
                )

        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{group.capitalize()} Skin (n={len(g_df)})", fontsize=13)
        ax.set_xlabel("")
        if ax != axes[0]:
            ax.set_ylabel("")
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            ax.set_ylabel("Confidence", fontsize=12)
            ax.legend(title="Prediction", loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Confidence Distributions by Signal Type and Demographic Group",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_equity_comparison(
    equity_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    font_family: str = "DejaVu Sans",
) -> plt.Figure:
    """
    Two-panel equity comparison: per-group ECE by confidence signal (left)
    and max ECE gap summary (right).

    Args:
        equity_df: DataFrame from DemographicAnalyzer.compute_equity_summary().
            Must have columns: confidence_type, max_ece_gap, and ece_<group> columns.
        save_path: Optional path to save figure.
        figsize: Figure size.
        font_family: Font family.

    Returns:
        Matplotlib figure.
    """
    plt.rcParams["font.family"] = font_family

    if equity_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    # Identify per-group ECE columns
    ece_cols = [c for c in equity_df.columns if c.startswith("ece_")]
    group_names = [c.removeprefix("ece_") for c in ece_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left panel: grouped bar chart of ECE per group per signal ---
    n_signals = len(equity_df)
    n_groups_plot = len(group_names)
    bar_width = 0.8 / n_groups_plot
    x = np.arange(n_signals)

    signal_labels = [
        s.replace("_confidence", "").replace("_", " ")
        for s in equity_df["confidence_type"]
    ]

    for gi, (group, col) in enumerate(zip(group_names, ece_cols)):
        offset = (gi - n_groups_plot / 2 + 0.5) * bar_width
        color = get_group_color(group, gi)
        vals = equity_df[col].values
        ax1.bar(
            x + offset, vals, width=bar_width * 0.9,
            color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
            label=f"{group.capitalize()} skin",
        )

    # Gap annotations
    for i in range(n_signals):
        group_vals = [equity_df.iloc[i][c] for c in ece_cols]
        min_v, max_v = min(group_vals), max(group_vals)
        gap = max_v - min_v
        mid_y = (min_v + max_v) / 2
        ax1.annotate(
            "", xy=(i + 0.42, max_v), xytext=(i + 0.42, min_v),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        ax1.text(
            i + 0.48, mid_y, f"{gap:.3f}",
            color="red", fontsize=9, va="center", fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(signal_labels, rotation=25, ha="right")
    ax1.set_ylabel("ECE", fontsize=12)
    ax1.set_title("ECE by Confidence Signal and Demographic Group", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right panel: horizontal bar of max ECE gap per signal ---
    gaps = equity_df["max_ece_gap"].values
    best_idx = np.argmin(gaps)
    bar_colors = ["#4CAF50" if i == best_idx else "#9E9E9E" for i in range(n_signals)]

    ax2.barh(x, gaps, color=bar_colors, edgecolor="black", linewidth=0.5)
    for i, g in enumerate(gaps):
        ax2.text(g + 0.002, i, f"{g:.4f}", va="center", fontsize=10)

    ax2.set_yticks(x)
    ax2.set_yticklabels(signal_labels)
    ax2.set_xlabel("Max ECE Gap Across Groups", fontsize=12)
    ax2.set_title("Calibration Equity (lower = more equitable)", fontsize=13)
    ax2.axvline(x=0, color="gray", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # Highlight best
    ax2.annotate(
        "Most equitable",
        xy=(gaps[best_idx], best_idx),
        xytext=(gaps[best_idx] + max(gaps) * 0.3, best_idx + 0.3),
        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.5),
        fontsize=10, color="#4CAF50", fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Constrained vs freeform logprob comparison plots
# ---------------------------------------------------------------------------


def plot_logprob_method_comparison(
    df: pd.DataFrame,
    constrained_col: str = "logprob_confidence",
    freeform_col: str = "logprob_confidence_freeform",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compare constrained A/B logprob vs freeform logprob extraction.

    Left panel: overlaid histograms of confidence distributions.
    Right panel: reliability curves for both methods with ECE annotations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    has_constrained = constrained_col in df.columns and df[constrained_col].notna().any()
    has_freeform = freeform_col in df.columns and df[freeform_col].notna().any()

    # --- Left panel: distribution histograms ---
    bins = np.linspace(0.4, 1.0, 25)
    if has_freeform:
        freeform_vals = df[freeform_col].dropna().values
        ax1.hist(
            freeform_vals, bins=bins, alpha=0.5, density=True,
            label="Freeform (old)", color="#E53935", edgecolor="white",
        )
    if has_constrained:
        constrained_vals = df[constrained_col].dropna().values
        ax1.hist(
            constrained_vals, bins=bins, alpha=0.5, density=True,
            label="Constrained A/B (new)", color="#1E88E5", edgecolor="white",
        )
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Logprob Confidence Distribution", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: reliability curves ---
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    n_bins = 10

    for col, label, color in [
        (freeform_col, "Freeform (old)", "#E53935"),
        (constrained_col, "Constrained A/B (new)", "#1E88E5"),
    ]:
        if col not in df.columns or not df[col].notna().any():
            continue
        sub = df[[col, "is_correct"]].dropna()
        if len(sub) < n_bins:
            continue
        try:
            sub = sub.copy()
            sub["bin"] = pd.qcut(sub[col], n_bins, duplicates="drop")
            grouped = sub.groupby("bin", observed=True)
            mean_conf = grouped[col].mean()
            mean_acc = grouped["is_correct"].mean()
            ece, _, _, _ = compute_ece(sub[col].values, sub["is_correct"].values.astype(float))
            ax2.plot(
                mean_conf, mean_acc, "o-", color=color, markersize=5,
                label=f"{label} (ECE={ece:.3f})",
            )
        except ValueError:
            pass

    ax2.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax2.set_ylabel("Observed Accuracy", fontsize=12)
    ax2.set_title("Calibration: Freeform vs Constrained Logprob", fontsize=13)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_compliance_summary(
    df: pd.DataFrame,
    compliant_col: str = "logprob_compliant",
    group_col: str = "demographic_group",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart showing A/B token compliance rate by demographic group."""
    if compliant_col not in df.columns or not df[compliant_col].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "No compliance data available",
            ha="center", va="center", fontsize=14,
        )
        ax.set_axis_off()
        return fig

    sub = df[[compliant_col, group_col]].dropna()
    groups = sorted(sub[group_col].unique())

    compliance_rates = []
    labels = []
    colors = []
    for g in groups:
        mask = sub[group_col] == g
        rate = sub.loc[mask, compliant_col].mean()
        compliance_rates.append(rate)
        labels.append(f"{g}\n(n={mask.sum()})")
        colors.append(get_group_color(g))

    # Overall
    overall_rate = sub[compliant_col].mean()
    compliance_rates.append(overall_rate)
    labels.append(f"Overall\n(n={len(sub)})")
    colors.append("#616161")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, compliance_rates, color=colors, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, compliance_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Compliance Rate", fontsize=12)
    ax.set_title("A/B Token Compliance Rate", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_method_grid(
    df: pd.DataFrame,
    conf_cols: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grid of scatter plots comparing confidence signals with correlations.

    Diagonal shows per-method histograms; off-diagonal shows pairwise scatter
    with Spearman correlation, color-coded by prediction correctness.
    """
    if conf_cols is None:
        candidates = [
            "logprob_confidence",
            "logprob_confidence_freeform",
            "verbalized_confidence",
        ]
        # Add self-consistency columns
        for col in df.columns:
            if col.startswith("sc_confidence_t"):
                candidates.append(col)
        conf_cols = [c for c in candidates if c in df.columns and df[c].notna().any()]

    if len(conf_cols) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "Not enough confidence signals for grid",
            ha="center", va="center", fontsize=14,
        )
        ax.set_axis_off()
        return fig

    n = len(conf_cols)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    short_names = {
        "logprob_confidence": "LP (constrained)",
        "logprob_confidence_freeform": "LP (freeform)",
        "verbalized_confidence": "Verbalized",
        "min_logprob_confidence": "Min LP",
    }

    for i, col_y in enumerate(conf_cols):
        for j, col_x in enumerate(conf_cols):
            ax = axes[i, j] if n > 1 else axes
            name_x = short_names.get(col_x, col_x.replace("_", " ").title())
            name_y = short_names.get(col_y, col_y.replace("_", " ").title())

            if i == j:
                # Diagonal: histogram
                vals = df[col_x].dropna().values
                ax.hist(vals, bins=20, alpha=0.7, color="#1E88E5", edgecolor="white")
                ax.set_title(name_x, fontsize=10)
            else:
                # Off-diagonal: scatter
                sub = df[[col_x, col_y, "is_correct"]].dropna()
                if len(sub) > 0:
                    correct = sub[sub["is_correct"]]
                    incorrect = sub[~sub["is_correct"]]
                    ax.scatter(
                        incorrect[col_x], incorrect[col_y],
                        alpha=0.4, s=15, c="#E53935", label="Incorrect",
                    )
                    ax.scatter(
                        correct[col_x], correct[col_y],
                        alpha=0.4, s=15, c="#43A047", label="Correct",
                    )
                    # Spearman correlation
                    if len(sub) >= 5:
                        rho, _pval = spearmanr(sub[col_x], sub[col_y])
                        ax.text(
                            0.05, 0.95, f"\u03c1={rho:.2f}",
                            transform=ax.transAxes, fontsize=9,
                            va="top", fontweight="bold",
                        )

            if j == 0:
                ax.set_ylabel(name_y, fontsize=9)
            if i == n - 1:
                ax.set_xlabel(name_x, fontsize=9)

    plt.suptitle("Confidence Signal Correlation Grid", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
