"""Analysis and visualization modules."""

from .reliability_diagram import (
    plot_reliability_diagram,
    plot_demographic_comparison,
    plot_confidence_violin,
    plot_confidence_violin_by_group,
    plot_equity_comparison,
    plot_overconfidence_bar,
    plot_confidence_scatter,
    plot_demographic_accuracy_bar,
)
from .demographic_analysis import DemographicAnalyzer
from .report import ReportGenerator

__all__ = [
    "plot_reliability_diagram",
    "plot_demographic_comparison",
    "plot_confidence_violin",
    "plot_confidence_violin_by_group",
    "plot_equity_comparison",
    "plot_overconfidence_bar",
    "plot_confidence_scatter",
    "plot_demographic_accuracy_bar",
    "DemographicAnalyzer",
    "ReportGenerator",
]
