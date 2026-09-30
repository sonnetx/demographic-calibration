"""Command-line interface for demographic calibration study."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .analysis.demographic_analysis import DemographicAnalyzer
from .analysis.reliability_diagram import (
    plot_calibration_lines,
    plot_compliance_summary,
    plot_confidence_method_grid,
    plot_confidence_scatter,
    plot_confidence_violin,
    plot_confidence_violin_by_group,
    plot_demographic_accuracy_bar,
    plot_demographic_comparison,
    plot_equity_comparison,
    plot_forest_with_table,
    plot_logprob_method_comparison,
    plot_metric_comparison,
    plot_overconfidence_bar,
    plot_roc_by_demographic,
    plot_roc_curves,
    plot_temperature_scaling_comparison,
)
from .analysis.report import ReportGenerator
from .config import ExperimentConfig
from .data.dataset import DDIDataset
from .models.base import BaseVLM
from .models.registry import get_model, list_models
from .utils.io import (
    append_checkpoint,
    checkpoint_to_results,
    load_checkpoint,
    load_results,
    save_results,
)

app = typer.Typer(
    name="democalib",
    help="Demographic Calibration Study of Medical Vision-Language Models",
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to YAML or JSON configuration file"),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Validate configuration without running experiment"
    ),
    subset: int = typer.Option(
        None, "--subset", "-s", help="Run on a subset of samples for testing"
    ),
    metadata_path: Path = typer.Option(
        None, "--metadata", "-m", help="Override: path to ddi_metadata.csv"
    ),
    images_dir: Path = typer.Option(
        None, "--images", "-i", help="Override: path to images directory"
    ),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Override: output directory"
    ),
):
    """
    Run the full calibration study pipeline.

    Examples:
        democalib run configs/default.yaml
        democalib run configs/default.yaml --metadata /data/ddi/ddi_metadata.csv --images /data/ddi/images
        democalib run configs/default.yaml --subset 10
    """
    # Load and validate configuration
    console.print(f"[bold]Loading configuration from {config}...[/bold]")

    try:
        if config.suffix in [".yaml", ".yml"]:
            exp_config = ExperimentConfig.from_yaml(config)
        else:
            exp_config = ExperimentConfig.from_json(config)
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)

    # Apply CLI overrides
    if metadata_path:
        exp_config.data.metadata_path = metadata_path
        console.print(f"[yellow]Override metadata_path: {metadata_path}[/yellow]")
    if images_dir:
        exp_config.data.images_dir = images_dir
        console.print(f"[yellow]Override images_dir: {images_dir}[/yellow]")
    if output_dir:
        exp_config.output.output_dir = output_dir
        console.print(f"[yellow]Override output_dir: {output_dir}[/yellow]")

    console.print(f"[green]Configuration validated: {exp_config.experiment_name}[/green]")

    if dry_run:
        console.print("[yellow]Dry run mode - exiting without execution[/yellow]")
        _print_config_summary(exp_config)
        return

    # Run the experiment
    asyncio.run(_run_experiment(exp_config, subset))


@app.command()
def analyze(
    results_path: Path = typer.Argument(..., help="Path to saved results JSON file"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory for figures and reports"
    ),
    confidence_type: str = typer.Option(
        "all",
        "--confidence",
        "-c",
        help="Confidence type to analyze: logprob, verbalized, self_consistency, or all",
    ),
):
    """
    Analyze saved results and generate visualizations.

    Example:
        democalib analyze outputs/results/gpt4v_results.json -o outputs/figures
    """
    console.print(f"[bold]Loading results from {results_path}...[/bold]")

    results = load_results(results_path)
    analyzer = DemographicAnalyzer(results)

    output_dir = output_dir or results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary table
    summary = analyzer.generate_summary_table()
    console.print("\n[bold]Calibration Summary:[/bold]")
    _print_dataframe_table(summary, "Calibration Metrics")

    # Generate reliability diagrams
    sc_cols = [c for c in summary.columns if "sc_confidence_t" in str(c)]
    if not sc_cols:
        # Detect from analyzer DataFrame
        sc_cols = [c for c in analyzer.df.columns if c.startswith("sc_confidence_t")]
    # Also detect the single self_consistency_confidence column
    if not sc_cols and "self_consistency_confidence" in analyzer.df.columns:
        sc_cols = ["self_consistency_confidence"]
    if confidence_type == "all":
        conf_types = ["logprob_confidence", "verbalized_confidence"] + sc_cols
        # Include freeform baseline if present
        if "logprob_confidence_freeform" in analyzer.df.columns:
            conf_types.append("logprob_confidence_freeform")
    elif confidence_type == "self_consistency":
        conf_types = sc_cols
    else:
        conf_types = [f"{confidence_type}_confidence"]

    for conf_type in conf_types:
        try:
            metrics_by_group = analyzer.compute_metrics_by_group(conf_type)
            if metrics_by_group:
                plot_demographic_comparison(
                    metrics_by_group,
                    confidence_type=conf_type.replace("_confidence", ""),
                    save_path=output_dir / f"reliability_{conf_type}.png",
                )
                console.print(f"[green]Saved reliability diagram: {conf_type}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate {conf_type} diagram: {e}[/yellow]")

    # Generate calibration line plots per confidence type
    for conf_type in conf_types:
        try:
            arrays_by_group = analyzer.get_confidence_arrays(conf_type)
            if arrays_by_group:
                plot_calibration_lines(
                    series={
                        group: (confs, corr)
                        for group, (confs, corr) in arrays_by_group.items()
                    },
                    title=f"Calibration: {conf_type.replace('_confidence', '')}",
                    save_path=output_dir / f"calibration_lines_{conf_type}.png",
                )
                console.print(f"[green]Saved calibration line plot: {conf_type}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate calibration line plot for {conf_type}: {e}[/yellow]")

    # Generate forest plot (AUROC by confidence type × demographic group)
    import pandas as pd

    bootstrap_rows = []
    for conf_type in conf_types:
        try:
            bootstrap_by_group = analyzer.compute_bootstrap_by_group(conf_type, n_bootstrap=1000)
            for group, metrics_dict in bootstrap_by_group.items():
                auroc_result = metrics_dict.get("auroc")
                if auroc_result and not np.isnan(auroc_result.point_estimate):
                    bootstrap_rows.append({
                        "Method": f"{conf_type.replace('_confidence', '')} ({group})",
                        "AUC Total": round(auroc_result.point_estimate, 3),
                        "CI Error": round(auroc_result.ci_width / 2, 3),
                        "Group": conf_type.replace("_confidence", ""),
                    })
        except Exception:
            pass

    if bootstrap_rows:
        forest_df = pd.DataFrame(bootstrap_rows)
        try:
            # Auto-assign palette based on confidence types
            unique_groups = forest_df["Group"].unique()
            color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
            auto_palette = {g: color_cycle[i % len(color_cycle)] for i, g in enumerate(unique_groups)}

            plot_forest_with_table(
                forest_df,
                value_col="AUC Total",
                error_col="CI Error",
                group_col="Group",
                palette=auto_palette,
                title="AUROC by Confidence Type & Demographic",
                save_path=output_dir / "forest_auroc.png",
                cmap_range=(
                    max(0.4, forest_df["AUC Total"].min() - 0.05),
                    min(1.0, forest_df["AUC Total"].max() + 0.05),
                ),
            )
            console.print("[green]Saved forest plot: forest_auroc.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate forest plot: {e}[/yellow]")

    # Generate strip plots for ECE and Brier score
    if not summary.empty and len(summary) > 1:
        # Build a Method column combining confidence type and group
        summary["Method"] = summary["confidence_type"] + " (" + summary["demographic_group"] + ")"
        summary["Group"] = summary["confidence_type"]

        for metric, metric_label in [("ece", "ECE"), ("brier", "Brier Score")]:
            try:
                plot_metric_comparison(
                    summary,
                    metric_col=metric,
                    method_col="Method",
                    group_col="Group",
                    title=metric_label,
                    save_path=output_dir / f"strip_{metric}.png",
                    xlim=(0, max(summary[metric].max() * 1.2, 0.1)),
                )
                console.print(f"[green]Saved strip plot: strip_{metric}.png[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not generate {metric} strip plot: {e}[/yellow]")

    # Generate ROC curves (all confidence types on one plot)
    roc_curves_all = {}
    for conf_type in conf_types:
        try:
            arrays_by_group = analyzer.get_confidence_arrays(conf_type)
            # Combine all groups for the overall ROC
            all_corr = []
            all_conf = []
            for group, (confs, corr) in arrays_by_group.items():
                all_conf.extend(confs)
                all_corr.extend(corr)
            if len(all_corr) > 1:
                label = conf_type.replace("_confidence", "").replace("_", " ")
                roc_curves_all[label] = (np.array(all_corr), np.array(all_conf))
        except Exception:
            pass

    if roc_curves_all:
        try:
            plot_roc_curves(
                roc_curves_all,
                title="ROC Curves by Confidence Type",
                save_path=output_dir / "roc_curves.png",
            )
            console.print("[green]Saved ROC curves: roc_curves.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate ROC curves: {e}[/yellow]")

    # Generate ROC curves by demographic group
    roc_by_group: dict[str, dict] = {}
    for conf_type in conf_types:
        try:
            arrays_by_group = analyzer.get_confidence_arrays(conf_type)
            label = conf_type.replace("_confidence", "").replace("_", " ")
            for group, (confs, corr) in arrays_by_group.items():
                if group not in roc_by_group:
                    roc_by_group[group] = {}
                roc_by_group[group][label] = (corr, confs)
        except Exception:
            pass

    if roc_by_group:
        try:
            plot_roc_by_demographic(
                roc_by_group,
                title="ROC Curves by Demographic",
                save_path=output_dir / "roc_by_demographic.png",
            )
            console.print("[green]Saved demographic ROC curves: roc_by_demographic.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate demographic ROC curves: {e}[/yellow]")

    # Compute calibration gaps
    console.print("\n[bold]Calibration Gaps:[/bold]")
    for conf_type in conf_types:
        gap_info = analyzer.compute_calibration_gap(conf_type)
        if "max_gap" in gap_info:
            console.print(
                f"  {conf_type.replace('_confidence', '')}: "
                f"max_gap={gap_info['max_gap']:.4f}"
            )


@app.command()
def report(
    results_path: Path = typer.Argument(..., help="Path to saved results JSON file"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory for report and figures"
    ),
    model_name: str = typer.Option(
        "Unknown Model", "--model-name", "-m", help="Model name for report header"
    ),
    n_bootstrap: int = typer.Option(
        1000, "--bootstrap", "-b", help="Number of bootstrap samples"
    ),
):
    """
    Generate a comprehensive analysis report with improved visualizations.

    Produces a Markdown report with statistical analysis and new figures
    designed for cases where logprob/self-consistency confidence signals
    are degenerate (near-constant values).

    Example:
        democalib report outputs/results/gpt4o_results.json -o outputs/report -m "GPT-4o"
    """
    console.print(f"[bold]Loading results from {results_path}...[/bold]")

    results = load_results(results_path)
    analyzer = DemographicAnalyzer(results)

    output_dir = output_dir or results_path.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Generate Markdown report
    generator = ReportGenerator(
        analyzer, model_name=model_name, n_bootstrap=n_bootstrap
    )
    report_path = output_dir / "report.md"
    report_text = generator.generate(report_path)
    console.print(f"[green]Report saved to {report_path}[/green]")

    # Print summary to console
    summary = analyzer.generate_summary_table()
    if not summary.empty:
        console.print("\n[bold]Calibration Summary:[/bold]")
        _print_dataframe_table(summary, "Calibration Metrics")

    # Detect confidence columns
    sc_cols = [c for c in analyzer.df.columns if c.startswith("sc_confidence_t")]
    conf_cols = []
    for c in ["logprob_confidence", "verbalized_confidence", "logprob_confidence_freeform"] + sc_cols:
        if c in analyzer.df.columns:
            conf_cols.append(c)

    # Generate new plots
    if conf_cols:
        try:
            plot_confidence_violin(
                analyzer.df, conf_cols,
                save_path=figures_dir / "confidence_violin.png",
            )
            console.print("[green]Saved: confidence_violin.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate violin plot: {e}[/yellow]")

        try:
            plot_overconfidence_bar(
                analyzer.df, conf_cols,
                save_path=figures_dir / "overconfidence_bar.png",
            )
            console.print("[green]Saved: overconfidence_bar.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate overconfidence bar: {e}[/yellow]")

    # Scatter plot (verbalized vs logprob)
    if (
        "verbalized_confidence" in analyzer.df.columns
        and "logprob_confidence" in analyzer.df.columns
    ):
        try:
            plot_confidence_scatter(
                analyzer.df,
                x_column="verbalized_confidence",
                y_column="logprob_confidence",
                save_path=figures_dir / "confidence_scatter.png",
            )
            console.print("[green]Saved: confidence_scatter.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate scatter plot: {e}[/yellow]")

    # Demographic accuracy bar
    best_col = generator._get_best_confidence_column()
    if best_col in analyzer.df.columns:
        try:
            plot_demographic_accuracy_bar(
                analyzer.df,
                confidence_column=best_col,
                bootstrap_n=n_bootstrap,
                save_path=figures_dir / "demographic_accuracy.png",
            )
            console.print("[green]Saved: demographic_accuracy.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate accuracy bar: {e}[/yellow]")

    # Per-group violin plots
    if conf_cols:
        try:
            plot_confidence_violin_by_group(
                analyzer.df, conf_cols,
                save_path=figures_dir / "confidence_violin_by_group.png",
            )
            console.print("[green]Saved: confidence_violin_by_group.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate per-group violin: {e}[/yellow]")

    # Equity comparison
    if conf_cols and len(conf_cols) >= 2:
        try:
            equity_df = analyzer.compute_equity_summary(conf_cols)
            if not equity_df.empty:
                plot_equity_comparison(
                    equity_df,
                    save_path=figures_dir / "equity_comparison.png",
                )
                console.print("[green]Saved: equity_comparison.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate equity comparison: {e}[/yellow]")

    # Logprob method comparison (constrained vs freeform)
    if (
        "logprob_confidence" in analyzer.df.columns
        and "logprob_confidence_freeform" in analyzer.df.columns
    ):
        try:
            plot_logprob_method_comparison(
                analyzer.df,
                save_path=figures_dir / "logprob_method_comparison.png",
            )
            console.print("[green]Saved: logprob_method_comparison.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate logprob comparison: {e}[/yellow]")

    # Compliance summary
    if "logprob_compliant" in analyzer.df.columns:
        try:
            plot_compliance_summary(
                analyzer.df,
                save_path=figures_dir / "compliance_summary.png",
            )
            console.print("[green]Saved: compliance_summary.png[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not generate compliance summary: {e}[/yellow]")

    # Confidence method grid
    try:
        plot_confidence_method_grid(
            analyzer.df,
            save_path=figures_dir / "confidence_method_grid.png",
        )
        console.print("[green]Saved: confidence_method_grid.png[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not generate method grid: {e}[/yellow]")

    console.print(f"\n[bold green]Report generation complete![/bold green]")
    console.print(f"Report: {report_path}")
    console.print(f"Figures: {figures_dir}")


@app.command()
def validate_config(config: Path = typer.Argument(..., help="Path to configuration file")):
    """
    Validate a configuration file without running.

    Example:
        democalib validate-config configs/experiment.yaml
    """
    try:
        if config.suffix in [".yaml", ".yml"]:
            exp_config = ExperimentConfig.from_yaml(config)
        else:
            exp_config = ExperimentConfig.from_json(config)

        console.print("[green]Configuration is valid![/green]")
        _print_config_summary(exp_config)

    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models():
    """List available models."""
    console.print("[bold]Available Models:[/bold]")
    for model in list_models():
        console.print(f"  - {model}")


async def _run_experiment(config: ExperimentConfig, subset: int | None = None):
    """Internal function to run the experiment."""
    from .confidence.aggregator import ConfidenceExtractor, ConfidenceResult
    from .models.task_config import TaskConfig

    # Load dataset
    if config.data.dataset_type == "chexpert":
        from .data.chexpert import CheXpertDataset

        condition = config.data.condition or "Pneumonia"
        console.print(f"[bold]Loading CheXpert dataset (condition: {condition})...[/bold]")
        try:
            dataset = CheXpertDataset(
                metadata_path=config.data.metadata_path,
                images_dir=config.data.images_dir,
                condition=condition,
                demographic_groups=config.data.demographic_groups,
            )
            console.print(f"[green]Loaded {len(dataset)} samples[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("[bold]Loading DDI dataset...[/bold]")
        try:
            dataset = DDIDataset(
                metadata_path=config.data.metadata_path,
                images_dir=config.data.images_dir,
                skin_tone_groups=config.data.skin_tone_groups or config.data.demographic_groups,
            )
            console.print(f"[green]Loaded {len(dataset)} samples[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            raise typer.Exit(1)

    # Print dataset statistics
    stats = dataset.get_group_statistics()
    _print_dataframe_table(stats, "Dataset Statistics")

    # Get all samples (no split — evaluate everything)
    all_samples = list(dataset)

    # Apply subset if specified (stratified to ensure both classes + groups)
    if subset:
        import random

        rng = random.Random(config.data.random_seed)

        # Group samples by (demographic_group, label_name) for stratification
        strata: dict[tuple[str, str], list] = {}
        for s in all_samples:
            key = (s.demographic_group, s.label_name)
            strata.setdefault(key, []).append(s)

        # Allocate proportionally, but guarantee at least 1 per stratum
        n_strata = len(strata)
        per_stratum_min = 1
        remaining = max(0, subset - n_strata * per_stratum_min)

        selected = []
        total_samples = len(all_samples)
        for key, samples in strata.items():
            rng.shuffle(samples)
            # Proportional share of the remaining budget
            proportion = len(samples) / total_samples
            n_take = per_stratum_min + round(remaining * proportion)
            n_take = min(n_take, len(samples))
            selected.extend(samples[:n_take])

        # Trim to exact subset size if rounding overshot
        if len(selected) > subset:
            rng.shuffle(selected)
            selected = selected[:subset]

        all_samples = selected
        # Show class distribution in subset
        from collections import Counter
        label_dist = Counter(s.label_name for s in all_samples)
        group_dist = Counter(s.demographic_group for s in all_samples)
        console.print(
            f"[yellow]Using stratified subset of {len(all_samples)} samples "
            f"(labels: {dict(label_dist)}, groups: {dict(group_dist)})[/yellow]"
        )

    console.print(f"[green]Evaluating {len(all_samples)} samples[/green]")

    # Initialize model
    api_key_env = config.model.get_api_key_env()
    if api_key_env is None:
        api_key = ""
        console.print(f"[cyan]Using local model: {config.model.name}[/cyan]")
    else:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            console.print(f"[red]API key not found in environment: {api_key_env}[/red]")
            console.print(f"Please set the {api_key_env} environment variable.")
            raise typer.Exit(1)

    model_kwargs = {
        "rate_limit_rpm": config.model.rate_limit_rpm,
        "timeout": config.model.timeout,
    }
    if config.model.base_url:
        model_kwargs["base_url"] = config.model.base_url

    model = get_model(
        config.model.name,
        api_key=api_key,
        **model_kwargs,
    )
    console.print(f"[green]Initialized model: {config.model.name}[/green]")

    # Configure task-specific prompts and labels
    if config.data.dataset_type == "chexpert":
        task_config = TaskConfig.chexpert(condition=config.data.condition or "Pneumonia")
        console.print(f"[cyan]Task: CheXpert {config.data.condition} detection[/cyan]")
    else:
        task_config = TaskConfig.ddi()

    # Apply prompt overrides (prompt-sensitivity experiments)
    if config.prompts is not None:
        from dataclasses import replace

        overrides = {
            k: v for k, v in config.prompts.model_dump().items() if v is not None
        }
        if overrides:
            task_config = replace(task_config, **overrides)
            console.print(f"[yellow]Prompt overrides active: {sorted(overrides)}[/yellow]")

    model.set_task_config(task_config)

    # Initialize confidence extractor
    extractor = ConfidenceExtractor(
        model=model,
        extract_logprob=config.confidence.extract_logprob,
        extract_verbalized=config.confidence.extract_verbalized,
        extract_self_consistency=config.confidence.extract_self_consistency,
        extract_embedding_consistency=config.confidence.extract_embedding_consistency,
        self_consistency_k=config.confidence.self_consistency_k,
        self_consistency_temperatures=config.confidence.self_consistency_temperatures,
        embedding_model=config.confidence.embedding_model,
        embedding_api_key=api_key,
        logprob_top_k=config.confidence.logprob_top_k,
        extract_logprob_freeform=config.confidence.extract_logprob_freeform,
        logprob_normalization=config.confidence.logprob_normalization,
    )

    # Compute output paths (needed before extraction for checkpointing)
    output_dir = config.output.output_dir / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    checkpoint_path = output_dir / "_checkpoint.jsonl"

    # Resume detection — load any existing checkpoint
    completed_ids: set[str] = set()
    resumed_results: list[ConfidenceResult] = []

    if checkpoint_path.exists():
        checkpoint_records = load_checkpoint(checkpoint_path)
        completed_ids = {r["sample_id"] for r in checkpoint_records}
        resumed_results = [ConfidenceResult.from_dict(r) for r in checkpoint_records]
        console.print(
            f"[yellow]Resuming: found {len(completed_ids)} completed "
            f"samples in checkpoint[/yellow]"
        )

    remaining_samples = [s for s in all_samples if s.image_id not in completed_ids]
    total_samples = len(all_samples)
    already_done = len(completed_ids)

    # Run extraction with checkpointing
    prompt = model.get_diagnosis_prompt()

    if not remaining_samples:
        console.print("[green]All samples already completed in checkpoint.[/green]")
        results = resumed_results
    else:
        new_results: list[ConfidenceResult] = []
        interrupted = False

        def on_checkpoint(result: ConfidenceResult):
            append_checkpoint(result, checkpoint_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Extracting confidence...",
                total=total_samples,
                completed=already_done,
            )

            def update_progress(current, total):
                progress.update(task, completed=already_done + current)

            try:
                new_results = await extractor.extract_batch(
                    remaining_samples,
                    prompt=prompt,
                    max_tokens=config.model.max_tokens,
                    progress_callback=update_progress,
                    checkpoint_callback=on_checkpoint,
                )
            except KeyboardInterrupt:
                interrupted = True
                console.print(
                    "\n[yellow]Interrupted! Partial results saved to "
                    "checkpoint.[/yellow]"
                )

        if interrupted:
            # Re-read checkpoint for accurate count (includes resumed + new)
            checkpoint_records = load_checkpoint(checkpoint_path)
            console.print(
                f"[yellow]Checkpoint has {len(checkpoint_records)}/{total_samples} "
                f"samples. Re-run the same command to resume.[/yellow]"
            )
            return

        results = resumed_results + new_results

    # Save final results and clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_to_results(checkpoint_path, results_path)
    else:
        save_results(results, results_path)
    console.print(f"[green]Saved {len(results)} results to {results_path}[/green]")

    # Analyze — all results
    analyzer = DemographicAnalyzer(results)

    summary = analyzer.generate_summary_table()

    console.print("\n[bold]Calibration Summary:[/bold]")
    _print_dataframe_table(summary, "Calibration Metrics by Group")

    # Log compliance rate for constrained classification
    compliance_rates = analyzer.compute_compliance_rate()
    if compliance_rates:
        console.print("\n[bold]A/B Classification Compliance:[/bold]")
        for group, rate in compliance_rates.items():
            console.print(f"  {group}: {rate:.1%}")

    # Bootstrap analysis
    console.print("\n[bold]Computing bootstrap confidence intervals...[/bold]")
    sc_conf_cols = [c for c in analyzer.df.columns if c.startswith("sc_confidence_t")]
    all_conf_types = ["logprob_confidence", "verbalized_confidence"] + sc_conf_cols
    # Include freeform baseline if present
    if "logprob_confidence_freeform" in analyzer.df.columns and analyzer.df["logprob_confidence_freeform"].notna().any():
        all_conf_types.append("logprob_confidence_freeform")
    for conf_type in all_conf_types:
        try:
            bootstrap_results = analyzer.compute_bootstrap_by_group(
                conf_type,
                n_bins=config.calibration.n_bins,
                n_bootstrap=config.calibration.bootstrap_n_samples,
                ci_level=config.calibration.bootstrap_ci_level,
            )

            for group, metrics in bootstrap_results.items():
                console.print(f"\n[cyan]{group} - {conf_type.replace('_confidence', '')}:[/cyan]")
                for metric_name, result in metrics.items():
                    console.print(
                        f"  {metric_name}: {result.point_estimate:.4f} "
                        f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
                    )
        except Exception as e:
            console.print(f"[yellow]Bootstrap failed for {conf_type}: {e}[/yellow]")

    # Temperature scaling (5-fold CV — no data wasted)
    console.print("\n[bold]Fitting demographic temperature scaling (5-fold CV)...[/bold]")
    from .calibration.temperature_scaling import DemographicTemperatureScaler

    fitted_scalers: dict = {}  # conf_type -> DemographicTemperatureScaler
    for conf_type in ["logprob_confidence", "verbalized_confidence"]:
        try:
            valid_mask = analyzer.df[conf_type].notna()
            if valid_mask.sum() < 10:
                continue
            confidences = analyzer.df.loc[valid_mask, conf_type].values.astype(float)
            correct_series = analyzer._get_correctness(analyzer.df, conf_type)
            correctness = correct_series.loc[valid_mask].values.astype(int)
            groups = analyzer.df.loc[valid_mask, "demographic_group"].values

            scaler = DemographicTemperatureScaler()
            scaler.fit_transform_cv(
                confidences, correctness, groups,
                n_folds=5,
                random_seed=config.data.random_seed,
            )
            fitted_scalers[conf_type] = scaler
            console.print(f"\n[cyan]Temperature scaling for {conf_type.replace('_confidence', '')}:[/cyan]")
            for group, result in scaler.results.items():
                console.print(
                    f"  {group}: T={result.temperature:.3f} "
                    f"(NLL improvement: {result.improvement:.4f})"
                )
        except Exception as e:
            console.print(f"[yellow]Temperature scaling failed for {conf_type}: {e}[/yellow]")

    # Generate figures
    if config.output.generate_figures:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        fig_conf_types = ["logprob_confidence", "verbalized_confidence"] + sc_conf_cols
        if "logprob_confidence_freeform" in analyzer.df.columns:
            fig_conf_types.append("logprob_confidence_freeform")
        for conf_type in fig_conf_types:
            try:
                metrics_by_group = analyzer.compute_metrics_by_group(conf_type)
                if metrics_by_group:
                    plot_demographic_comparison(
                        metrics_by_group,
                        confidence_type=conf_type.replace("_confidence", ""),
                        save_path=figures_dir / f"reliability_{conf_type}.{config.output.figure_format}",
                    )
            except Exception as e:
                console.print(f"[yellow]Could not generate {conf_type} figure: {e}[/yellow]")

        # Logprob method comparison (constrained vs freeform)
        if (
            "logprob_confidence" in analyzer.df.columns
            and "logprob_confidence_freeform" in analyzer.df.columns
        ):
            try:
                plot_logprob_method_comparison(
                    analyzer.df,
                    save_path=figures_dir / f"logprob_method_comparison.{config.output.figure_format}",
                )
                console.print("[green]Saved: logprob_method_comparison[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not generate logprob comparison: {e}[/yellow]")

        # Compliance summary
        if "logprob_compliant" in analyzer.df.columns:
            try:
                plot_compliance_summary(
                    analyzer.df,
                    save_path=figures_dir / f"compliance_summary.{config.output.figure_format}",
                )
                console.print("[green]Saved: compliance_summary[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not generate compliance summary: {e}[/yellow]")

        # Temperature scaling before/after plots
        for conf_type, scaler in fitted_scalers.items():
            try:
                arrays = analyzer.get_confidence_arrays(conf_type)
                confs_by_group = {g: arr[0] for g, arr in arrays.items() if len(arr[0]) > 0}
                corr_by_group = {g: arr[1] for g, arr in arrays.items() if len(arr[0]) > 0}
                if not confs_by_group:
                    continue
                conf_label = conf_type.replace("_confidence", "")
                plot_temperature_scaling_comparison(
                    confidences_by_group=confs_by_group,
                    correctness_by_group=corr_by_group,
                    scaler=scaler,
                    confidence_type=conf_label,
                    n_bins=config.calibration.n_bins,
                    save_path=figures_dir / f"temp_scaling_{conf_label}.{config.output.figure_format}",
                )
            except Exception as e:
                console.print(f"[yellow]Could not generate temp scaling plot for {conf_type}: {e}[/yellow]")

        console.print(f"[green]Figures saved to {figures_dir}[/green]")

    # Generate comprehensive report with new plots
    console.print("\n[bold]Generating analysis report...[/bold]")
    try:
        report_dir = output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        report_figures_dir = report_dir / "figures"
        report_figures_dir.mkdir(exist_ok=True)

        generator = ReportGenerator(
            analyzer,
            model_name=config.model.name,
            n_bootstrap=config.calibration.bootstrap_n_samples,
        )
        report_path = report_dir / "report.md"
        generator.generate(report_path)
        console.print(f"[green]Report saved to {report_path}[/green]")

        # New plots
        fig_fmt = config.output.figure_format
        conf_cols = [c for c in all_conf_types if c in analyzer.df.columns]

        if conf_cols:
            try:
                plot_confidence_violin(
                    analyzer.df, conf_cols,
                    save_path=report_figures_dir / f"confidence_violin.{fig_fmt}",
                )
            except Exception:
                pass

            try:
                plot_overconfidence_bar(
                    analyzer.df, conf_cols,
                    save_path=report_figures_dir / f"overconfidence_bar.{fig_fmt}",
                )
            except Exception:
                pass

        if (
            "verbalized_confidence" in analyzer.df.columns
            and "logprob_confidence" in analyzer.df.columns
        ):
            try:
                plot_confidence_scatter(
                    analyzer.df,
                    save_path=report_figures_dir / f"confidence_scatter.{fig_fmt}",
                )
            except Exception:
                pass

        best_col = generator._get_best_confidence_column()
        if best_col in analyzer.df.columns:
            try:
                plot_demographic_accuracy_bar(
                    analyzer.df,
                    confidence_column=best_col,
                    bootstrap_n=config.calibration.bootstrap_n_samples,
                    save_path=report_figures_dir / f"demographic_accuracy.{fig_fmt}",
                )
            except Exception:
                pass

        # Per-group violin plots
        if conf_cols:
            try:
                plot_confidence_violin_by_group(
                    analyzer.df, conf_cols,
                    save_path=report_figures_dir / f"confidence_violin_by_group.{fig_fmt}",
                )
            except Exception:
                pass

        # Equity comparison
        if conf_cols and len(conf_cols) >= 2:
            try:
                equity_df = analyzer.compute_equity_summary(conf_cols)
                if not equity_df.empty:
                    plot_equity_comparison(
                        equity_df,
                        save_path=report_figures_dir / f"equity_comparison.{fig_fmt}",
                    )
            except Exception:
                pass

        # Logprob method comparison (constrained vs freeform)
        if (
            "logprob_confidence" in analyzer.df.columns
            and "logprob_confidence_freeform" in analyzer.df.columns
        ):
            try:
                plot_logprob_method_comparison(
                    analyzer.df,
                    save_path=report_figures_dir / f"logprob_method_comparison.{fig_fmt}",
                )
                console.print("[green]Saved: logprob_method_comparison[/green]")
            except Exception:
                pass

        # Compliance summary
        if "logprob_compliant" in analyzer.df.columns:
            try:
                plot_compliance_summary(
                    analyzer.df,
                    save_path=report_figures_dir / f"compliance_summary.{fig_fmt}",
                )
                console.print("[green]Saved: compliance_summary[/green]")
            except Exception:
                pass

        # Confidence method grid
        try:
            plot_confidence_method_grid(
                analyzer.df,
                save_path=report_figures_dir / f"confidence_method_grid.{fig_fmt}",
            )
            console.print("[green]Saved: confidence_method_grid[/green]")
        except Exception:
            pass

        console.print(f"[green]Report figures saved to {report_figures_dir}[/green]")
    except Exception as e:
        console.print(f"[yellow]Report generation failed: {e}[/yellow]")

    console.print("\n[bold green]Experiment completed successfully![/bold green]")


def _print_config_summary(config: ExperimentConfig):
    """Print configuration summary."""
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="yellow")

    table.add_row("Experiment", "Name", config.experiment_name)
    table.add_row("Data", "Metadata", str(config.data.metadata_path))
    table.add_row("Data", "Images", str(config.data.images_dir))
    table.add_row("Data", "Dataset type", config.data.dataset_type)
    table.add_row("Data", "Demographic groups", str(config.data.demographic_groups))
    if config.data.condition:
        table.add_row("Data", "Condition", config.data.condition)
    table.add_row("Model", "Name", config.model.name)
    table.add_row("Model", "API key env", config.model.get_api_key_env() or "N/A (local)")
    table.add_row("Confidence", "Self-consistency K", str(config.confidence.self_consistency_k))
    table.add_row("Confidence", "Temperatures", str(config.confidence.self_consistency_temperatures))
    table.add_row("Calibration", "Bins", str(config.calibration.n_bins))
    table.add_row("Calibration", "Bootstrap samples", str(config.calibration.bootstrap_n_samples))
    table.add_row("Output", "Directory", str(config.output.output_dir))

    console.print(table)


def _print_dataframe_table(df, title: str = ""):
    """Print DataFrame as rich table."""
    table = Table(title=title)

    for col in df.columns:
        table.add_column(str(col))

    for _, row in df.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
