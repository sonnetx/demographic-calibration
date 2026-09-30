"""Extract temperature scaling results for all model x dataset combinations.

Usage:
    python scripts/get_temp_scaling.py                    # run all
    python scripts/get_temp_scaling.py gpt4o_ddi_calibration  # run one
"""
import sys
from pathlib import Path

from democalib.utils.io import load_results
from democalib.analysis.demographic_analysis import DemographicAnalyzer
from democalib.calibration.temperature_scaling import DemographicTemperatureScaler
from democalib.calibration.metrics import compute_ece

# All 9 model x dataset experiments
ALL_EXPERIMENTS = [
    "gpt4o_ddi_calibration",
    "chexpert_pneumonia_gpt4o",
    "chexpert_effusion_gpt4o",
    "r1_ddi_calibration",
    "chexpert_pneumonia_r1",
    "chexpert_effusion_r1",
    "qwen_vl_ddi_calibration",
    "chexpert_pneumonia_qwen_vl",
    "chexpert_effusion_qwen_vl",
]

CONFIDENCE_SIGNALS = [
    "logprob_confidence",
    "verbalized_confidence",
    "sc_confidence_t0.5",
    "sc_confidence_t1.0",
]


def run_temp_scaling(experiment_name: str) -> None:
    results_path = Path("outputs") / experiment_name / "results.json"
    if not results_path.exists():
        print(f"\n*** SKIPPING {experiment_name}: {results_path} not found ***")
        return

    print(f"\n{'=' * 60}")
    print(f"  {experiment_name}")
    print(f"{'=' * 60}")

    results = load_results(str(results_path))
    analyzer = DemographicAnalyzer(results)

    for conf_type in CONFIDENCE_SIGNALS:
        if conf_type not in analyzer.df.columns:
            continue

        valid = analyzer.df[conf_type].notna()
        if valid.sum() < 10:
            continue

        confs = analyzer.df.loc[valid, conf_type].values.astype(float)
        correct = analyzer._get_correctness(analyzer.df, conf_type).loc[valid].values.astype(int)
        groups = analyzer.df.loc[valid, "demographic_group"].values

        scaler = DemographicTemperatureScaler()
        scaled = scaler.fit_transform_cv(confs, correct, groups, n_folds=5, random_seed=42)

        print(f"\n--- {conf_type} ---")
        for g, r in scaler.results.items():
            mask = groups == g
            post_ece, _, _, _ = compute_ece(scaled[mask], correct[mask])
            pre_ece, _, _, _ = compute_ece(confs[mask], correct[mask])
            print(
                f"  {g:>8s}: T={r.temperature:.3f}, "
                f"ECE {pre_ece:.3f} -> {post_ece:.3f}, "
                f"NLL improvement={r.improvement:.4f}"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific experiment(s)
        for name in sys.argv[1:]:
            run_temp_scaling(name)
    else:
        # Run all
        for name in ALL_EXPERIMENTS:
            run_temp_scaling(name)
