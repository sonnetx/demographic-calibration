"""Intersectional (sex x age) calibration analysis for CheXpert.

Reported in the paper appendix. For each (model, condition, signal), computes:
  - per-cell ECE (sex x age binarized at 65) with bootstrap 95% CIs,
  - the marginal (sex) and intersectional (sex x age) max ECE gaps,
  - a permutation p-value for the intersectional gap, where the null shuffles
    age labels within each sex group (finer splitting inflates max gaps under
    noise alone, so the observed gap must beat this splitting-inflation null).

DDI is excluded: the public DDI metadata releases only skin tone and diagnosis,
so intersectional strata cannot be constructed for the dermatology task.

Usage:
    python scripts/intersectional_analysis.py \
        --outputs-dir outputs \
        --chexpert-metadata /path/to/train_valid_combined.csv \
        -o outputs/intersectional

The CheXpert metadata CSV must contain `image_id` and `Age` columns; join is on
results.json `sample_id` == metadata `image_id`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    ("gpt4o", "pneumonia"): "chexpert_pneumonia_gpt4o",
    ("r1", "pneumonia"): "chexpert_pneumonia_r1",
    ("qwen", "pneumonia"): "chexpert_pneumonia_qwen_vl",
    ("gpt4o", "effusion"): "chexpert_effusion_gpt4o",
    ("r1", "effusion"): "chexpert_effusion_r1",
    ("qwen", "effusion"): "chexpert_effusion_qwen_vl",
}

SIGNALS = {
    "TLP": "logprob_confidence",
    "CE": "verbalized_confidence",
    "SC0.5": "sc_confidence_t0.5",
    "SC1.0": "sc_confidence_t1.0",
}

AGE_CUTOFF = 65
MIN_CELL = 30


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if mask.sum():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def bootstrap_ci(conf, correct, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    stats = [compute_ece(conf[i], correct[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(stats, [2.5, 97.5])


def correctness(df: pd.DataFrame, conf_col: str) -> pd.Series:
    """SC signals score the majority-vote prediction; others use is_correct."""
    if conf_col.startswith("sc_confidence_"):
        pred_col = "sc_prediction_" + conf_col.removeprefix("sc_confidence_")
        if pred_col in df.columns:
            return (df[pred_col] == df["ground_truth"]).astype(float)
    return df["is_correct"].astype(float)


def max_gap(eces: list[float]) -> float:
    return max(eces) - min(eces)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--chexpert-metadata", type=Path, required=True,
                    help="CSV with image_id and Age columns")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/intersectional"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.chexpert_metadata, usecols=["image_id", "Age"])
    meta = meta.drop_duplicates("image_id").set_index("image_id")["Age"]
    rng = np.random.default_rng(args.seed)

    cell_rows, gap_rows = [], []
    for (model, cond), run in RUNS.items():
        path = args.outputs_dir / run / "results.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        df = pd.DataFrame(json.load(open(path)))
        df["Age"] = df["sample_id"].map(meta)
        if df["Age"].isna().any():
            print(f"[warn] {run}: {df['Age'].isna().sum()} samples missing age")
        df["age_group"] = np.where(df["Age"] >= AGE_CUTOFF, "65+", "<65")

        for sig, col in SIGNALS.items():
            sub = df[df[col].notna()].copy()
            if sub.empty:
                continue
            conf = sub[col].values.astype(float)
            corr = correctness(sub, col).values
            sex = sub["demographic_group"].values
            age = sub["age_group"].values

            # per-cell ECE + CI
            cell_eces = {}
            for s in np.unique(sex):
                for a in ["<65", "65+"]:
                    m = (sex == s) & (age == a)
                    if m.sum() < MIN_CELL:
                        continue
                    ece = compute_ece(conf[m], corr[m])
                    lo, hi = bootstrap_ci(conf[m], corr[m])
                    cell_eces[f"{s} {a}"] = ece
                    cell_rows.append(dict(model=model, condition=cond, signal=sig,
                                          cell=f"{s} {a}", n=int(m.sum()),
                                          ece=round(ece, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

            marg = [compute_ece(conf[sex == s], corr[sex == s]) for s in np.unique(sex)]

            def four_cell_gap(agelab):
                eces = [compute_ece(conf[(sex == s) & (agelab == a)], corr[(sex == s) & (agelab == a)])
                        for s in np.unique(sex) for a in ["<65", "65+"]
                        if ((sex == s) & (agelab == a)).sum() >= MIN_CELL]
                return max_gap(eces)

            obs = four_cell_gap(age)
            null = []
            for _ in range(args.n_perm):
                perm = age.copy()
                for s in np.unique(sex):
                    m = sex == s
                    perm[m] = rng.permutation(perm[m])
                null.append(four_cell_gap(perm))
            null = np.asarray(null)
            gap_rows.append(dict(model=model, condition=cond, signal=sig,
                                 marginal_sex_gap=round(max_gap(marg), 4),
                                 intersectional_gap=round(obs, 4),
                                 null_mean=round(float(null.mean()), 4),
                                 null_p95=round(float(np.percentile(null, 95)), 4),
                                 p_value=round(float((null >= obs).mean()), 4),
                                 degenerate=bool(np.std(conf) < 1e-6)))

    cells = pd.DataFrame(cell_rows)
    gaps = pd.DataFrame(gap_rows)
    cells.to_csv(args.output_dir / "intersectional_cells.csv", index=False)
    gaps.to_csv(args.output_dir / "intersectional_gaps.csv", index=False)
    print(gaps.to_string(index=False))
    n_tests = len(gaps)
    print(f"\nBonferroni threshold for {n_tests} tests: {0.05 / n_tests:.4f}")
    print(f"Combinations where intersectional > marginal gap: "
          f"{(gaps.intersectional_gap > gaps.marginal_sex_gap).sum()}/{n_tests}")
    print(f"Wrote {args.output_dir}/intersectional_cells.csv and intersectional_gaps.csv")


if __name__ == "__main__":
    main()
