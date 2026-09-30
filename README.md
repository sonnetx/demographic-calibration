# When Reasoning Hurts Confidence: Demographic Calibration of Vision-Language Models in Medicine

This repository contains the code for the paper of the same name by Sonnet Xu and Roxana Daneshjou, presented at the AAAI/ACM Conference on AI, Ethics, and Society (AIES 2026).

The paper asks whether the confidence a vision-language model (VLM) attaches to a medical diagnosis is equally reliable for every patient group. We evaluate GPT-4o, R1-Onevision-7B, and Qwen2.5-VL-7B on skin lesion and chest radiograph classification, stratify every metric by Fitzpatrick skin type or patient sex, and compare three ways of extracting confidence from the same answer. In six of nine model and task combinations, the best-calibrated confidence signal is not the most equitable one. Temperature scaling reduces ECE by up to 96% but leaves the gaps between groups unchanged. R1-Onevision and Qwen2.5-VL share a backbone, yet a different signal is most useful for each. In our study, no single confidence method transfers reliably across models and tasks.

## Study design

We run two datasets as three binary tasks. DDI (Diverse Dermatology Images) contains 656 clinical photographs labeled benign or malignant, grouped by Fitzpatrick skin type into light (I-II), medium (III-IV), and dark (V-VI). CheXpert supplies a stratified 656-image subset of chest radiographs for two conditions, Pneumonia (3.4% positive) and Pleural Effusion (27.7% positive), grouped by patient sex.

Every model answers the same constrained A/B prompt, and we derive three confidence signals from it:

- Token-level probability (TLP) normalizes the probabilities of the A and B answer tokens. For R1-Onevision we read them at the answer token after the `<think>` block closes.
- Verbalized confidence (CE) asks the model, in a second turn, how confident it is in its answer on a 0 to 100 scale.
- Self-consistency (SC) samples 15 answers at temperatures 0.5 and 1.0 and takes the fraction that agree with the majority.

For each signal we report ECE (10 equal-width bins), AUROC, accuracy, and balanced accuracy, all with 95% bootstrap intervals over 1,000 resamples. Equity is the maximum ECE gap and the ECE ratio across groups. We also fit global and group-conditional temperature scaling. A Llama-3.2-11B-Vision replication on CheXpert tests whether the constant CE and SC outputs we see from Qwen2.5-VL come from the protocol or the model.

## Installation

```bash
git clone https://github.com/sonnetx/demographic-calibration.git
cd demographic-calibration
pip install -e .            # analysis code and the GPT-4o (API) backend
pip install -e ".[local]"   # adds torch and transformers for the open-weight models
```

The package needs Python 3.9 or newer. R1-Onevision, Qwen2.5-VL, and Llama-3.2-Vision run locally and need a CUDA GPU that can hold a 7B to 11B model in bfloat16. Llama-3.2-Vision also needs transformers 4.45 or newer, and its weights are gated on Hugging Face, so accept the license and log in before the first run. `scripts/setup_llama_env.sh` builds a separate environment for it.

GPT-4o runs through the OpenAI API and reads its key from `OPENAI_API_KEY`.

## Data

Neither dataset ships with this repository. Both are available from the Stanford AIMI Center under their own data use agreements.

- DDI: https://ddi-dataset.github.io/
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/

DDI works as released. Pass its `ddi_metadata.csv` and image folder to the run command.

For CheXpert, concatenate the official `train.csv` and `valid.csv` into one CSV and copy every image into a single flat directory. The loader takes each row's `Path` column and derives a flat filename of the form `patient00001_study1_view1_frontal.jpg`, then looks for that file in the image directory. Alternatively, add an `image_id` column holding that filename, which the intersectional analysis also requires. Uncertain (-1) and blank labels count as absent.

Configs in `configs/` point to placeholder paths such as `/path/to/ddi/`. Edit them, or override the paths on the command line as shown below.

## Running an experiment

Each config defines one model on one task. A run writes per-sample predictions and all three confidence signals to `outputs/<experiment_name>/results.json`, along with metrics and figures.

```bash
# DDI
democalib run configs/gpt4o_experiment.yaml \
    --metadata /data/ddi/ddi_metadata.csv --images /data/ddi/images

# CheXpert (the paper uses the stratified 656-image subset)
democalib run configs/chexpert_gpt4o.yaml \
    --metadata /data/chexpert/train_valid_combined.csv --images /data/chexpert/images \
    --subset 656
```

`--subset` samples each (group, label) stratum in proportion to its size, keeping at least one sample per stratum, and draws with the config's `random_seed` (42 in every config). Use `--dry-run` to check a config without calling a model.

| Task | GPT-4o | R1-Onevision-7B | Qwen2.5-VL-7B | Llama-3.2-11B-Vision |
|---|---|---|---|---|
| DDI | `gpt4o_experiment.yaml` | `r1_experiment.yaml` | `qwen_vl_experiment.yaml` | |
| CheXpert Pneumonia | `chexpert_gpt4o.yaml` | `chexpert_r1_experiment.yaml` | `chexpert_qwen_vl.yaml` | `chexpert_llama_vision.yaml` |
| CheXpert Pleural Effusion | `chexpert_effusion_gpt4o.yaml` | `chexpert_effusion_r1_experiment.yaml` | `chexpert_effusion_qwen_vl.yaml` | `chexpert_effusion_llama_vision.yaml` |

### On a SLURM cluster

`scripts/run_experiment.sh` (DDI) and `scripts/run_chexpert.sh` (CheXpert) are the batch launchers we used. Select the model and condition with environment variables.

```bash
sbatch --export=MODEL=qwen scripts/run_experiment.sh
sbatch --export=MODEL=r1,CONDITION=effusion scripts/run_chexpert.sh
sbatch --export=MODEL=llama scripts/run_chexpert.sh
```

The launchers read `PROJECT_DIR` (default `$HOME/demographic-calibration`), `VENV_LOCAL` and `VENV_LLAMA` for the model environments, and `SCRATCH` for model caches. They source `~/.secrets` for API keys if that file exists. The `ml` module names and partitions come from our cluster, so change them to match yours.

## Reproducing the paper

Once all nine main runs have finished, these commands regenerate the paper's results from the saved `results.json` files. None of them call a model.

| Result | Command |
|---|---|
| Per-group metrics and reliability diagrams for one run | `democalib analyze outputs/<run>/results.json` |
| Markdown report for one run, including a test for disproportionate high-confidence failures across groups | `democalib report outputs/<run>/results.json` |
| Calibration-equity tradeoff figure, reasoning vs. standard figure, DDI and Pleural Effusion reliability diagrams, and the aggregate metric, equity, and signal-correlation tables | `python scripts/cross_model_figures.py --results ...` (full argument list in the script's docstring) |
| Group-conditional temperature scaling with 5-fold cross-validation | `python scripts/get_temp_scaling.py` |
| Intersectional sex by age analysis (supplementary material) | `python scripts/intersectional_analysis.py --chexpert-metadata <csv>` |

`scripts/reanalyze.sh` reruns `analyze` and `report` over every run in `outputs/`.

## Repository layout

```
src/democalib/
  data/          DDI and CheXpert loaders, stratified splits
  models/        GPT-4o (OpenAI API) and local Hugging Face backends, task prompts
  confidence/    TLP, verbalized, and self-consistency extraction
  calibration/   ECE and related metrics, bootstrap intervals, temperature scaling
  analysis/      per-group analysis, reliability diagrams, reports
  cli.py         the democalib command
configs/         one YAML file per model and task
scripts/         SLURM launchers and paper analysis scripts
```

The model registry also contains backends for Gemini, GLM-4.1V, MedGemma, OpenFlamingo, and a vLLM server. The paper does not use them, and we have not tested them for this release.

## Citation

```bibtex
@inproceedings{xu2026reasoning,
  title     = {When Reasoning Hurts Confidence: Demographic Calibration of Vision-Language Models in Medicine},
  author    = {Xu, Sonnet and Daneshjou, Roxana},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year      = {2026}
}
```

## License

We release the code under the MIT License (see [LICENSE](LICENSE)). The DDI and CheXpert data remain under their own licenses and data use agreements. Do not redistribute images or per-sample model outputs derived from them.
