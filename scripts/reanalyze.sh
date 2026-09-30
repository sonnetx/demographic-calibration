#!/bin/bash
#SBATCH --job-name=democalib-reanalyze
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Reanalyze existing results (no GPU / no inference needed).
# Recomputes all metrics (including balanced accuracy) from saved results.json files.
#
# Results live at: outputs/<experiment_name>/results.json
# Experiment names (from configs):
#   gpt4o_ddi_calibration, r1_ddi_calibration, qwen_vl_ddi_calibration,
#   chexpert_pneumonia_gpt4o, chexpert_effusion_r1, etc.
#
# Usage:
#   sbatch scripts/reanalyze.sh                          # all experiments
#   sbatch --export=MODEL=gpt4o scripts/reanalyze.sh     # matching experiments
#   bash scripts/reanalyze.sh                             # run locally

set -e

PROJECT_DIR="$HOME/demographic-calibration"
OUTPUTS_DIR="${PROJECT_DIR}/outputs"
VENV_API="${PROJECT_DIR}/venv"
MODEL="${MODEL:-all}"

# =============================================================================
# Setup
# =============================================================================

if [ -f ~/.secrets ]; then
    source ~/.secrets
fi

# Use the lightweight API venv (no GPU deps needed)
if command -v ml &>/dev/null; then
    ml purge
    ml openblas/0.3.28
    ml python/3.12.1
fi

if [ -d "$VENV_API" ]; then
    source "$VENV_API/bin/activate"
else
    echo "ERROR: venv not found at $VENV_API"
    echo "Run the experiment first, or create the venv manually."
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

# Ensure package is installed
pip install --quiet -e "${PROJECT_DIR}" 2>/dev/null || true

mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "Reanalyzing results (model=${MODEL})"
echo "Outputs dir: ${OUTPUTS_DIR}"
echo "=========================================="

# =============================================================================
# Reanalyze
# =============================================================================

reanalyze() {
    local results_json="$1"
    local name
    name="$(basename "$(dirname "$results_json")")"

    if [ ! -f "$results_json" ]; then
        echo "SKIP: $results_json not found"
        return
    fi

    local parent_dir
    parent_dir="$(dirname "$results_json")"

    echo ""
    echo "--- ${name} ---"

    # Regenerate figures
    echo "  analyze -> ${parent_dir}/figures/"
    democalib analyze "$results_json" -o "${parent_dir}/figures"

    # Regenerate full report
    echo "  report  -> ${parent_dir}/report/"
    democalib report "$results_json" -o "${parent_dir}/report"

    echo "  done."
}

if [ "$MODEL" = "all" ]; then
    # Process every results.json under outputs/*/
    found=0
    for f in "${OUTPUTS_DIR}"/*/results.json; do
        if [ -f "$f" ]; then
            reanalyze "$f"
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "ERROR: No results.json files found in ${OUTPUTS_DIR}/*/results.json"
        echo "Available directories:"
        ls "${OUTPUTS_DIR}" 2>/dev/null || echo "  (outputs/ does not exist)"
        exit 1
    fi
else
    # Match experiment dirs containing the MODEL string
    # e.g., MODEL=gpt4o matches gpt4o_ddi_calibration, quick_test_gpt4o, etc.
    found=0
    for f in "${OUTPUTS_DIR}"/*${MODEL}*/results.json; do
        if [ -f "$f" ]; then
            reanalyze "$f"
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "ERROR: No results.json found matching '${MODEL}' in ${OUTPUTS_DIR}"
        echo "Available directories:"
        ls "${OUTPUTS_DIR}" 2>/dev/null || echo "  (outputs/ does not exist)"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Reanalysis complete!"
echo "=========================================="
