#!/bin/bash
#SBATCH --job-name=democalib-tempscaling
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Compute temperature scaling results from saved experiment data.
# No GPU / no API calls needed — pure local computation.
#
# Checks which output directories exist and runs temperature scaling
# for all 9 model x dataset combinations (GPT-4o, R1, Qwen x DDI, Pneumonia, Effusion).
#
# Usage:
#   sbatch scripts/run_temp_scaling.sh
#   bash scripts/run_temp_scaling.sh    # run locally

set -e

PROJECT_DIR="$HOME/demographic-calibration"
VENV_API="${PROJECT_DIR}/venv"

# Setup
if [ -f ~/.secrets ]; then
    source ~/.secrets
fi

if command -v ml &>/dev/null; then
    ml purge
    ml openblas/0.3.28
    ml python/3.12.1
fi

if [ -d "$VENV_API" ]; then
    source "$VENV_API/bin/activate"
else
    echo "ERROR: venv not found at $VENV_API"
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
pip install --quiet -e "${PROJECT_DIR}" 2>/dev/null || true

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

echo "=========================================="
echo "Temperature Scaling: All Experiments"
echo "=========================================="

# First, check which results exist
echo ""
echo "Checking available results..."
for exp in gpt4o_ddi_calibration chexpert_pneumonia_gpt4o chexpert_effusion_gpt4o \
           r1_ddi_calibration chexpert_pneumonia_r1 chexpert_effusion_r1 \
           qwen_vl_ddi_calibration chexpert_pneumonia_qwen_vl chexpert_effusion_qwen_vl; do
    if [ -f "outputs/${exp}/results.json" ]; then
        echo "  ✓ ${exp}"
    else
        echo "  ✗ ${exp} (missing)"
    fi
done

echo ""
echo "Running temperature scaling..."
python "${PROJECT_DIR}/scripts/get_temp_scaling.py"

echo ""
echo "Done!"
