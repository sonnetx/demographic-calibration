#!/bin/bash
#SBATCH --job-name=setup_llama
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP: Creates llama_env venv for Llama-3.2-11B-Vision-Instruct.
# Requires transformers >= 4.45 for MllamaForConditionalGeneration.
# The model is gated on Hugging Face, so accept its license and log in
# (huggingface-cli login, or export HF_TOKEN) before the first run.
#
# Usage:
#   sbatch scripts/setup_llama_env.sh
# =============================================================================

set -e

PROJECT_DIR="${PROJECT_DIR:-$HOME/demographic-calibration}"
VENV_LLAMA="${VENV_LLAMA:-${PROJECT_DIR}/llama_env}"
SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"

export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_HOME="${SCRATCH_DIR}/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME" "${PROJECT_DIR}/logs"

# --- Module setup ---
# CUDA 12.1 is required: PyTorch wheels for cu117 are 2.0.x but
# transformers >= 4.45 (MllamaForConditionalGeneration) needs PyTorch >= 2.1.
ml python/3.9.0
ml cuda/12.1.1
ml gcc/14.2.0

# --- Create venv ---
if [ -d "$VENV_LLAMA" ]; then
    echo "INFO: Venv already exists at $VENV_LLAMA"
else
    echo "INFO: Creating virtual environment..."
    python3 -m venv "$VENV_LLAMA"
fi

source "$VENV_LLAMA/bin/activate"
pip install --quiet --upgrade pip

# --- Install PyTorch (CUDA 12.1) ---
echo "INFO: Installing PyTorch..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Install transformers >= 4.45 (Llama 3.2 Vision support) ---
echo "INFO: Installing transformers and accelerate..."
pip install --quiet "transformers>=4.45.0" "accelerate>=0.25.0"

# --- Install project CLI dependencies ---
echo "INFO: Installing project dependencies..."
pip install --quiet \
    "numpy>=1.24.0,<2" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0,<1.17.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "pillow>=10.0.0" \
    "pydantic>=2.0.0" \
    "pyyaml>=6.0.0" \
    "typer>=0.9.0" \
    "rich>=13.0.0" \
    "openai>=1.0.0" \
    "tqdm>=4.66.0"

# --- Verify ---
echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="

export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

from transformers import MllamaForConditionalGeneration, AutoProcessor
print('MllamaForConditionalGeneration: OK')

from democalib.config import ExperimentConfig
from democalib.models.registry import get_model
print('democalib imports: OK')
"

deactivate

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Venv: $VENV_LLAMA"
echo ""
echo "Next: sbatch --export=MODEL=llama,MODE=quick scripts/run_chexpert.sh"
echo "=========================================="
