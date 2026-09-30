#!/bin/bash
#SBATCH --job-name=democalib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Demographic Calibration Study - DDI Experiment Runner
# SPDX-License-Identifier: MIT
#
# Runs the three models studied in the paper on DDI (skin-lesion) classification.
# GPT-4o is API-only (no GPU needed); R1-Onevision and Qwen2.5-VL run locally.
#
# Usage:
#     sbatch scripts/run_experiment.sh                              # all three models
#     sbatch --export=MODEL=gpt4o scripts/run_experiment.sh         # GPT-4o only
#     sbatch --export=MODEL=r1 scripts/run_experiment.sh            # R1-Onevision only
#     sbatch --export=MODEL=qwen scripts/run_experiment.sh          # Qwen2.5-VL only
#
#   Quick check on a subset:
#     sbatch --export=MODEL=gpt4o,SUBSET=50 scripts/run_experiment.sh

set -e

# =============================================================================
# Configuration
# =============================================================================

MODEL="${MODEL:-all}"    # gpt4o, r1, qwen, or all
SUBSET="${SUBSET:-}"     # Number of samples (empty = full dataset, e.g. SUBSET=50)

PROJECT_DIR="${PROJECT_DIR:-$HOME/demographic-calibration}"
VENV_API="${PROJECT_DIR}/venv"
VENV_LOCAL="${VENV_LOCAL:-${PROJECT_DIR}/local_env}"  # shared by R1-Onevision and Qwen2.5-VL
SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"

CONFIG_GPT4O="${PROJECT_DIR}/configs/gpt4o_experiment.yaml"
CONFIG_R1="${PROJECT_DIR}/configs/r1_experiment.yaml"
CONFIG_QWEN="${PROJECT_DIR}/configs/qwen_vl_experiment.yaml"

# =============================================================================
# Common Setup
# =============================================================================

# Load API keys
if [ -f ~/.secrets ]; then
    source ~/.secrets
fi

export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_HOME="${SCRATCH_DIR}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_DIR}/huggingface/datasets"
export TORCH_HOME="${SCRATCH_DIR}/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME "${PROJECT_DIR}/logs"

echo "=========================================="
echo "Model: ${MODEL}"
[ -n "$SUBSET" ] && echo "Subset: ${SUBSET} samples"
[ -z "$SUBSET" ] && echo "Subset: full dataset"
echo "=========================================="

# =============================================================================
# Environment Setup Functions
# =============================================================================

setup_gpt4o_env() {
    echo "Setting up GPT-4o environment..."
    ml purge
    ml openblas/0.3.28
    ml python/3.12.1

    if [ -d "$VENV_API" ] && ! "$VENV_API/bin/python3" --version &>/dev/null; then
        echo "Stale venv detected, removing..."
        rm -rf "$VENV_API"
    fi
    if [ ! -d "$VENV_API" ]; then
        echo "Creating API virtual environment..."
        python3 -m venv "$VENV_API"
    fi
    source "$VENV_API/bin/activate"
    export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
    pip install --quiet --upgrade pip
    pip install --quiet -e "${PROJECT_DIR}"
}

setup_local_env() {
    # R1-Onevision and Qwen2.5-VL share the same Qwen2.5-VL backbone and venv.
    echo "Setting up local (Qwen2.5-VL) environment..."
    ml python/3.9.0
    ml cuda/11.7.1
    ml gcc/14.2.0

    # Unset - not supported by the older PyTorch in the local venv
    unset PYTORCH_CUDA_ALLOC_CONF

    source "$VENV_LOCAL/bin/activate"

    # Install base CLI dependencies if not present (doesn't modify ML deps)
    if ! python -c "import typer" &>/dev/null; then
        echo "Installing CLI dependencies..."
        pip install --quiet typer rich pydantic pyyaml numpy pandas scikit-learn scipy matplotlib seaborn pillow tqdm openai google-genai
    fi

    export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
}

# =============================================================================
# Run Functions
# =============================================================================

run_gpt4o() {
    echo ""
    echo "=========================================="
    echo "Running GPT-4o"
    echo "Config: ${CONFIG_GPT4O}"
    echo "=========================================="
    setup_gpt4o_env
    echo "Python: $(which python) ($(python --version))"
    SUBSET_FLAG=""
    [ -n "$SUBSET" ] && SUBSET_FLAG="--subset $SUBSET"
    democalib run "${CONFIG_GPT4O}" $SUBSET_FLAG
    deactivate
}

run_r1() {
    echo ""
    echo "=========================================="
    echo "Running R1-Onevision-7B"
    echo "Config: ${CONFIG_R1}"
    echo "=========================================="
    setup_local_env
    echo "Python: $(which python) ($(python --version))"
    echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
    SUBSET_FLAG=""
    [ -n "$SUBSET" ] && SUBSET_FLAG="--subset $SUBSET"
    python -m democalib.cli run "${CONFIG_R1}" $SUBSET_FLAG
    deactivate
}

run_qwen() {
    echo ""
    echo "=========================================="
    echo "Running Qwen2.5-VL-7B-Instruct"
    echo "Config: ${CONFIG_QWEN}"
    echo "=========================================="
    setup_local_env
    echo "Python: $(which python) ($(python --version))"
    echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
    SUBSET_FLAG=""
    [ -n "$SUBSET" ] && SUBSET_FLAG="--subset $SUBSET"
    python -m democalib.cli run "${CONFIG_QWEN}" $SUBSET_FLAG
    deactivate
}

# =============================================================================
# Main
# =============================================================================

case "$MODEL" in
    gpt4o)
        run_gpt4o
        ;;
    r1)
        run_r1
        ;;
    qwen)
        run_qwen
        ;;
    all)
        run_gpt4o
        run_r1
        run_qwen
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Valid options: gpt4o, r1, qwen, all"
        exit 1
        ;;
esac

echo ""
echo "Experiment Complete!"
