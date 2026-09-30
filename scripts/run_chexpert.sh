#!/bin/bash
#SBATCH --job-name=chexpert_calib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CheXpert Calibration Study - Experiment Runner
# SPDX-License-Identifier: MIT
#
# Runs the three models studied in the paper on CheXpert classification, plus
# the Llama-3.2-11B-Vision replication of the CE/SC degeneracy.
# Default #SBATCH directives are for the local models (GPU required).
# For GPT-4o (API-only), override partition and drop the GPU:
#
# Usage:
#   R1-Onevision (default, GPU, Pneumonia):
#     sbatch scripts/run_chexpert.sh
#
#   GPT-4o (API-only, no GPU needed):
#     sbatch --partition=normal --mem=16G --gres="" --export=MODEL=gpt4o scripts/run_chexpert.sh
#
#   Qwen2.5-VL (GPU required, local inference):
#     sbatch --export=MODEL=qwen scripts/run_chexpert.sh
#
#   Llama-3.2-11B-Vision (GPU required, local inference; run setup_llama_env.sh once first):
#     sbatch --export=MODEL=llama scripts/run_chexpert.sh
#
#   Pleural Effusion condition:
#     sbatch --partition=normal --mem=16G --gres="" --export=MODEL=gpt4o,CONDITION=effusion scripts/run_chexpert.sh
#
#   Custom subset:
#     sbatch --export=MODEL=r1,SUBSET=100 scripts/run_chexpert.sh

set -e

# =============================================================================
# Configuration
# =============================================================================

MODEL="${MODEL:-r1}"        # r1, gpt4o, qwen, or llama
MODE="${MODE:-matched}"     # quick (10 samples), matched (656 = DDI parity), full
SUBSET="${SUBSET:-}"        # Override subset size (empty = use MODE default)
CONDITION="${CONDITION:-pneumonia}"  # pneumonia or effusion

# Set subset based on mode
if [ -z "$SUBSET" ]; then
    case "$MODE" in
        quick)   SUBSET=10 ;;
        matched) SUBSET=656 ;;
        full)    SUBSET="" ;;  # no subset = full dataset
    esac
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/demographic-calibration}"
VENV_API="${PROJECT_DIR}/venv"
VENV_LOCAL="${VENV_LOCAL:-${PROJECT_DIR}/local_env}"  # shared by R1-Onevision and Qwen2.5-VL
VENV_LLAMA="${VENV_LLAMA:-${PROJECT_DIR}/llama_env}"  # created by scripts/setup_llama_env.sh
SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"

# Select config based on MODEL and CONDITION
case "$MODEL" in
    r1)
        case "$CONDITION" in
            pneumonia) CONFIG="${PROJECT_DIR}/configs/chexpert_r1_experiment.yaml" ;;
            effusion)  CONFIG="${PROJECT_DIR}/configs/chexpert_effusion_r1_experiment.yaml" ;;
            *)
                echo "Unknown condition: $CONDITION (valid: pneumonia, effusion)"
                exit 1
                ;;
        esac
        ;;
    gpt4o)
        case "$CONDITION" in
            pneumonia) CONFIG="${PROJECT_DIR}/configs/chexpert_gpt4o.yaml" ;;
            effusion)  CONFIG="${PROJECT_DIR}/configs/chexpert_effusion_gpt4o.yaml" ;;
            *)
                echo "Unknown condition: $CONDITION (valid: pneumonia, effusion)"
                exit 1
                ;;
        esac
        ;;
    qwen)
        case "$CONDITION" in
            pneumonia) CONFIG="${PROJECT_DIR}/configs/chexpert_qwen_vl.yaml" ;;
            effusion)  CONFIG="${PROJECT_DIR}/configs/chexpert_effusion_qwen_vl.yaml" ;;
            *)
                echo "Unknown condition: $CONDITION (valid: pneumonia, effusion)"
                exit 1
                ;;
        esac
        ;;
    llama)
        case "$CONDITION" in
            pneumonia) CONFIG="${PROJECT_DIR}/configs/chexpert_llama_vision.yaml" ;;
            effusion)  CONFIG="${PROJECT_DIR}/configs/chexpert_effusion_llama_vision.yaml" ;;
            *)
                echo "Unknown condition: $CONDITION (valid: pneumonia, effusion)"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Valid options: r1, gpt4o, qwen, llama"
        exit 1
        ;;
esac

# =============================================================================
# Common Setup
# =============================================================================

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
echo "CheXpert Calibration (${CONDITION})"
echo "Model: ${MODEL} | Mode: ${MODE}"
echo "Config: ${CONFIG}"
[ -n "$SUBSET" ] && echo "Subset: ${SUBSET} samples (stratified)"
[ -z "$SUBSET" ] && echo "Subset: full dataset"
echo "=========================================="

# =============================================================================
# Environment Setup
# =============================================================================

setup_local_env() {
    # R1-Onevision and Qwen2.5-VL share the same Qwen2.5-VL backbone and venv.
    echo "Setting up local (Qwen2.5-VL) environment..."
    ml python/3.9.0
    ml cuda/11.7.1
    ml gcc/14.2.0

    unset PYTORCH_CUDA_ALLOC_CONF

    source "$VENV_LOCAL/bin/activate"

    if ! python -c "import typer" &>/dev/null; then
        echo "Installing CLI dependencies..."
        pip install --quiet typer rich pydantic pyyaml numpy pandas scikit-learn scipy matplotlib seaborn pillow tqdm openai google-genai
    fi

    export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
}

setup_llama_env() {
    echo "Setting up Llama-3.2-Vision environment..."
    ml python/3.9.0
    ml cuda/12.1.1
    ml gcc/14.2.0

    unset PYTORCH_CUDA_ALLOC_CONF

    source "$VENV_LLAMA/bin/activate"
    export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
}

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

# =============================================================================
# Run
# =============================================================================

case "$MODEL" in
    r1|qwen) setup_local_env ;;
    llama)   setup_llama_env ;;
    gpt4o)   setup_gpt4o_env ;;
esac

echo "Python: $(which python) ($(python --version))"
[[ "$MODEL" = "r1" || "$MODEL" = "qwen" || "$MODEL" = "llama" ]] && echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

SUBSET_FLAG=""
[ -n "$SUBSET" ] && SUBSET_FLAG="--subset $SUBSET"

if [ "$MODEL" = "r1" ] || [ "$MODEL" = "qwen" ] || [ "$MODEL" = "llama" ]; then
    python -m democalib.cli run "${CONFIG}" $SUBSET_FLAG
else
    democalib run "${CONFIG}" $SUBSET_FLAG
fi

deactivate

echo ""
echo "CheXpert experiment complete! (model=${MODEL}, condition=${CONDITION}, mode=${MODE})"
