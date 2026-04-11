#!/bin/bash
# ============================================================================
# scripts/setup_pace.sh
# ----------------------------------------------------------------------------
# One-shot environment setup for the dl-cxr-report-gen project on PACE-ICE.
# Run this from a PACE login node after cloning the repo into scratch.
#
# Usage:
#   cd ~/scratch/dl-project
#   git clone git@github.com:vermouthwang/dl-cxr-report-gen.git
#   cd dl-cxr-report-gen
#   bash scripts/setup_pace.sh
#
# Idempotent: safe to re-run. Creates the conda env only if it doesn't exist.
# ============================================================================

set -euo pipefail  # fail fast on any error, unset var, or failed pipe

# --- Config ----------------------------------------------------------------
PROJECT_ROOT="$HOME/scratch/dl-project"
VENV_PATH="$PROJECT_ROOT/venv"
REPO_DIR="$PROJECT_ROOT/dl-cxr-report-gen"
PYTHON_VERSION="3.11"
ANACONDA_MODULE="anaconda3/2022.05.0.1"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"

echo "============================================================"
echo "  dl-cxr-report-gen PACE setup"
echo "============================================================"
echo "Project root : $PROJECT_ROOT"
echo "Venv path    : $VENV_PATH"
echo "Python       : $PYTHON_VERSION"
echo "PyTorch CUDA : cu129 (compatible with PACE cuda/12.9)"
echo ""

# --- 1. Scratch folder structure -------------------------------------------
echo "[1/6] Creating scratch folder structure..."
mkdir -p "$PROJECT_ROOT"/{data,checkpoints,outputs}
mkdir -p "$PROJECT_ROOT"/.cache/{huggingface,torch,wandb,pip}
echo "    OK"

# --- 2. Cache env vars in ~/.bashrc ----------------------------------------
echo "[2/6] Ensuring cache env vars are in ~/.bashrc..."
BASHRC_MARKER="# ---------- DL project: cache redirects to scratch ----------"
if grep -qF "$BASHRC_MARKER" ~/.bashrc 2>/dev/null; then
    echo "    Already present, skipping."
else
    cat >> ~/.bashrc <<'EOF'

# ---------- DL project: cache redirects to scratch ----------
export DL_PROJECT_ROOT=$HOME/scratch/dl-project
export HF_HOME=$DL_PROJECT_ROOT/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$DL_PROJECT_ROOT/.cache/torch
export WANDB_DIR=$DL_PROJECT_ROOT/.cache/wandb
export WANDB_CACHE_DIR=$DL_PROJECT_ROOT/.cache/wandb
export PIP_CACHE_DIR=$DL_PROJECT_ROOT/.cache/pip
EOF
    echo "    Added."
fi
# Source in current shell so subsequent steps see the vars
export DL_PROJECT_ROOT=$HOME/scratch/dl-project
export HF_HOME=$DL_PROJECT_ROOT/.cache/huggingface
export TORCH_HOME=$DL_PROJECT_ROOT/.cache/torch
export WANDB_DIR=$DL_PROJECT_ROOT/.cache/wandb
export PIP_CACHE_DIR=$DL_PROJECT_ROOT/.cache/pip

# --- 3. Load anaconda module -----------------------------------------------
echo "[3/6] Loading anaconda module..."
module load "$ANACONDA_MODULE" 2>/dev/null || {
    echo "    WARN: $ANACONDA_MODULE not available, trying anaconda3/2023.03..."
    module load anaconda3/2023.03
}
echo "    OK"

# --- 4. Create conda env (if missing) --------------------------------------
echo "[4/6] Creating conda env at $VENV_PATH..."
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
    echo "    Env already exists, skipping creation."
else
    conda create --prefix "$VENV_PATH" python="$PYTHON_VERSION" -y
fi

# Activate
source activate "$VENV_PATH" || conda activate "$VENV_PATH"

# Sanity-check that pip is now pointing to scratch
WHICH_PIP=$(which pip3)
if [[ "$WHICH_PIP" != *"$VENV_PATH"* ]]; then
    echo "    ERROR: pip3 is '$WHICH_PIP', not inside $VENV_PATH."
    echo "    Try: conda deactivate && conda activate $VENV_PATH"
    exit 1
fi
echo "    OK  (pip3 -> $WHICH_PIP)"

# --- 5. Install PyTorch (cu129) and project dependencies -------------------
echo "[5/6] Installing Python packages (this takes a few minutes)..."
pip3 install --upgrade pip

if [ -f "$REPO_DIR/requirements.txt" ]; then
    echo "    Found requirements.txt, installing pinned versions..."
    # Install torch first from the cu129 index, then the rest from PyPI
    pip3 install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
    pip3 install -r "$REPO_DIR/requirements.txt"
else
    echo "    No requirements.txt found, installing latest compatible versions..."
    pip3 install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
    pip3 install transformers wandb pycocoevalcap nltk pandas matplotlib \
                 pyyaml tqdm scikit-learn Pillow omegaconf einops sacrebleu
fi
echo "    OK"

# --- 6. Final sanity check -------------------------------------------------
echo "[6/6] Running import sanity check..."
python -c "
import torch, transformers, wandb, pycocoevalcap, nltk, pandas, matplotlib
import yaml, tqdm, sklearn, PIL, omegaconf, einops, sacrebleu
print('    torch:', torch.__version__)
print('    CUDA available (needs GPU node, OK if False on login):', torch.cuda.is_available())
print('    all imports OK')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run 'wandb login' (needs your W&B API key from"
echo "     https://wandb.ai/authorize)"
echo "  2. Request a GPU node to test training:"
echo "     salloc -N1 -t0:15:00 --cpus-per-task=4 --gres=gpu:1 --mem-per-gpu=16G"
echo "  3. Or submit the smoke test: sbatch scripts/test_gpu.sbatch"
echo ""
echo "NOTE: METEOR evaluation (pycocoevalcap) requires Java at runtime."
echo "      Load it before eval jobs with: module load java"
