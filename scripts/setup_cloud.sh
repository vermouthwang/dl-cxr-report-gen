#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${REPO_NAME:-dl-cxr-report-gen}"
BRANCH="${BRANCH:-main}"
PROJECT_DIR="${PROJECT_DIR:-/workspace/project}"
CONDA_DIR="/workspace/miniconda3"
ENV_NAME="dl-cxr"
DATA_DIR="${PROJECT_DIR}/data/iu-xray"
DATASET_GDRIVE_ID="1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg"  # R2Gen split

echo "==> Pre-flight checks"
: "${GITHUB_USER:?GITHUB_USER must be set}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. This is not a GPU host."
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed to initialize. The GPU is not accessible from this pod."
    echo "This usually means RunPod's host migration broke GPU passthrough."
    echo "Recommended action: terminate this pod and deploy a fresh one."
    echo
    echo "Output of nvidia-smi:"
    nvidia-smi || true
    exit 1
fi

echo "  GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
echo

# ---- 1. Apt packages ---------------------------------------------------------
echo "==> [1/7] Installing apt packages (tmux, unzip, wget, git)"
apt-get update -qq
apt-get install -y -qq tmux unzip wget git
echo

# ---- 2. Project dirs ---------------------------------------------------------
echo "==> [2/7] Creating project directory tree"
mkdir -p "${PROJECT_DIR}/.cache"/{huggingface,torch,wandb,pip}
mkdir -p "${DATA_DIR}"
echo

# ---- 3. Miniconda + dl-cxr env -----------------------------------------------
echo "==> [3/7] Installing miniconda + creating ${ENV_NAME} env"
if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    cd /workspace
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "${CONDA_DIR}"
    rm miniconda.sh
    "${CONDA_DIR}/bin/conda" init bash
else
    echo "  miniconda already installed at ${CONDA_DIR}, skipping"
fi

# Accept Anaconda ToS (required by conda 26.x) — idempotent
"${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! "${CONDA_DIR}/bin/conda" env list | grep -q "^${ENV_NAME} "; then
    "${CONDA_DIR}/bin/conda" create -n "${ENV_NAME}" python=3.11 -y
else
    echo "  conda env ${ENV_NAME} already exists, skipping create"
fi

# Activate env for the rest of this script
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "  Activated: $(which python) ($(python --version))"
echo

# ---- 4. Python packages ------------------------------------------------------
echo "==> [4/7] Installing Python packages"
pip install --quiet --upgrade pip

# PyTorch with CUDA 12.8 wheels (matches RunPod's typical driver)
if ! python -c "import torch" &>/dev/null; then
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "  torch already installed, skipping"
fi

pip install --quiet transformers wandb pycocoevalcap nltk pandas matplotlib pyyaml tqdm gdown

# Verify torch sees the GPU before we go further
python -c "
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see CUDA. Aborting.'
print(f'  torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')
"
echo

# ---- 5. Persist env vars in .bashrc ------------------------------------------
echo "==> [5/7] Persisting env vars in ~/.bashrc"
BASHRC_MARKER="# >>> dl-cxr cloud setup >>>"
if ! grep -q "${BASHRC_MARKER}" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

${BASHRC_MARKER}
export HF_HOME=${PROJECT_DIR}/.cache/huggingface
export TORCH_HOME=${PROJECT_DIR}/.cache/torch
export WANDB_DIR=${PROJECT_DIR}/.cache/wandb
export PIP_CACHE_DIR=${PROJECT_DIR}/.cache/pip
export IU_XRAY_ROOT=${DATA_DIR}/iu_xray

if [ -f ${CONDA_DIR}/etc/profile.d/conda.sh ]; then
    source ${CONDA_DIR}/etc/profile.d/conda.sh
    conda activate ${ENV_NAME} 2>/dev/null || true
fi
# <<< dl-cxr cloud setup <<<
EOF
    echo "  added bashrc block"
else
    echo "  bashrc already configured, skipping"
fi

# Set for current shell too
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export WANDB_DIR="${PROJECT_DIR}/.cache/wandb"
export PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"
export IU_XRAY_ROOT="${DATA_DIR}/iu_xray"
echo

# ---- 6. Clone repo + checkout branch -----------------------------------------
echo "==> [6/7] Cloning repo (branch: ${BRANCH})"
REPO_DIR="${PROJECT_DIR}/${REPO_NAME}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    cd "${PROJECT_DIR}"
    # Embed token only for the clone, then strip it
    git clone --branch "${BRANCH}" \
        "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
    cd "${REPO_DIR}"
    git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

    # Configure credential helper so future push/pull prompts cache the token
    git config credential.helper 'cache --timeout=86400'
else
    echo "  ${REPO_DIR} already exists, fetching + checking out ${BRANCH}"
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull --ff-only
fi
echo

# ---- 7. W&B login + dataset --------------------------------------------------
echo "==> [7/7] W&B login + dataset"

# wandb login is idempotent — re-running just refreshes the netrc entry
wandb login --relogin "${WANDB_API_KEY}" >/dev/null

# Verify entity is correct
python -c "
import wandb
api = wandb.Api()
print(f'  wandb username: {api.viewer.username}')
print(f'  default entity: {api.default_entity}')
"

# Dataset download (skip if already extracted)
if [[ -f "${DATA_DIR}/iu_xray/annotation.json" ]]; then
    echo "  dataset already present at ${DATA_DIR}/iu_xray, skipping download"
else
    cd "${DATA_DIR}"
    if [[ ! -f iu_xray.zip ]]; then
        echo "  downloading IU X-Ray (~1.1 GB) from Google Drive..."
        gdown "${DATASET_GDRIVE_ID}"
    fi
    echo "  extracting iu_xray.zip..."
    unzip -q iu_xray.zip
fi

# Authoritative dataset verification (your README's check)
echo
echo "==> Running dataset inspection script"
cd "${PROJECT_DIR}"
if [[ -f "${REPO_DIR}/scripts/inspect_iu_xray.py" ]]; then
    python "${REPO_DIR}/scripts/inspect_iu_xray.py"
else
    echo "  inspect_iu_xray.py not found on this branch, skipping"
fi

# ---- Done --------------------------------------------------------------------
cat << EOF

================================================================================
✅ Setup complete.

To start training, run:
    cd ${REPO_DIR}
    python -m src.training.train --config configs/lstm_smoke.yaml

If this is a new SSH session, log out and back in first so .bashrc applies.
================================================================================
EOF
