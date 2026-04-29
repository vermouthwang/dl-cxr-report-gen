#!/usr/bin/env bash
# run_smoke_transformerbase.sh
#
# Vanilla Transformer integration test on a non-SLURM GPU box (e.g. RunPod).
# Runs three stages back-to-back:
#   [1/3] Sanity test (CPU, no data, ~30s)        -- factory + forward + backward + generate
#   [2/3] Smoke training (2 epochs)               -- configs/transformer_smoke.yaml
#   [3/3] Full training (30 epochs)               -- configs/transformer.yaml
#
# Usage (from repo root):
#   bash scripts/run_smoke_transformerbase.sh
#
# Stop early at any time with Ctrl-C; checkpoints from completed phases are
# preserved under outputs/<experiment-name>/{last,best}.pt.
#
# Assumptions (RunPod):
#   - Repo + IU X-ray data already present
#   - IU_XRAY_ROOT env var is set (or a default path is configured)
#   - WANDB_API_KEY is set (or you've already `wandb login`-ed)
#   - `python` resolves to a torch-enabled interpreter

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "Vanilla Transformer smoke runner"
echo "Repo:    $REPO_ROOT"
echo "Python:  $($PYTHON --version 2>&1)  ($(which $PYTHON))"
echo "Started: $(date)"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "=== GPU info ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# -------------------------------------------------------------------- #
echo ""
echo "=== [1/3] Sanity test (no data) ==="
$PYTHON scripts/sanity_test_transformer.py
echo "Sanity OK."

# -------------------------------------------------------------------- #
echo ""
echo "=== [2/3] Smoke training: 2 epochs (configs/transformer_smoke.yaml) ==="
echo "Started: $(date)"
$PYTHON -u -m src.training.train --config configs/transformer_smoke.yaml
echo "Smoke finished: $(date)"

# -------------------------------------------------------------------- #
echo ""
echo "=== [3/3] Full training: 30 epochs (configs/transformer.yaml) ==="
echo "Started: $(date)"
echo "Tip: stop early with Ctrl-C; the last/best checkpoint is preserved."
$PYTHON -u -m src.training.train --config configs/transformer.yaml
echo "Full training finished: $(date)"

echo ""
echo "============================================================"
echo "All stages complete."
echo "Outputs: $REPO_ROOT/outputs/"
echo "============================================================"
