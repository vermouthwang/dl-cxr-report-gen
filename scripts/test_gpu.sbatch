#!/bin/bash
# ============================================================================
# scripts/test_gpu.sbatch
# ----------------------------------------------------------------------------
# Minimal SLURM batch smoke test for the dl-cxr-report-gen project.
# Verifies that a submitted job can: (a) activate the conda env,
# (b) load CUDA, (c) import torch, and (d) see a GPU.
#
# Submit with:
#   sbatch scripts/test_gpu.sbatch
#
# Check status with:
#   squeue -u $USER
#
# Output lands in:
#   ~/scratch/dl-project/outputs/slurm-<jobid>.out
# ============================================================================

#SBATCH --job-name=dl-cxr-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/hice1/%u/scratch/dl-project/outputs/slurm-%j.out
#SBATCH --error=/home/hice1/%u/scratch/dl-project/outputs/slurm-%j.err

set -euo pipefail

echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "Working dir   : $(pwd)"
echo "=========================================="

# Load modules
module load anaconda3/2022.05.0.1 2>/dev/null || module load anaconda3/2023.03
module load cuda

# Activate conda env
source activate ~/scratch/dl-project/venv || conda activate ~/scratch/dl-project/venv

echo ""
echo "--- Environment ---"
which python
which pip3
python --version
nvcc --version | tail -2

echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo ""
echo "--- PyTorch CUDA check ---"
python -c "
import torch
print('torch version     :', torch.__version__)
print('compiled CUDA     :', torch.version.cuda)
print('cuda available    :', torch.cuda.is_available())
print('device count      :', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name       :', torch.cuda.get_device_name(0))
    # Actually put a tensor on the GPU to prove it works end-to-end
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print('matmul on GPU     : OK, result shape', tuple(y.shape))
    print('result mean       :', y.mean().item())
"

echo ""
echo "=========================================="
echo "End time      : $(date)"
echo "Smoke test passed"
echo "=========================================="
