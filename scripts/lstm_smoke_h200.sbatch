#!/bin/bash
#SBATCH --job-name=lstm_smoke
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -e
module load anaconda3/2022.05.0.1
source activate ~/scratch/dl-project/venv
cd ~/scratch/dl-project/dl-cxr-report-gen

echo "=== Node info ==="
hostname
nvidia-smi
echo ""

echo "=== Starting smoke test at $(date) ==="
python -u -m src.training.train --config configs/lstm_smoke.yaml
echo ""
echo "=== Done at $(date) ==="
