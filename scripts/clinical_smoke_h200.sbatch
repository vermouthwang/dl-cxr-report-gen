#!/bin/bash
#SBATCH --job-name=clinical_smoke
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

echo "=== [Stage 4a] In-process smoke driver at $(date) ==="
python -u -m scripts.smoke_clinical
echo ""

echo "=== [Stage 4b] 2-epoch training smoke at $(date) ==="
python -u -m src.training.train --config configs/clinical_transformer_smoke.yaml
echo ""

echo "=== Done at $(date) ==="
