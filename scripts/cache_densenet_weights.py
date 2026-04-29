"""
Pre-cache pretrained DenseNet-121 weights before running training jobs.

Run this ONCE on a PACE node that has internet access (the login node works).
After this, compute-node training runs will find the weights in $TORCH_HOME
and won't need to download.

Usage:
    python scripts/cache_densenet_weights.py
"""
import os
from pathlib import Path

import torch
import torchvision.models as models


def main():
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        print("[WARN] TORCH_HOME env var not set; using default ~/.cache/torch")
    else:
        print(f"TORCH_HOME = {torch_home}")
        Path(torch_home).mkdir(parents=True, exist_ok=True)

    print("Downloading DenseNet-121 ImageNet weights...")
    _ = models.densenet121(pretrained=True)
    print("Done. Weights are cached and ready for offline use on compute nodes.")


if __name__ == "__main__":
    main()
