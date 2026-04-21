"""
Some training utilities: seeding, git info, RNG state management.
"""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch/numpy/random/cuda for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_states() -> dict:
    """Snapshot RNG states for checkpoint serialization."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_states(states: dict) -> None:
    """Restore RNG states from a checkpoint. Tolerant of missing keys."""
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available() and states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "python" in states:
        random.setstate(states["python"])


def git_info(repo_root: Path | str | None = None) -> dict:
    """
    Return current git commit and dirty flag, for provenance.
    Returns {"commit": "<hash>", "dirty": bool} or {"commit": None, ...} on failure.
    """
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"commit": commit, "dirty": bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": False}


def count_parameters(model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_n(n: int) -> str:
    """Human-readable parameter count: 12345678 -> '12.3M'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)