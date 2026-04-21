"""
Checkpoint utilities.

Design goals:
  - Atomic writes (survive SIGTERM mid-save): write to .tmp, fsync, rename.
  - Corruption tolerance on load: try last.pt, fall back to best.pt, fail loud.
  - Full provenance: config, git hash, vocab, RNG states, W&B run ID.
  - Structural config drift detection on resume (see config.check_resume_compatibility).

Checkpoint dict schema:
    {
        # Training state
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ...,
        "scaler_state_dict": ...,
        "epoch": int,                  # NEXT epoch to run
        "global_step": int,
        "best_metric": float | None,
        "best_epoch": int | None,
        "epochs_without_improvement": int,
        "rng_states": {...},

        # Provenance
        "config": dict,                # full merged YAML
        "vocab_state": dict,           # from Vocabulary.state_dict()
        "git_commit": str | None,
        "git_dirty": bool,
        "timestamp": str,
        "torch_version": str,

        # W&B
        "wandb_run_id": str | None,
    }
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Optional

import torch


CHECKPOINT_SCHEMA_VERSION = 1 


def atomic_save(state: dict, target_path: Path) -> None:
    """
    Atomic checkpoint save: write to <target>.tmp, fsync, rename.
    On POSIX, rename is atomic — either old or new exists, never partial.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    # Save to tmp
    torch.save(state, tmp_path)

    # fsync to ensure bytes are on disk before rename
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())

    # Atomic rename
    os.replace(tmp_path, target_path)


def build_checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_metric: Optional[float],
    best_epoch: Optional[int],
    epochs_without_improvement: int,
    rng_states: dict,
    config: dict,
    vocab_state: dict,
    git_commit: Optional[str],
    git_dirty: bool,
    wandb_run_id: Optional[str],
) -> dict:
    return {
        "_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "rng_states": rng_states,
        "config": config,
        "vocab_state": vocab_state,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "wandb_run_id": wandb_run_id,
    }


def load_checkpoint_any(checkpoint_dir: Path, map_location: str = "cpu") -> tuple[dict, Path]:
    """
    Load the most useful checkpoint in `checkpoint_dir`, with corruption fallback.

    Order of attempts:
      1. last.pt
      2. last.pt.tmp  (exists if save was killed mid-rename)
      3. best.pt

    Returns (loaded_state, path_loaded_from).
    Raises FileNotFoundError if none work.
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = [
        checkpoint_dir / "last.pt",
        checkpoint_dir / "last.pt.tmp",
        checkpoint_dir / "best.pt",
    ]

    last_error: Optional[Exception] = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location=map_location, weights_only=False)
            return state, path
        except Exception as e:
            # Don't silently skip; log clearly
            print(f"[WARN] Failed to load {path}: {type(e).__name__}: {e}")
            last_error = e

    if last_error is not None:
        raise RuntimeError(
            f"All checkpoint files in {checkpoint_dir} are corrupt. "
            f"Last error: {last_error}. Refusing to silently restart from scratch."
        )
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def is_better(new: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    """Return True if `new` is a meaningful improvement over `best` in the given mode."""
    if best is None:
        return True
    if mode == "min":
        return new < (best - min_delta)
    if mode == "max":
        return new > (best + min_delta)
    raise ValueError(f"Unknown mode: {mode!r}")