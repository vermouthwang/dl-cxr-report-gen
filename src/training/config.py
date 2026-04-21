"""
Usage:
    cfg = load_config("configs/dummy.yaml")

Config inheritance:
    Any YAML can set `_base_: "other.yaml"` at the top level. The base is
    loaded first, then the child config is deep-merged over it. The path
    is resolved relative to the child file's directory.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Schema: required keys and per-key validation
# -----------------------------------------------------------------------------
# Top-level sections. Every config must have all of these (fail-loud principle).
_REQUIRED_SECTIONS = {
    "experiment", "data", "model", "optimizer", "scheduler",
    "training", "validation", "early_stopping", "checkpoint", "wandb",
}

_VALID_OPTIMIZERS = {"adamw"}
_VALID_SCHEDULERS = {"linear_warmup_cosine", "none"}
_VALID_EARLY_STOP_MODES = {"min", "max"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on scalar conflicts."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_with_inheritance(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML file, recursively resolving `_base_` inheritance."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular _base_ inheritance detected at {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_rel = cfg.pop("_base_", None)
    if base_rel is not None:
        base_path = (path.parent / base_rel).resolve()
        base_cfg = _load_with_inheritance(base_path, _seen)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def load_config(path: str | Path) -> dict:
    cfg = _load_with_inheritance(Path(path))
    validate_config(cfg)
    return cfg


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_config(cfg: dict) -> None:
    missing = _REQUIRED_SECTIONS - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required top-level sections: {sorted(missing)}")

    _require(cfg, "experiment.name", str)
    _require(cfg, "experiment.seed", int)
    _require(cfg, "experiment.output_dir", str)

    _require(cfg, "data.batch_size", int)
    _require(cfg, "data.num_workers", int)
    _require(cfg, "data.max_report_len", int)
    _require(cfg, "data.min_word_freq", int)
    # data.root may be null (falls back to env var / default)

    _require(cfg, "model.name", str)
    if not isinstance(cfg["model"].get("config", {}), dict):
        raise ValueError("model.config must be a dict (may be empty)")

    opt_name = _require(cfg, "optimizer.name", str)
    if opt_name not in _VALID_OPTIMIZERS:
        raise ValueError(f"optimizer.name must be one of {sorted(_VALID_OPTIMIZERS)}, got {opt_name!r}")
    _require(cfg, "optimizer.lr", (int, float))
    _require(cfg, "optimizer.weight_decay", (int, float))
    # optimizer.param_groups is optional; validate structure if present
    pgs = cfg["optimizer"].get("param_groups")
    if pgs is not None:
        if not isinstance(pgs, list):
            raise ValueError("optimizer.param_groups must be a list if set")
        for i, pg in enumerate(pgs):
            if not isinstance(pg, dict) or "name" not in pg:
                raise ValueError(f"optimizer.param_groups[{i}] must be a dict with a 'name' key")

    sched_name = _require(cfg, "scheduler.name", str)
    if sched_name not in _VALID_SCHEDULERS:
        raise ValueError(f"scheduler.name must be one of {sorted(_VALID_SCHEDULERS)}, got {sched_name!r}")
    if sched_name == "linear_warmup_cosine":
        w_steps = cfg["scheduler"].get("warmup_steps")
        w_epochs = cfg["scheduler"].get("warmup_epochs")
        if (w_steps is None) == (w_epochs is None):
            raise ValueError(
                "scheduler: set exactly one of warmup_steps or warmup_epochs "
                f"(got warmup_steps={w_steps}, warmup_epochs={w_epochs})"
            )
        _require(cfg, "scheduler.min_lr_ratio", (int, float))

    _require(cfg, "training.epochs", int)
    _require(cfg, "training.mixed_precision", bool)
    _require(cfg, "training.log_every_n_steps", int)
    # grad_clip_norm may be null to disable
    gc = cfg["training"].get("grad_clip_norm")
    if gc is not None and not isinstance(gc, (int, float)):
        raise ValueError("training.grad_clip_norm must be a number or null")

    _require(cfg, "validation.every_n_epochs", int)
    _require(cfg, "validation.generate_samples_every_n_epochs", int)
    _require(cfg, "validation.num_sample_generations", int)
    _require(cfg, "validation.generation_max_length", int)
    _require(cfg, "validation.beam_size", int)

    _require(cfg, "early_stopping.enabled", bool)
    if cfg["early_stopping"]["enabled"]:
        _require(cfg, "early_stopping.metric", str)
        mode = _require(cfg, "early_stopping.mode", str)
        if mode not in _VALID_EARLY_STOP_MODES:
            raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {mode!r}")
        _require(cfg, "early_stopping.patience", int)
        _require(cfg, "early_stopping.min_delta", (int, float))

    _require(cfg, "checkpoint.save_last", bool)
    _require(cfg, "checkpoint.save_best", bool)

    _require(cfg, "wandb.enabled", bool)
    if cfg["wandb"]["enabled"]:
        _require(cfg, "wandb.entity", str)
        _require(cfg, "wandb.project", str)

# helper function for validation
def _require(cfg: dict, dotted_path: str, types) -> Any:
    """Fetch a required nested key, assert its type, return the value."""
    keys = dotted_path.split(".")
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise ValueError(f"Config missing required key: {dotted_path}")
        cur = cur[k]
    if not isinstance(cur, types):
        expected = types if isinstance(types, type) else tuple(t.__name__ for t in types)
        raise ValueError(
            f"Config key {dotted_path} must be of type {expected}, got {type(cur).__name__} ({cur!r})"
        )
    return cur


# -----------------------------------------------------------------------------
# Resume-time config compatibility check
# -----------------------------------------------------------------------------
_STRUCTURAL_KEYS = [
    "model.name",
    "data.min_word_freq",
    "data.max_report_len",
]


def check_resume_compatibility(saved_cfg: dict, current_cfg: dict) -> list[str]:
    """
    Compare a checkpoint's saved config against the current config.

    Returns a list of warning messages for tunable-key differences.
    Raises ValueError if any structural key differs.
    """
    warnings: list[str] = []

    for key in _STRUCTURAL_KEYS:
        old = _safe_get(saved_cfg, key)
        new = _safe_get(current_cfg, key)
        if old != new:
            raise ValueError(
                f"Checkpoint incompatible with current config: "
                f"{key} changed ({old!r} -> {new!r}). Structural keys cannot be changed on resume."
            )

    # model.config sub-dict: any shared key with different value is structural
    saved_mcfg = saved_cfg.get("model", {}).get("config", {}) or {}
    curr_mcfg = current_cfg.get("model", {}).get("config", {}) or {}
    for k in set(saved_mcfg) & set(curr_mcfg):
        if saved_mcfg[k] != curr_mcfg[k]:
            raise ValueError(
                f"Checkpoint incompatible: model.config.{k} changed "
                f"({saved_mcfg[k]!r} -> {curr_mcfg[k]!r})"
            )

    # Tunable keys: warn only
    tunable_checks = [
        "optimizer.lr",
        "optimizer.weight_decay",
        "training.grad_clip_norm",
        "training.mixed_precision",
        "data.batch_size",
    ]
    for key in tunable_checks:
        old = _safe_get(saved_cfg, key)
        new = _safe_get(current_cfg, key)
        if old != new:
            warnings.append(f"{key} changed on resume: {old!r} -> {new!r} (using new value)")

    return warnings


def _safe_get(cfg: dict, dotted: str):
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur