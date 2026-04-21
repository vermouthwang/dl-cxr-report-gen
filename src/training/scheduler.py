"""
Learning rate scheduler builder.
Called as: scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, total_epochs)
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: dict,
    steps_per_epoch: int,
    total_epochs: int,
) -> LambdaLR:
    name = scheduler_cfg["name"]
    total_steps = steps_per_epoch * total_epochs

    if name == "none":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if name == "linear_warmup_cosine":
        # Resolve warmup_steps from either warmup_steps or warmup_epochs
        w_steps = scheduler_cfg.get("warmup_steps")
        w_epochs = scheduler_cfg.get("warmup_epochs")
        if w_steps is not None:
            warmup_steps = int(w_steps)
        else:
            warmup_steps = int(w_epochs) * steps_per_epoch
        min_lr_ratio = float(scheduler_cfg["min_lr_ratio"])

        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total training steps ({total_steps})"
            )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup from 0 -> 1
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay from 1 -> min_lr_ratio over remaining steps
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler name: {name!r}")


def build_optimizer(model: torch.nn.Module, cfg: dict) -> Optimizer:
    """
    Build AdamW optionally with param_groups for per-prefix LR/weight_decay overrides.

    Example config:
        optimizer:
          name: adamw
          lr: 1.0e-4
          weight_decay: 0.01
          betas: [0.9, 0.999]
          param_groups:
            - name: "encoder"      # params whose name starts with "encoder." get these overrides
              lr: 5.0e-5
    """
    if cfg["name"] != "adamw":
        raise ValueError(f"Unknown optimizer: {cfg['name']!r}")

    base_lr = float(cfg["lr"])
    base_wd = float(cfg["weight_decay"])
    betas = tuple(cfg.get("betas", [0.9, 0.999]))

    param_groups_cfg = cfg.get("param_groups") or []

    if not param_groups_cfg:
        # Simple single-group case
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=base_wd,
            betas=betas,
        )

    # Bucket params by the first matching prefix (longest match wins for determinism)
    prefixes = sorted(
        [(pg["name"], pg) for pg in param_groups_cfg],
        key=lambda x: -len(x[0]),
    )
    buckets: dict[str, list] = {pg["name"]: [] for pg in param_groups_cfg}
    default_bucket: list = []

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        placed = False
        for prefix, _ in prefixes:
            if pname.startswith(prefix):
                buckets[prefix].append(p)
                placed = True
                break
        if not placed:
            default_bucket.append(p)

    groups = []
    for pg in param_groups_cfg:
        name = pg["name"]
        if not buckets[name]:
            raise ValueError(
                f"optimizer.param_groups[{name!r}] matched zero parameters. "
                f"Check that your model has parameters whose names start with {name!r}."
            )
        groups.append({
            "params": buckets[name],
            "lr": float(pg.get("lr", base_lr)),
            "weight_decay": float(pg.get("weight_decay", base_wd)),
            "_name": name,
        })
    if default_bucket:
        groups.append({
            "params": default_bucket,
            "lr": base_lr,
            "weight_decay": base_wd,
            "_name": "default",
        })

    return torch.optim.AdamW(groups, betas=betas)