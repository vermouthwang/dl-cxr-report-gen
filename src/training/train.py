"""
!!!!Main training pipeline!

Usage:
    python -m src.training.train --config configs/dummy.yaml
    python -m src.training.train --config configs/lstm.yaml --resume

The --resume flag looks for outputs/<experiment.name>/last.pt and picks up
where it left off, reattaching to the same W&B run.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from src.data.iu_xray import Vocabulary, build_dataloaders
from src.models import get_model
from src.training.checkpoint import load_checkpoint_any
from src.training.config import check_resume_compatibility, load_config
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.training.utils import (
    count_parameters,
    format_n,
    git_info,
    restore_rng_states,
    seed_everything,
)

# Optional: linguistic metrics. wait for @YousufQ7 implementation here
# importable (e.g., syntax error while he's writing it), we shouldn't
# crash the whole training script.
try:
    from src.evaluation.linguistic_metrics import compute_linguistic_metrics
except ImportError as e:
    print(f"[WARN] Could not import linguistic_metrics: {e}. Training will run with val_loss only.")
    compute_linguistic_metrics = None


def parse_args():
    p = argparse.ArgumentParser(description="Train a chest X-ray report generation model.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt in output dir")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override config's output dir. Final path = <output_dir>/<experiment.name>.")
    return p.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("train")

    # Load + validate config (fails loudly on missing/wrong keys)
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Seed
    seed_everything(int(cfg["experiment"]["seed"]))

    # Output dir: <output_dir>/<experiment.name>/
    base_out = Path(args.output_dir or cfg["experiment"]["output_dir"])
    run_dir = base_out / cfg["experiment"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training on CPU will be very slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    data_cfg = cfg["data"]
    bundle = build_dataloaders(
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        max_report_len=int(data_cfg["max_report_len"]),
        min_word_freq=int(data_cfg["min_word_freq"]),
        data_root=data_cfg.get("root"),
    )
    logger.info(
        f"Data: train={len(bundle.train_loader.dataset)} "
        f"val={len(bundle.val_loader.dataset)} "
        f"test={len(bundle.test_loader.dataset)} "
        f"vocab_size={bundle.vocab.size}"
    )

    # Model
    model = get_model(
        name=cfg["model"]["name"],
        vocab_size=bundle.vocab.size,
        config=cfg["model"].get("config", {}),
    ).to(device)
    total_p, trainable_p = count_parameters(model)
    logger.info(f"Model '{cfg['model']['name']}': {format_n(total_p)} params ({format_n(trainable_p)} trainable)")

    # Optimizer + scheduler
    optimizer = build_optimizer(model, cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        scheduler_cfg=cfg["scheduler"],
        steps_per_epoch=len(bundle.train_loader),
        total_epochs=int(cfg["training"]["epochs"]),
    )

    # Resume
    resume_state = None
    resumed_wandb_run_id = None
    if args.resume:
        logger.info(f"Attempting resume from {run_dir}")
        try:
            resume_state, loaded_path = load_checkpoint_any(run_dir, map_location="cpu")
            logger.info(f"Loaded checkpoint from {loaded_path}")
            # Compatibility check — raises on structural drift
            warnings = check_resume_compatibility(resume_state["config"], cfg)
            for w in warnings:
                logger.warning(f"Config drift: {w}")
            # Vocab must match — else embedding sizes mismatch silently
            saved_vocab = Vocabulary.from_state_dict(resume_state["vocab_state"])
            if saved_vocab.size != bundle.vocab.size:
                raise ValueError(
                    f"Vocab size changed on resume: checkpoint={saved_vocab.size}, "
                    f"current={bundle.vocab.size}. Retrain from scratch."
                )
            resumed_wandb_run_id = resume_state.get("wandb_run_id")
        except FileNotFoundError:
            logger.info("No checkpoint found; starting fresh.")
            resume_state = None

    # W&B init
    wandb_run = None
    wandb_run_id = resumed_wandb_run_id
    if cfg["wandb"]["enabled"]:
        import wandb
        # Respect WANDB_MODE env var. If user set offline, wandb.init will honor it.
        # If online mode fails to connect, wandb.init will raise — which is what we want.
        init_kwargs = dict(
            entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            name=cfg["experiment"]["name"],
            config=cfg,
            tags=cfg["wandb"].get("tags", []),
            dir=os.environ.get("WANDB_DIR"),
        )
        if wandb_run_id:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        wandb_run = wandb.init(**init_kwargs)
        wandb_run_id = wandb_run.id
        logger.info(f"W&B run: {wandb_run.url if hasattr(wandb_run, 'url') else wandb_run_id}")
    else:
        logger.info("W&B logging disabled in config.")

    # Git info (for provenance in the checkpoint)
    gi = git_info(Path.cwd())
    if gi["commit"]:
        logger.info(f"Git commit: {gi['commit'][:10]}{' (dirty)' if gi['dirty'] else ''}")

    # Build Trainer
    trainer = Trainer(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
        vocab=bundle.vocab,
        output_dir=run_dir,
        wandb_run=wandb_run,
        wandb_run_id=wandb_run_id,
        git_commit=gi["commit"],
        git_dirty=gi["dirty"],
        compute_linguistic_metrics=compute_linguistic_metrics,
    )

    # If resuming, load state AFTER trainer is constructed (so optimizer/scheduler exist)
    if resume_state is not None:
        trainer.load_state(resume_state)
        restore_rng_states(resume_state["rng_states"])
        logger.info("RNG states restored.")

    # Go
    try:
        trainer.fit()
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()