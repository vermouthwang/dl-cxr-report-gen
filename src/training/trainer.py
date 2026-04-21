"""
Trainer: the actual training loop.
"""
from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.checkpoint import (
    atomic_save,
    build_checkpoint_state,
    is_better,
)
from src.training.utils import capture_rng_states

logger = logging.getLogger(__name__)


LinguisticMetricsFn = Callable[[list[str], list[str]], dict[str, float]]


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        vocab,                                            # src.data.iu_xray.Vocabulary
        output_dir: Path,
        wandb_run=None,                                   # a wandb.Run or None
        wandb_run_id: Optional[str] = None,
        git_commit: Optional[str] = None,
        git_dirty: bool = False,
        compute_linguistic_metrics: Optional[LinguisticMetricsFn] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = config
        self.vocab = vocab
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb_run
        self.wandb_run_id = wandb_run_id
        self.git_commit = git_commit
        self.git_dirty = git_dirty
        self.compute_linguistic_metrics = compute_linguistic_metrics

        # AMP
        self.use_amp = bool(config["training"]["mixed_precision"]) and torch.cuda.is_available()
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_without_improvement = 0
        self.stop_requested = False

        # Cache validation config for readability
        vc = config["validation"]
        self.val_every_n = int(vc["every_n_epochs"])
        self.gen_every_n = int(vc["generate_samples_every_n_epochs"])
        self.num_sample_gens = int(vc["num_sample_generations"])
        self.gen_max_len = int(vc["generation_max_length"])
        self.beam_size = int(vc["beam_size"])

        # Early stopping config
        es = config["early_stopping"]
        self.es_enabled = bool(es["enabled"])
        self.es_metric = es.get("metric", "val_loss")
        self.es_mode = es.get("mode", "min")
        self.es_patience = int(es.get("patience", 10))
        self.es_min_delta = float(es.get("min_delta", 0.0))

        # SIGTERM handler
        self._install_signal_handlers()

    # ----- lifecycle -----
    def _install_signal_handlers(self):
        def handler(signum, frame):
            logger.warning(f"Received signal {signum}; will checkpoint and exit after current batch.")
            self.stop_requested = True

        signal.signal(signal.SIGTERM, handler)
        # SIGUSR1: SLURM can be configured to send this before SIGTERM as an early warning.
        try:
            signal.signal(signal.SIGUSR1, handler)
        except (AttributeError, ValueError):
            pass

    # ----- resume -----
    def load_state(self, state: dict) -> None:
        """Restore training state from a checkpoint dict."""
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.epoch = int(state["epoch"])
        self.global_step = int(state["global_step"])
        self.best_metric = state.get("best_metric")
        self.best_epoch = state.get("best_epoch")
        self.epochs_without_improvement = int(state.get("epochs_without_improvement", 0))

        # RNG restore is handled by caller so it can be before dataloader iteration starts.
        logger.info(
            f"Resumed state: epoch={self.epoch}, global_step={self.global_step}, "
            f"best={self.best_metric} at epoch {self.best_epoch}"
        )

    # ----- main loop -----
    def fit(self) -> None:
        total_epochs = int(self.cfg["training"]["epochs"])
        if self.epoch >= total_epochs:
            logger.info(f"Already trained for {self.epoch}/{total_epochs} epochs; nothing to do.")
            return

        logger.info(f"Starting training: epoch {self.epoch} -> {total_epochs}")
        self._sanity_check()

        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self._train_one_epoch(epoch)

            if (epoch + 1) % self.val_every_n == 0 or epoch == total_epochs - 1:
                val_metrics, sample_rows = self._validate(epoch)
            else:
                val_metrics, sample_rows = {}, []

            epoch_time = time.time() - t0
            self._log_epoch(epoch, train_metrics, val_metrics, sample_rows, epoch_time)

            # Checkpoint + early stopping (only on epochs where we validated)
            if val_metrics:
                monitored = val_metrics.get(self.es_metric)
                if monitored is None and self.es_enabled:
                    logger.warning(
                        f"early_stopping.metric={self.es_metric!r} not in val metrics "
                        f"{list(val_metrics.keys())}; skipping early-stopping update."
                    )

                improved = False
                if monitored is not None:
                    improved = is_better(
                        monitored, self.best_metric, self.es_mode, self.es_min_delta
                    )
                    if improved:
                        self.best_metric = float(monitored)
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1

                self._save_checkpoints(save_best=improved)

                if self.es_enabled and self.epochs_without_improvement >= self.es_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}: "
                        f"{self.es_metric} has not improved for {self.es_patience} epochs."
                    )
                    break
            else:
                # Still save `last` even if we didn't validate this epoch
                self._save_checkpoints(save_best=False)

            if self.stop_requested:
                logger.warning(f"Stop requested; exiting after epoch {epoch} checkpoint.")
                break

        # Advance epoch counter so resume starts at the *next* epoch
        # self.epoch = self.epoch + 1
        # self._save_checkpoints(save_best=False)
        # logger.info("Training complete.")
        # self.epoch = self.epoch + 1
        logger.info("Training complete.")

    # ----- sanity check at startup -----
    def _sanity_check(self) -> None:
        """One-batch forward + generate to catch interface violations early."""
        logger.info("Running startup sanity check...")
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            images, input_tokens, target_tokens, lengths, _ = batch
            images = images.to(self.device)
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)
            lengths = lengths.to(self.device)

            with torch.no_grad():
                out = self.model(images, input_tokens, target_tokens, lengths)
                if "loss" not in out or "logits" not in out:
                    raise ValueError(
                        f"Model forward() must return dict with 'loss' and 'logits' keys. "
                        f"Got: {list(out.keys())}"
                    )
                # Test generate
                gen = self.model.generate(images[:1], max_length=self.gen_max_len, beam_size=self.beam_size)
                if not isinstance(gen, list) or not isinstance(gen[0], list):
                    raise ValueError(f"Model generate() must return list[list[int]]. Got: {type(gen)}")
            logger.info("Sanity check passed.")
        except Exception as e:
            raise RuntimeError(f"Model failed startup sanity check: {e}") from e
        finally:
            self.model.train()

    # ----- train one epoch -----
    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        losses: list[float] = []
        grad_norms: list[float] = []
        log_every = int(self.cfg["training"]["log_every_n_steps"])
        grad_clip = self.cfg["training"].get("grad_clip_norm")

        pbar = tqdm(
            self.train_loader,
            desc=f"train ep{epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):
            images, input_tokens, target_tokens, lengths, _ = batch
            images = images.to(self.device, non_blocking=True)
            input_tokens = input_tokens.to(self.device, non_blocking=True)
            target_tokens = target_tokens.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(images, input_tokens, target_tokens, lengths)
                loss = output["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                # Don't recover — dump state and exit so the user investigates.
                crash_path = self.output_dir / "nan_crash.pt"
                torch.save({"step": self.global_step, "epoch": epoch, "loss": float(loss)}, crash_path)
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch}, step {step}. State dumped to {crash_path}."
                )

            self.scaler.scale(loss).backward()

            # Unscale before clipping so clip threshold is in real-LR units
            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                grad_norms.append(float(gn))

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(float(loss.item()))
            self.global_step += 1

            # Progress bar + periodic W&B step logging
            if step % log_every == 0:
                cur_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{losses[-1]:.4f}", lr=f"{cur_lr:.2e}")
                if self.wandb is not None:
                    log_dict = {
                        "train/loss_step": losses[-1],
                        "train/lr": cur_lr,
                        "train/global_step": self.global_step,
                        "epoch": epoch,
                    }
                    if grad_norms:
                        log_dict["train/grad_norm"] = grad_norms[-1]
                    self.wandb.log(log_dict, step=self.global_step)

            if self.stop_requested:
                # Finish the step cleanly then bail out of the epoch
                logger.warning(f"Stop requested mid-epoch at step {step}. Breaking.")
                break

        return {
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
        }

    # ----- validation -----
    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[dict, list]:
        self.model.eval()
        losses: list[float] = []
        hypotheses: list[str] = []
        references: list[str] = []
        sample_rows: list[list[str]] = []

        do_generate = ((epoch + 1) % self.gen_every_n == 0)

        for batch in tqdm(self.val_loader, desc=f"val ep{epoch}", leave=False, dynamic_ncols=True):
            images, input_tokens, target_tokens, lengths, image_ids = batch
            images = images.to(self.device, non_blocking=True)
            input_tokens = input_tokens.to(self.device, non_blocking=True)
            target_tokens = target_tokens.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(images, input_tokens, target_tokens, lengths)
            losses.append(float(output["loss"].item()))

            # Generate only while we still need sample reports
            if do_generate and len(hypotheses) < self.num_sample_gens:
                need = self.num_sample_gens - len(hypotheses)
                sub_images = images[:need]
                gen_ids_batch = self.model.generate(
                    sub_images, max_length=self.gen_max_len, beam_size=self.beam_size
                )
                for j, gen_ids in enumerate(gen_ids_batch):
                    hyp = self.vocab.decode(gen_ids)
                    ref = self.vocab.decode(target_tokens[j, : int(lengths[j].item())].tolist())
                    hypotheses.append(hyp)
                    references.append(ref)
                    sample_rows.append([image_ids[j], ref, hyp])

        metrics = {"val_loss": float(np.mean(losses))}

        # Linguistic metrics hook (stub returns {} for now)
        if hypotheses and self.compute_linguistic_metrics is not None:
            try:
                ling = self.compute_linguistic_metrics(hypotheses, references)
                # Prefix with val_ for clean W&B naming
                metrics.update({f"val_{k}": float(v) for k, v in ling.items()})
            except Exception as e:
                logger.warning(f"Linguistic metrics crashed: {type(e).__name__}: {e}. Continuing.")

        return metrics, sample_rows

    # ----- logging -----
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        sample_rows: list,
        epoch_time: float,
    ) -> None:
        parts = [f"[epoch {epoch}]", f"time={epoch_time:.1f}s"]
        for k, v in train_metrics.items():
            if v is not None:
                parts.append(f"{k}={v:.4f}")
        for k, v in val_metrics.items():
            parts.append(f"{k}={v:.4f}")
        logger.info(" ".join(parts))

        if self.wandb is not None:
            log_dict = {f"epoch": epoch, "epoch_time_sec": epoch_time}
            log_dict.update({k: v for k, v in train_metrics.items() if v is not None})
            log_dict.update(val_metrics)
            self.wandb.log(log_dict, step=self.global_step)

            if sample_rows:
                import wandb
                table = wandb.Table(
                    columns=["image_id", "reference", "hypothesis"],
                    data=sample_rows,
                )
                self.wandb.log({"samples/val_generations": table}, step=self.global_step)

    # ----- checkpointing -----
    def _save_checkpoints(self, save_best: bool) -> None:
        state = build_checkpoint_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            epoch=self.epoch + 1,    # NEXT epoch to run
            global_step=self.global_step,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            epochs_without_improvement=self.epochs_without_improvement,
            rng_states=capture_rng_states(),
            config=self.cfg,
            vocab_state=self.vocab.state_dict(),
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            wandb_run_id=self.wandb_run_id,
        )
        if self.cfg["checkpoint"]["save_last"]:
            atomic_save(state, self.output_dir / "last.pt")
            logger.info(f"Saved last.pt (epoch {self.epoch}, step {self.global_step})")
        if save_best and self.cfg["checkpoint"]["save_best"]:
            atomic_save(state, self.output_dir / "best.pt")
            logger.info(
                f"Saved best.pt (epoch {self.epoch}, {self.es_metric}={self.best_metric:.4f})"
            )