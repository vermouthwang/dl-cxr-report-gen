"""
Final test-set evaluation: load a trained checkpoint, generate reports for the
chosen split, and compute linguistic + clinical metrics.

Designed for end-of-training paper numbers, NOT in-loop validation. CheXbert is
too slow (BERT inference per report) for every val epoch but fine for one final
test-set pass per model.

Usage:
    python -m src.evaluation.evaluate --checkpoint outputs/lstm_hierarchical/best.pt
    python -m src.evaluation.evaluate --checkpoint outputs/transformer_unfrozen/best.pt --split val
    python -m src.evaluation.evaluate --checkpoint outputs/clinical_transformer/best.pt --no-clinical
    python -m src.evaluation.evaluate --checkpoint .../best.pt --beam-size 3 --batch-size 16

Outputs to <checkpoint_dir>/eval_<split>_<timestamp>/:
  - metrics.json     all metric values, plus run metadata (config, git, timing)
  - samples.csv      every (image_id, reference, hypothesis) triple from the split
  - summary.txt      human-readable summary table
A new W&B run is created (not reattaching to training run) for evaluation visibility.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.data.iu_xray import Vocabulary, build_dataloaders
from src.models import get_model
from src.training.utils import format_n, count_parameters

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained chest X-ray report generation checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a .pt checkpoint (typically best.pt from a training run).")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Which split to evaluate on (default: test).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override checkpoint's batch size (useful if eval OOMs).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-length", type=int, default=None,
                   help="Override generation max_length (default: from checkpoint config).")
    p.add_argument("--beam-size", type=int, default=1,
                   help="Greedy decoding (1) or beam search (>1).")
    p.add_argument("--no-linguistic", action="store_true",
                   help="Skip linguistic metrics (BLEU/METEOR/ROUGE/CIDEr).")
    p.add_argument("--no-clinical", action="store_true",
                   help="Skip clinical metrics (CheXbert F1).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Skip W&B logging.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory (default: alongside checkpoint).")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override IU X-Ray data root (default: from checkpoint config or env var).")
    return p.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


@torch.no_grad()
def run_generation(model, loader, vocab, device, max_length, beam_size):
    model.eval()
    predictions = {}
    references = {}
    for batch in tqdm(loader, desc="generating", dynamic_ncols=True):
        images, _, target_tokens, lengths, image_ids = batch
        images = images.to(device, non_blocking=True)
        gen_ids_batch = model.generate(images, max_length=max_length, beam_size=beam_size)
        for j, gen_ids in enumerate(gen_ids_batch):
            img_id = image_ids[j]
            hyp = vocab.decode(gen_ids)
            ref = vocab.decode(target_tokens[j, : int(lengths[j].item())].tolist())
            predictions[img_id] = hyp
            references[img_id] = ref
    return predictions, references


def compute_metrics_safe(predictions, references, do_linguistic, do_clinical):
    metrics = {}
    if do_linguistic:
        logger.info("Computing linguistic metrics (BLEU/METEOR/ROUGE/CIDEr)...")
        try:
            from src.evaluation.linguistic_metrics import compute_linguistic_metrics
            t0 = time.time()
            ling = compute_linguistic_metrics(predictions, references)
            logger.info(f"Linguistic metrics computed in {time.time() - t0:.1f}s")
            metrics.update({k: float(v) for k, v in ling.items()})
        except Exception as e:
            logger.error(f"Linguistic metrics FAILED: {type(e).__name__}: {e}")
            metrics["linguistic_error"] = str(e)
    else:
        logger.info("Skipping linguistic metrics (--no-linguistic)")
    if do_clinical:
        logger.info("Computing clinical metrics (CheXbert F1)... (this is slow)")
        try:
            from src.evaluation.clinical_metrics import compute_clinical_metrics
            t0 = time.time()
            clinical = compute_clinical_metrics(predictions, references)
            logger.info(f"Clinical metrics computed in {time.time() - t0:.1f}s")
            metrics.update({k: float(v) for k, v in clinical.items()})
        except Exception as e:
            logger.error(f"Clinical metrics FAILED: {type(e).__name__}: {e}")
            metrics["clinical_error"] = str(e)
    else:
        logger.info("Skipping clinical metrics (--no-clinical)")
    return metrics


def write_outputs(output_dir, metrics, predictions, references, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "metadata": metadata}, f, indent=2, default=str)
    logger.info(f"Wrote {metrics_path}")
    samples_path = output_dir / "samples.csv"
    with open(samples_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "reference", "hypothesis"])
        for img_id in sorted(predictions.keys()):
            writer.writerow([img_id, references[img_id], predictions[img_id]])
    logger.info(f"Wrote {samples_path} ({len(predictions)} rows)")
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation summary\n==================\n\n")
        f.write(f"Checkpoint:    {metadata.get('checkpoint_path')}\n")
        f.write(f"Model:         {metadata.get('model_name')}\n")
        f.write(f"Split:         {metadata.get('split')}\n")
        f.write(f"Num samples:   {metadata.get('num_samples')}\n")
        f.write(f"Beam size:     {metadata.get('beam_size')}\n")
        f.write(f"Generation:    {metadata.get('generation_time_sec'):.1f}s\n")
        f.write(f"Total wall:    {metadata.get('total_time_sec'):.1f}s\n\n")
        f.write(f"Metrics:\n--------\n")
        if not metrics:
            f.write("(none)\n")
        else:
            for k in sorted(metrics.keys()):
                v = metrics[k]
                if isinstance(v, float):
                    f.write(f"  {k:35s} = {v:.4f}\n")
                else:
                    f.write(f"  {k:35s} = {v}\n")
        f.write("\n")
    logger.info(f"Wrote {summary_path}")


def print_summary(metrics, metadata):
    print()
    print("=" * 60)
    print(f"Eval results: {metadata.get('model_name')} on {metadata.get('split')} ({metadata.get('num_samples')} samples)")
    print("=" * 60)
    if not metrics:
        print("  (no metrics computed)")
    else:
        ling_keys = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]
        clinical_main = ["chexbert_f1_micro", "chexbert_f1_macro"]
        per_finding = sorted(k for k in metrics if k.startswith("chexbert_f1_") and k not in clinical_main)
        for k in ling_keys:
            if k in metrics:
                print(f"  {k:35s} = {metrics[k]:.4f}")
        print()
        for k in clinical_main:
            if k in metrics:
                print(f"  {k:35s} = {metrics[k]:.4f}")
        if per_finding:
            print()
            for k in per_finding:
                print(f"  {k:35s} = {metrics[k]:.4f}")
        for k in sorted(metrics):
            if k.endswith("_error"):
                print(f"  WARN  {k}: {metrics[k]}")
    print("=" * 60)
    print()


def main():
    args = parse_args()
    setup_logging()
    t_start = time.time()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    logger.info(f"Checkpoint config: model={cfg['model']['name']}, "
                f"trained_to_epoch={state['epoch']}, best_metric={state.get('best_metric')}")

    saved_vocab = Vocabulary.from_state_dict(state["vocab_state"])
    saved_vocab_size = saved_vocab.size
    logger.info(f"Checkpoint vocab size: {saved_vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available - eval will be slow")
    logger.info(f"Device: {device}")

    data_cfg = cfg["data"]
    batch_size = args.batch_size or int(data_cfg["batch_size"])
    bundle = build_dataloaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        max_report_len=int(data_cfg["max_report_len"]),
        min_word_freq=int(data_cfg["min_word_freq"]),
        data_root=args.data_root or data_cfg.get("root"),
        shuffle_train=False,
    )

    if bundle.vocab.size != saved_vocab_size:
        raise RuntimeError(
            f"Vocab size mismatch: checkpoint={saved_vocab_size}, freshly-built={bundle.vocab.size}. "
            f"Tokenizer/data has changed since training."
        )

    loader = bundle.val_loader if args.split == "val" else bundle.test_loader
    n_samples = len(loader.dataset)
    logger.info(f"Evaluating on {args.split} split: {n_samples} samples, batch_size={batch_size}")

    model = get_model(
        name=cfg["model"]["name"],
        vocab_size=saved_vocab_size,
        config=cfg["model"].get("config", {}),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    total_p, _ = count_parameters(model)
    logger.info(f"Model: {cfg['model']['name']}, {format_n(total_p)} params")

    max_length = args.max_length or int(cfg["validation"]["generation_max_length"])
    logger.info(f"Generating: max_length={max_length}, beam_size={args.beam_size}")
    t_gen = time.time()
    predictions, references = run_generation(model, loader, bundle.vocab, device, max_length, args.beam_size)
    gen_time = time.time() - t_gen
    logger.info(f"Generated {len(predictions)} reports in {gen_time:.1f}s ({gen_time / len(predictions) * 1000:.1f}ms/sample)")

    metrics = compute_metrics_safe(
        predictions=predictions,
        references=references,
        do_linguistic=not args.no_linguistic,
        do_clinical=not args.no_clinical,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / f"eval_{args.split}_{timestamp}"

    metadata = {
        "checkpoint_path": str(ckpt_path),
        "model_name": cfg["model"]["name"],
        "experiment_name": cfg["experiment"].get("name"),
        "split": args.split,
        "num_samples": n_samples,
        "batch_size": batch_size,
        "beam_size": args.beam_size,
        "max_length": max_length,
        "generation_time_sec": gen_time,
        "total_time_sec": time.time() - t_start,
        "trained_to_epoch": state["epoch"],
        "best_metric_at_train_time": state.get("best_metric"),
        "git_commit_at_train_time": state.get("git_commit"),
        "git_dirty_at_train_time": state.get("git_dirty"),
        "torch_version": torch.__version__,
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    write_outputs(output_dir, metrics, predictions, references, metadata)
    print_summary(metrics, metadata)

    if not args.no_wandb and cfg["wandb"]["enabled"]:
        try:
            import wandb
            run = wandb.init(
                entity=cfg["wandb"]["entity"],
                project=cfg["wandb"]["project"],
                name=f"eval-{cfg['experiment']['name']}-{args.split}-{timestamp}",
                config={**cfg, "evaluation": metadata},
                tags=cfg["wandb"].get("tags", []) + ["evaluation", args.split],
                dir=os.environ.get("WANDB_DIR"),
            )
            wandb.log({f"{args.split}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            sample_rows = [
                [img_id, references[img_id], predictions[img_id]]
                for img_id in sorted(predictions.keys())[:20]
            ]
            table = wandb.Table(columns=["image_id", "reference", "hypothesis"], data=sample_rows)
            wandb.log({f"samples/{args.split}_first20": table})
            logger.info(f"W&B run: {run.url}")
            run.finish()
        except Exception as e:
            logger.warning(f"W&B logging failed (continuing): {type(e).__name__}: {e}")
    else:
        logger.info("W&B logging disabled")

    logger.info(f"Total wall time: {time.time() - t_start:.1f}s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
