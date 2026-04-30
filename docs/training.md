# Training and Evaluation

How to train each model and produce final test-set numbers.

For interface contracts (model signatures, batch format, etc.), see `interfaces.md`.
For one-time setup, see `setup_iu_xray.md` and `setup_chexbert.md`.

---

## Models and configs

| Model | Training config | Smoke config | Output dir |
|---|---|---|---|
| Hierarchical LSTM | `configs/lstm.yaml` | `configs/lstm_smoke.yaml` | `outputs/lstm_hierarchical/` |
| Vanilla Transformer | `configs/transformer_unfrozen.yaml` | `configs/transformer_smoke.yaml` | `outputs/transformer_unfrozen/` |
| Clinical-term-guided Transformer | `configs/clinical_transformer.yaml` | `configs/clinical_transformer_smoke.yaml` | `outputs/clinical_transformer/` |

Smoke configs train for 2 epochs with small batch size for fast pipeline verification (~3 min on a recent NVIDIA GPU). Production configs train for 30 epochs with full batch size.

---

## Prerequisites

Per-environment, only once:
- Conda env with PyTorch 2.x, CUDA, all deps. Confirm with:
```bash
  python -c "import torch, tqdm, wandb, yaml, pandas, transformers, pycocoevalcap, sklearn, statsmodels; print('OK')"
```
- Dataset present. Dataloader looks for `IU_XRAY_ROOT` env var, then a built-in default. Set it explicitly:
```bash
  export IU_XRAY_ROOT=/path/to/iu-xray/iu_xray
  python -m src.data.iu_xray   # should print: vocab size 975, batch shapes
```
- Java available on `PATH` (required for METEOR/PTBTokenizer):
```bash
  java -version   # any version 8+
```
- DenseNet-121 weights cached (only needed for LSTM, harmless to do anyway):
```bash
  python scripts/cache_densenet_weights.py
```
- W&B credentials in `~/.netrc`:
```bash
  wandb login   # paste API key from https://wandb.ai/authorize
```
- For final clinical evaluation only: CheXbert checkpoint at `external/CheXbert/weights/chexbert.pth`. See `setup_chexbert.md`.

---

## Quick start

Verify everything is wired correctly with the dummy model:

```bash
python -m src.training.train --config configs/dummy.yaml
```

Should complete 2 epochs in under a minute on GPU. If this fails, fix it before trying real models.

---

## Training a model

### Standard invocation

```bash
python -m src.training.train --config configs/lstm.yaml
```

What happens:
- Reads YAML, validates schema (fails loud on missing keys).
- Builds dataloaders, instantiates model via factory, sets up AMP/scheduler/optimizer.
- Initializes a W&B run (entity `yinghou-georgia-institute-of-technology`, project `dl-cxr-report-gen`).
- Runs a startup sanity check (one batch through forward + generate).
- Trains for `training.epochs` epochs.
- Saves `last.pt` after every epoch and `best.pt` whenever the early-stopping metric improves.

Outputs:
- `outputs/<experiment.name>/last.pt` — checkpoint at end of last completed epoch.
- `outputs/<experiment.name>/best.pt` — best checkpoint by `early_stopping.metric`.
- W&B run with per-step train loss, per-epoch val loss + linguistic metrics, sample generations.

### Resuming after interruption

If training is killed (Ctrl+C, walltime, OOM, machine reboot), resume from the last checkpoint:

```bash
python -m src.training.train --config configs/lstm.yaml --resume
```

This loads `outputs/<experiment.name>/last.pt`, restores model/optimizer/scheduler/RNG state, and reattaches to the same W&B run (no duplicate runs).

The resume-time config check enforces structural compatibility: changing `model.name`, `data.min_word_freq`, `data.max_report_len`, or any `model.config.*` value will hard-error. Tunable values (lr, batch_size, grad_clip_norm) print a warning and use the new value.

### Smoke testing changes

Before kicking off a 30-epoch run, validate end-to-end:

```bash
python -m src.training.train --config configs/lstm_smoke.yaml
```

Same model, 2 epochs, smaller batch. Confirms training, validation, linguistic metrics, checkpointing, and W&B logging all work without committing to a long run.

---

## Evaluating a trained model

After training, produce final test-set numbers (linguistic + clinical):

```bash
python -m src.evaluation.evaluate --checkpoint outputs/lstm_hierarchical/best.pt --split test
```

What happens:
- Loads the checkpoint, reconstructs model + vocab from saved state.
- Verifies vocab consistency (hard error if the live tokenizer differs from the checkpoint's).
- Runs greedy generation over the chosen split (`test` by default, 1180 samples; `--split val` for 592).
- Computes BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr (linguistic) and CheXbert F1 micro/macro/per-condition (clinical).
- Creates a new W&B run named `eval-<experiment_name>-<split>-<timestamp>` (does NOT reattach to the training run).

Outputs to `outputs/<experiment.name>/eval_<split>_<timestamp>/`:
- `metrics.json` — all metric values plus run metadata (config, git commit, timing).
- `samples.csv` — every (image_id, reference, hypothesis) triple. Useful for qualitative inspection.
- `summary.txt` — human-readable.

Stdout also prints a summary table.

### Skipping clinical metrics

If CheXbert is not set up (or you only need the cheap path):

```bash
python -m src.evaluation.evaluate --checkpoint outputs/lstm_hierarchical/best.pt --split test --no-clinical
```

Useful for quick iterations during development. CheXbert F1 should still be computed for the final paper numbers.

### Beam search

Greedy decoding is the default. Larger beam sizes typically improve BLEU by a few hundredths absolute, at proportional generation-time cost:

```bash
python -m src.evaluation.evaluate --checkpoint outputs/X/best.pt --split test --beam-size 3
```

Beam search is supported only on models that implement it. Models that don't (LSTM, dummy) accept `beam_size > 1` but silently fall back to greedy.

---

## Reproducing the project's three benchmarks

Train and evaluate all three in sequence. Total ~1.5 hours on RTX 4090, longer on H100/A100 due to lower clock speeds at larger batch sizes (the workload is small).

```bash
# 1. Train
python -m src.training.train --config configs/lstm.yaml
python -m src.training.train --config configs/transformer_unfrozen.yaml
python -m src.training.train --config configs/clinical_transformer.yaml

# 2. Evaluate on test set
python -m src.evaluation.evaluate --checkpoint outputs/lstm_hierarchical/best.pt --split test
python -m src.evaluation.evaluate --checkpoint outputs/transformer_unfrozen/best.pt --split test
python -m src.evaluation.evaluate --checkpoint outputs/clinical_transformer/best.pt --split test
```

Each evaluation produces a timestamped `eval_test_*/metrics.json` alongside its checkpoint.
