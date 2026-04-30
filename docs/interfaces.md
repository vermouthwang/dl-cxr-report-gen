# Shared Interfaces

This document defines the **contracts** that all three models and the shared
training/evaluation code must follow. **Do not change anything here without
team agreement** — breaking these interfaces breaks everyone's code downstream.

Last updated: post-integration. Owner: team.

---

## 1. Data loader batch format

All `DataLoader`s — for training, validation, and test — yield batches as
**5-tuples** in this exact order:

```python
(images, input_tokens, target_tokens, lengths, image_ids)
```

### Field specification

| Field           | Type            | Shape                      | Description |
|-----------------|-----------------|----------------------------|-------------|
| `images`        | `torch.FloatTensor` | `(B, 3, 224, 224)`     | ImageNet-normalized X-rays. Single view (no frontal+lateral fusion). |
| `input_tokens`  | `torch.LongTensor`  | `(B, T)`               | Decoder input. Prepended with `<BOS>`, last token of full sequence dropped. PAD-padded. |
| `target_tokens` | `torch.LongTensor`  | `(B, T)`               | Loss target. `<BOS>` dropped, ends with `<EOS>`. Same padding as `input_tokens`. |
| `lengths`       | `torch.LongTensor`  | `(B,)`                 | True (un-padded) length of each target sequence, including `<EOS>`. |
| `image_ids`     | `List[str]`         | length `B`             | Filenames like `"CXR3297_IM-1575/1.png"`. Used for logging/joining hypotheses to references. |

### Conventions

- Batch dimension is first; `T` is **dynamic** per batch (= max length in this batch, NOT a fixed global max).
- Special token IDs are **locked**:
  - `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`, `<SEP>=4`
- Vocabulary is built once from the training split (`min_word_freq=3`). Current size: 975.
- Vocabulary is encapsulated in the `Vocabulary` class in `src/data/iu_xray.py`. It is **not** persisted as a separate file; it is rebuilt deterministically from the same source data each run, and stored in checkpoints via `vocab.state_dict()`.

### About `<SEP>`

`<SEP>` is emitted by the tokenizer wherever a period appeared in the source text. Hierarchical models (e.g., `HierarchicalLSTM`) split reports into sentences on `<SEP>`. Flat models (Transformer, dummy) treat it as just another token. `Vocabulary.decode()` strips `<SEP>` along with PAD/BOS/EOS, so BLEU/METEOR-visible text is unchanged across model families.

### Why a tuple and not a dict

Unpacking is `images, inp, tgt, lens, ids = batch`. Order is enforced. New fields, if ever added, append at the end so older code keeps working.

---

## 2. Model interface

Every model is a `torch.nn.Module` with a fixed constructor signature and two public methods: `forward` and `generate`.

### 2.1 Construction

```python
def __init__(self, vocab_size: int, config: dict):
    ...
```

- `vocab_size`: size of the shared vocabulary (currently 975).
- `config`: a dict containing **only the model-specific hyperparameters** (i.e., the contents of `config["model"]["config"]` from a YAML, NOT the full top-level config).

The training script instantiates models via the factory:

```python
from src.models import get_model
model = get_model(name=cfg["model"]["name"],
                  vocab_size=vocab.size,
                  config=cfg["model"]["config"])
```

To register a new model: add an entry to `_MODEL_REGISTRY` in `src/models/__init__.py` and add a `configs/<name>.yaml`.

### 2.2 `forward()`

Used during training. Teacher-forced; returns a dict containing the loss.

```python
def forward(
    self,
    images:        torch.FloatTensor,   # (B, 3, 224, 224)
    input_tokens:  torch.LongTensor,    # (B, T)
    target_tokens: torch.LongTensor,    # (B, T)
    lengths:       torch.LongTensor,    # (B,)
) -> dict:
    return {"loss": loss_scalar, "logits": logits_tensor}
```

**Contract:**
- Returns a `dict` with at least these two keys:
  - `"loss"`: scalar tensor, ready for `.backward()`. Already mean-reduced over non-pad tokens (using `ignore_index=PAD_ID=0`).
  - `"logits"`: tensor with shape `(B, T, vocab_size)`. Reserved for token-level metrics; **not currently consumed by the trainer**, so a placeholder zeros tensor is acceptable if your model doesn't naturally produce flat logits (e.g., hierarchical decoders).
- Auxiliary losses (e.g., clinical-term prediction in the clinical-term-guided model) are combined internally into the single `"loss"` value. Sub-loss weighting is a model hyperparameter, not a trainer concern.

### 2.3 `generate()`

Used during validation and test. Runs autoregressive decoding.

```python
@torch.no_grad()
def generate(
    self,
    images:     torch.FloatTensor,   # (B, 3, 224, 224)
    max_length: int,
    beam_size:  int = 1,             # 1 = greedy
) -> List[List[int]]:                # B variable-length lists
    ...
```

**Contract:**
- Decorated with `@torch.no_grad()` so it works correctly outside the trainer's no-grad context.
- Returns a `list[list[int]]` of length `B`. Each inner list contains generated token IDs for that sample.
- **Excludes** `<BOS>`. **Excludes** `<EOS>` — generation stops upon emitting `<EOS>` but the token itself is not appended to the returned list. (`Vocabulary.decode()` strips `<EOS>` regardless, so BLEU-visible text is unchanged either way; we standardize on excluding it for consistency.)
- `beam_size=1` (greedy) must always work. `beam_size > 1` is optional per model; if not implemented, accept the argument silently (treat as greedy) or raise `NotImplementedError`.

---

## 3. Vocabulary interface

A single `Vocabulary` class lives in `src/data/iu_xray.py`:

```python
class Vocabulary:
    word_to_id:   dict[str, int]    # the canonical map
    id_to_word:   dict[int, str]
    size:         int               # property; same as len(self)

    def encode(self, text: str) -> list[int]:
        """Tokenize and map to IDs. Unknown words -> <UNK>. Does NOT add BOS/EOS."""

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """IDs -> string. Strips PAD/BOS/EOS/SEP."""

    def __len__(self) -> int: ...

    def state_dict(self) -> dict:
        """For checkpoint persistence."""

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocabulary": ...
```

### Vocabulary construction rules

- Lowercase all text.
- Replace `.` with `<SEP>` so sentence boundaries are tokens.
- Strip remaining light punctuation (`,!?`).
- Split on whitespace.
- Minimum word frequency `3` (rarer words → `<UNK>`).
- Built from training split ONLY (no leakage from val/test).

Special tokens occupy fixed IDs `0..4`. Content words begin at ID 5. Sort order beyond specials is `(-count, word)` for determinism across runs.

---

## 4. Evaluation interface

The project splits evaluation into **two scoring functions** rather than one combined `compute_metrics`. This keeps the slow path optional.

### 4.1 Linguistic metrics (cheap, every val epoch)

`src/evaluation/linguistic_metrics.py`:

```python
def compute_linguistic_metrics(
    predictions: dict[str, str],     # image_id -> generated report
    references:  dict[str, str],     # image_id -> ground-truth report
) -> dict[str, float]:
    """
    Returns a flat dict with the following keys:
        bleu1, bleu2, bleu3, bleu4, meteor, rouge_l, cider
    All values are floats in [0, 1]. Empty dict on internal failure.
    """
```

Implementation uses `pycocoevalcap` (`Bleu`, `Meteor`, `Rouge`, `Cider`, `PTBTokenizer`). METEOR and CIDEr require Java to be on `PATH` for `PTBTokenizer`.

### 4.2 Clinical metrics (expensive, end-of-training only)

`src/evaluation/clinical_metrics.py`:

```python
def compute_clinical_metrics(
    predictions: dict[str, str],     # image_id -> generated report
    references:  dict[str, str],     # image_id -> ground-truth report
) -> dict[str, float]:
    """
    Returns a flat dict with the following keys:
        chexbert_f1_micro
        chexbert_f1_macro
        chexbert_f1_<finding>     for each of the 14 CheXpert conditions
    All values are floats in [0, 1]. Empty dict on internal failure.
    """
```

Implementation uses CheXbert (Stanford, BERT-based labeler). Requires the CheXbert checkpoint at `external/CheXbert/weights/chexbert.pth` and Java for tokenization.

### 4.3 When each is called

| Phase           | Linguistic | Clinical | Caller |
|-----------------|------------|----------|--------|
| Training (val epoch) | Yes        | No       | `src/training/trainer.py::Trainer._validate` |
| Final evaluation | Yes        | Yes      | `src/evaluation/evaluate.py` |

CheXbert is excluded from training-time validation because it adds ~5 min per val epoch. Loss-based or BLEU-based early stopping is sufficient signal; clinical scoring is reserved for end-of-training paper numbers.

### 4.4 Trainer hookup

The trainer imports `compute_linguistic_metrics` lazily and wraps the call in try/except so a failure in the evaluator does not kill a multi-hour training run. Empty dict → no linguistic keys logged that epoch.

The trainer does **not** import `compute_clinical_metrics`. That call lives entirely in `evaluate.py`.

---

## 5. Change process

If you need to change any of the above:

1. Open a GitHub issue tagged `interface-change`.
2. Post in the team channel tagging everyone.
3. Get explicit confirmation from both other teammates before merging.
4. Update this doc in the same PR as the code change.

Interface drift is the #1 killer of multi-person ML projects. Be strict.
