# Shared Interfaces

This document defines the **contracts** that all three models and the shared
training/evaluation code must follow. **Do not change anything here without
team agreement** — breaking these interfaces breaks everyone's code downstream.

Last updated: Day 1. Owner: team.

---

## 1. Data loader batch format

All `DataLoader`s — for training, validation, and test — must yield batches as
**tuples** in this exact order:
```python
(images, input_tokens, target_tokens, lengths, image_ids)
```

### Field specification

| Field           | Type            | Shape                      | Description |
|-----------------|-----------------|----------------------------|-------------|
| `images`        | `torch.FloatTensor` | `(B, C, H, W)`         | Preprocessed X-ray images. `C=3` (grayscale replicated to RGB for ImageNet-pretrained encoders). `H=W=224` by default. Already normalized with ImageNet mean/std. |
| `input_tokens`  | `torch.LongTensor`  | `(B, T)`               | Tokenized report, **right-shifted by one** (i.e., prepended with `<BOS>`, last token dropped). Used as decoder input during teacher forcing. Padded with `<PAD>` to the longest sequence in the batch. |
| `target_tokens` | `torch.LongTensor`  | `(B, T)`               | Tokenized report, **not shifted** (i.e., `<BOS>` dropped, `<EOS>` at end). Used as the loss target. Same padding as `input_tokens`. |
| `lengths`       | `torch.LongTensor`  | `(B,)`                 | True (unpadded) length of each target sequence, including `<EOS>`. Used for masking loss over padding. |
| `image_ids`     | `List[str]`         | length `B`             | IU X-Ray image identifiers (e.g., `"CXR1_1_IM-0001-3001"`). Used for logging and for joining generated reports back to ground truth during evaluation. |

### Conventions

- **Batch dimension is always first** (`B` = batch size).
- `T` = max sequence length *in this batch* (not a fixed global max).
- Padding token ID: `0`. BOS: `1`. EOS: `2`. UNK: `3`. These are locked.
- Vocabulary is built once from the training split and saved to
  `~/scratch/dl-project/data/vocab.pkl`. All models share this vocab.
- Images that are part of the same patient study are treated as independent
  samples for training (we do not fuse frontal+lateral views in the baseline).
  This may change for the clinical-term-guided model; if so, update this doc.

### Why a tuple and not a dict?

We considered a dict (`{"images": ..., "input_tokens": ...}`) but chose a tuple
for speed and simplicity: unpacking is `images, inp, tgt, lens, ids = batch`,
and it forces everyone to think about order, which catches bugs earlier. If
we add more fields later (e.g., clinical tags), we will **append** to the
tuple, never insert in the middle, so older code keeps working.

---

## 2. Model interface

Every model — `HierarchicalLSTM`, `VanillaTransformer`, `ClinicalTransformer` —
must subclass `torch.nn.Module` and implement **exactly two** public methods
beyond `__init__`: `forward` and `generate`.

### 2.1 `forward(images, captions) -> loss`

Used during training. Runs teacher-forced decoding and returns a scalar loss.
```python
def forward(
    self,
    images: torch.FloatTensor,       # (B, C, H, W)
    captions: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor],
    # captions = (input_tokens, target_tokens, lengths)
) -> torch.FloatTensor:              # scalar loss
    ...
```

**Contract:**
- Returns a **single scalar tensor** (the loss), not a dict, not a tuple.
- Loss is already reduced (mean over non-pad tokens). The training loop will
  call `.backward()` on it directly.
- Internal cross-entropy uses `ignore_index=0` (pad token).
- Models that have auxiliary losses (e.g., clinical tag prediction in the
  clinical-term-guided model) **combine them internally** into a single scalar
  before returning. The weighting is a model hyperparameter, not a training
  loop concern. If you need to log the sub-losses, use `self.last_losses` as
  a dict attribute that the training loop can read after `forward()`.

### 2.2 `generate(images) -> token_ids`

Used during validation and test. Runs autoregressive decoding.
```python
@torch.no_grad()
def generate(
    self,
    images: torch.FloatTensor,       # (B, C, H, W)
    max_length: int = 100,
    beam_size: int = 1,              # 1 = greedy
) -> torch.LongTensor:               # (B, T_out), padded with 0
    ...
```

**Contract:**
- Always runs in `eval()` mode internally — do not rely on the caller.
- Returns a padded `LongTensor` of generated token IDs. Sequences that hit
  `<EOS>` early are padded with `0` to the longest sequence in the batch.
- **Does not** include the initial `<BOS>` token in the output.
- **Does** include the `<EOS>` token when generated (so the decoder knows
  where to stop).
- `beam_size=1` (greedy) is the default and must always work. Beam search is
  optional per model; if not implemented, raise `NotImplementedError` for
  `beam_size > 1`.

### 2.3 Construction

All models accept a single `config` argument (a dict or OmegaConf object
loaded from a YAML in `configs/`). Model-specific hyperparameters live under
`config.model.*`. This means the training script can instantiate any model
via:
```python
model = MODEL_REGISTRY[config.model.name](config)
```

where `MODEL_REGISTRY` is a dict mapping `"lstm"`, `"transformer"`,
`"clinical_transformer"` to the corresponding class.

---

## 3. Tokenizer interface

A single tokenizer is shared across all models. It lives in
`src/data/tokenizer.py` and exposes:
```python
class ReportTokenizer:
    def encode(self, text: str) -> List[int]:
        """Text → token IDs. Does NOT add BOS/EOS (caller does that)."""

    def decode(self, ids: List[int]) -> str:
        """Token IDs → text. Strips BOS, EOS, PAD automatically."""

    def __len__(self) -> int:
        """Vocabulary size."""

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ReportTokenizer": ...
```

Vocabulary construction rules:
- Lowercase all text.
- Split on whitespace + basic punctuation.
- Minimum word frequency: `3` (rarer words → `<UNK>`).
- Special tokens reserved: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`.
- Vocabulary is built **only from the training split** (no leakage from val/test).

---

## 4. Evaluation interface

`src/eval/metrics.py` exposes one top-level function:
```python
def compute_metrics(
    predictions: Dict[str, str],     # image_id → generated report
    references:  Dict[str, str],     # image_id → ground-truth report
) -> Dict[str, float]:
    """
    Returns a flat dict of metric name → score. Keys include:
      bleu_1, bleu_2, bleu_3, bleu_4, meteor, cider,
      chexbert_f1_micro, chexbert_f1_macro,
      plus per-finding F1 under keys like chexbert_f1_<finding>.
    """
```

This function is called by the training loop after each validation pass and
all returned metrics are logged to W&B under the `val/` prefix.

---

## 5. Change process

If you need to change any of the above:

1. Open a GitHub issue tagged `interface-change`.
2. Post in `#dl-project` on Discord tagging everyone.
3. Get explicit 👍 from both other teammates before merging.
4. Update this doc in the same PR as the code change.

Interface drift is the #1 killer of multi-person ML projects. Be strict.
