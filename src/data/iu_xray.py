"""
IU X-Ray dataloader for chest X-ray report generation.

Refactor of notebooks/data_processing.ipynb -> importable module.
No side effects on import: all work happens inside build_dataloaders().

BATCH FORMAT (important for all model authors):
    Each batch is a 5-tuple:
        (images, input_tokens, target_tokens, lengths, image_ids)
      - images:        tensor (B, 3, 224, 224), ImageNet-normalized
      - input_tokens:  tensor (B, T), BOS-prefixed, right-padded with PAD_ID=0
      - target_tokens: tensor (B, T), EOS-suffixed, right-padded with PAD_ID=0
      - lengths:       tensor (B,), true un-padded lengths of input/target
      - image_ids:     list[str] of length B, filenames like "CXR123_IM-0001/0.png"

    T is DYNAMIC: it equals max(lengths) in that batch, not a fixed 59.
    Models must NOT assume any fixed sequence length.

    Teacher-forcing offset: input[t] predicts target[t]. They are shifted by one:
        full_seq   = [BOS, w1, w2, ..., wN, EOS]
        input_tok  = [BOS, w1, w2, ..., wN]        (length N+1)
        target_tok = [w1, w2, ..., wN, EOS]        (length N+1)

MODEL INTERFACE (agreed with team):
    forward(images, input_tokens, target_tokens, lengths) -> {"loss": ..., "logits": ...}
        - Loss must use ignore_index=PAD_ID for padding masking.
        - Logits shape: (B, T, vocab_size), T matching input_tokens.shape[1].
        - If your model internally reshapes (e.g., hierarchical), reshape logits
          back to flat (B, T, V) before returning so the trainer's metrics work.

    generate(images, max_length, beam_size=1) -> list[list[int]]
        - Returns B variable-length lists of token IDs.
        - Generated IDs should exclude BOS and stop at (but exclude) EOS.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# -----------------------------------------------------------------------------
# Special tokens. Keep in sync with model configs.
# -----------------------------------------------------------------------------
PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3
SEP_ID: int = 4   # sentence separator (replaces periods). Used by hierarchical models
                  # to split reports into sentences. Decoded output skips <SEP> so BLEU
                  # references/hypotheses look identical to the non-hierarchical case.
SPECIAL_TOKENS = {
    "<PAD>": PAD_ID,
    "<BOS>": BOS_ID,
    "<EOS>": EOS_ID,
    "<UNK>": UNK_ID,
    "<SEP>": SEP_ID,
}

DEFAULT_DATA_ROOT = Path.home() / "scratch" / "dl-project" / "data" / "iu-xray" / "iu_xray"
DEFAULT_MAX_REPORT_LEN = 60
DEFAULT_MIN_WORD_FREQ = 3

# ImageNet normalization (matches torchvision pretrained encoders)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------
class Vocabulary:
    """
    Mapping between tokens (strings) and integer IDs.

    Built from the training split only, with minimum frequency filter.
    Special tokens (<PAD>, <BOS>, <EOS>, <UNK>) always occupy IDs 0-3.

    Persistence: use state_dict() to save, from_state_dict() to restore.
    This is what gets stored in training checkpoints.
    """

    def __init__(self, word_to_id: dict[str, int]):
        # Validate special tokens are where we expect.
        for tok, expected_id in SPECIAL_TOKENS.items():
            if word_to_id.get(tok) != expected_id:
                raise ValueError(
                    f"Vocabulary special token {tok} must have ID {expected_id}, "
                    f"got {word_to_id.get(tok)}"
                )
        self.word_to_id: dict[str, int] = dict(word_to_id)
        self.id_to_word: dict[int, str] = {v: k for k, v in word_to_id.items()}

    def __len__(self) -> int:
        return len(self.word_to_id)

    @property
    def size(self) -> int:
        return len(self)

    def encode(self, text: str) -> list[int]:
        """Tokenize and convert to IDs. Unknown words -> UNK_ID. Does NOT add BOS/EOS."""
        return [self.word_to_id.get(t, UNK_ID) for t in simple_tokenizer(text)]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Convert IDs back to a space-joined string. Strips PAD/BOS/EOS/SEP."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (PAD_ID, BOS_ID, EOS_ID, SEP_ID):
                continue
            words.append(self.id_to_word.get(tid, "<UNK>"))
        return " ".join(words)

    def state_dict(self) -> dict:
        """For checkpoint serialization."""
        return {"word_to_id": dict(self.word_to_id)}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocabulary":
        return cls(state["word_to_id"])


def simple_tokenizer(text: str) -> list[str]:
    """
    Tokenize text into a list of string tokens.

    Process:
      1. Lowercase.
      2. Replace every period with " <SEP> " so sentence boundaries become tokens.
         (Whitespace on both sides ensures <SEP> is split out as its own token.)
      3. Strip other light punctuation (commas, exclamation, question marks).
      4. Split on whitespace.

    The <SEP> token is needed by hierarchical models to split reports into
    sentences. Models that don't care (Transformer, dummy) simply learn to
    treat <SEP> as another token; the Vocabulary.decode() call strips it so
    BLEU-visible text is unchanged from the pre-<SEP> tokenizer.
    """
    text = text.lower()
    text = text.replace(".", " <SEP> ")                 # sentence boundary marker
    text = re.sub(r"[,!?]", "", text)                   # strip other light punctuation
    return text.split()


def _build_vocabulary(train_findings: list[str], min_word_freq: int) -> Vocabulary:
    """Build vocabulary from training split only."""
    counter: Counter[str] = Counter()
    for finding in train_findings:
        counter.update(simple_tokenizer(finding))

    word_to_id: dict[str, int] = dict(SPECIAL_TOKENS)  # ensure specials at 0-4
    # Sort by (-count, word) for determinism across runs
    for word, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < min_word_freq:
            continue
        if word in word_to_id:
            # Already a special token (e.g., "<SEP>" emitted by the tokenizer).
            # Don't reassign - would clobber the reserved ID.
            continue
        word_to_id[word] = len(word_to_id)

    return Vocabulary(word_to_id)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class XRayDataset(Dataset):
    """
    Produces (image, token sequences) per sample. Collation into batches happens
    in pad_collate() below.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: Path,
        vocab: Vocabulary,
        split: Optional[str] = None,
        max_len: int = DEFAULT_MAX_REPORT_LEN,
        transform=None,
    ):
        if split is not None:
            dataframe = dataframe[dataframe["split"] == split].reset_index(drop=True)
        if len(dataframe) == 0:
            raise ValueError(f"Empty dataset for split={split}")
        self.df = dataframe
        self.img_dir = Path(img_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Image
        img_path = self.img_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Text: tokenize -> encode -> wrap with BOS/EOS -> truncate
        token_ids = self.vocab.encode(row["finding"])
        token_ids = [BOS_ID] + token_ids + [EOS_ID]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len - 1] + [EOS_ID]

        # Teacher-forcing shift: input is [BOS, ..., wN]; target is [..., wN, EOS]
        input_tokens = token_ids[:-1]
        target_tokens = token_ids[1:]
        length = len(target_tokens)

        return {
            "image": image,
            "input_tokens": torch.tensor(input_tokens, dtype=torch.long),
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "length": length,
            "image_id": row["filename"],
        }


def pad_collate(batch: list[dict]):
    """
    Collate a list of samples into the 5-tuple batch format.
    Sequences are right-padded with PAD_ID. T = max(lengths) in this batch.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    input_tokens = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    target_tokens = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["length"]
        input_tokens[i, :L] = b["input_tokens"]
        target_tokens[i, :L] = b["target_tokens"]

    image_ids = [b["image_id"] for b in batch]
    return images, input_tokens, target_tokens, lengths, image_ids


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
@dataclass
class DataBundle:
    """
    Everything train.py needs from the data pipeline.
    Returned by build_dataloaders().
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab: Vocabulary


def _default_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _resolve_data_root(explicit: Optional[str | Path]) -> Path:
    """
    Priority: explicit arg -> IU_XRAY_ROOT env var -> built-in default.
    Fail loudly if the resolved path doesn't exist.
    """
    if explicit is not None:
        root = Path(explicit)
    elif os.environ.get("IU_XRAY_ROOT"):
        root = Path(os.environ["IU_XRAY_ROOT"])
    else:
        root = DEFAULT_DATA_ROOT

    if not root.is_dir():
        raise FileNotFoundError(
            f"IU X-Ray dataset root not found: {root}\n"
            f"Set IU_XRAY_ROOT env var or pass data.root in config. "
            f"See docs/setup_iu_xray.md for download instructions."
        )
    annot = root / "annotation.json"
    if not annot.exists():
        raise FileNotFoundError(f"annotation.json missing at {annot}")
    return root


def build_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    max_report_len: int = DEFAULT_MAX_REPORT_LEN,
    min_word_freq: int = DEFAULT_MIN_WORD_FREQ,
    data_root: Optional[str | Path] = None,
    transform=None,
    shuffle_train: bool = True,
) -> DataBundle:
    """
    Build train/val/test DataLoaders and the Vocabulary.

    Args:
        batch_size: batch size for all three loaders.
        num_workers: DataLoader workers. Set to 0 if you see SLURM worker hangs.
        max_report_len: max tokens per report INCLUDING BOS/EOS.
        min_word_freq: words with count < this are mapped to <UNK>.
        data_root: override. If None, uses IU_XRAY_ROOT env var, else default path.
        transform: torchvision transform. If None, uses 224x224 + ImageNet norm.
        shuffle_train: disable for deterministic debugging.

    Returns:
        DataBundle with train/val/test loaders and vocab.
    """
    root = _resolve_data_root(data_root)
    images_dir = root / "images"
    annotation_path = root / "annotation.json"

    # Load R2Gen annotation
    with open(annotation_path) as f:
        annotation = json.load(f)

    # Build (filename, report, split) table
    # R2Gen stores one report per study; each image in a study shares that report.
    filename_to_split: dict[str, str] = {}
    records: list[tuple[str, str, str]] = []
    for split_name in ("train", "val", "test"):
        for entry in annotation[split_name]:
            report = entry["report"]
            if not isinstance(report, str):
                continue
            cleaned = " ".join(report.lower().split())
            for rel_path in entry["image_path"]:
                # Skip duplicate filenames (shouldn't happen in R2Gen, but defensive).
                if rel_path in filename_to_split:
                    continue
                filename_to_split[rel_path] = split_name
                records.append((rel_path, cleaned, split_name))

    df = pd.DataFrame(records, columns=["filename", "finding", "split"])
    if df.empty:
        raise RuntimeError(f"No records loaded from {annotation_path}")

    # Build vocab from TRAIN SPLIT ONLY
    train_findings = df.loc[df["split"] == "train", "finding"].tolist()
    vocab = _build_vocabulary(train_findings, min_word_freq=min_word_freq)

    tfm = transform if transform is not None else _default_transforms()

    train_ds = XRayDataset(df, images_dir, vocab, split="train", max_len=max_report_len, transform=tfm)
    val_ds = XRayDataset(df, images_dir, vocab, split="val", max_len=max_report_len, transform=tfm)
    test_ds = XRayDataset(df, images_dir, vocab, split="test", max_len=max_report_len, transform=tfm)

    # pin_memory=True speeds up host->device copies when using CUDA; harmless on CPU.
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=shuffle_train, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, vocab=vocab)


# -----------------------------------------------------------------------------
# CLI sanity check: `python -m src.data.iu_xray`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    bundle = build_dataloaders(batch_size=4, num_workers=0)
    print(f"Vocab size: {bundle.vocab.size}")
    print(f"Train/Val/Test batches: "
          f"{len(bundle.train_loader)}/{len(bundle.val_loader)}/{len(bundle.test_loader)}")

    images, input_tokens, target_tokens, lengths, image_ids = next(iter(bundle.train_loader))
    print(f"\nBatch shapes:")
    print(f"  images:        {images.shape}")
    print(f"  input_tokens:  {input_tokens.shape}")
    print(f"  target_tokens: {target_tokens.shape}")
    print(f"  lengths:       {lengths.tolist()}")
    print(f"  image_ids[:2]: {image_ids[:2]}")
    print(f"\nExample decoded target: {bundle.vocab.decode(target_tokens[0, :lengths[0]])!r}")