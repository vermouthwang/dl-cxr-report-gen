"""Clinical vocabulary loader for the clinical-term-guided Transformer.

Reads a curated clinical lexicon (data_meta/clinical_terms.json) and a vocab
(word_to_id mapping), and produces:

  - `weight_vector`  : (vocab_size,) float tensor for nn.CrossEntropyLoss weighting
  - `bias_mask`      : (vocab_size,) float tensor (0 or 1) for inference-time logit bias
  - `category_to_ids`: dict[str, list[int]] for diagnostics and CheXbert evaluation later

The lexicon defines three tiers — `findings`, `negations`, `hedges`. The model's
config controls the per-tier weight multipliers. Missing seed words (i.e. words
present in the lexicon but absent from the vocab, typically because they fell
below min_word_freq) are tolerated and logged.

Design notes:
  - This module has no torch dependency at import time. Tensors are produced
    on demand via `weight_vector(...)` and `bias_mask(...)` so the lexicon
    object itself is cheap to construct and inspect (e.g., from a notebook).
  - Tier overlap (e.g., 'normal' is both a No Finding seed and a negation)
    is resolved by taking the MAX weight across tiers. This is conservative:
    a word can only be more important, not less.
  - PAD_ID (=0) always gets weight 1.0; ignored by CE via ignore_index=0 anyway.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Special token IDs — must match src/data/iu_xray.py
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SEP_ID = 4
SPECIAL_IDS = frozenset([PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID])


@dataclass
class ClinicalLexicon:
    """In-memory representation of the clinical lexicon, resolved against a vocab.

    Attributes:
        vocab_size: total vocab size (including specials)
        category_to_ids: maps CheXpert finding category name -> sorted list of token IDs
        negation_ids: token IDs in the 'negations' tier
        hedge_ids: token IDs in the 'hedges' tier
        all_finding_ids: union of all category IDs (for convenience)
        missing: per-tier list of seed words not found in vocab (for diagnostics)
    """
    vocab_size: int
    category_to_ids: dict[str, list[int]]
    negation_ids: list[int]
    hedge_ids: list[int]
    all_finding_ids: list[int] = field(init=False)
    missing: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        seen: set[int] = set()
        for ids in self.category_to_ids.values():
            seen.update(ids)
        self.all_finding_ids = sorted(seen)

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_files(
        cls,
        lexicon_path: str | Path,
        word_to_id: dict[str, int],
    ) -> "ClinicalLexicon":
        """Load a lexicon JSON and resolve seed words to token IDs.

        Args:
            lexicon_path: path to clinical_terms.json
            word_to_id: vocabulary mapping (from Vocabulary.word_to_id)

        Returns:
            A populated ClinicalLexicon.
        """
        with open(lexicon_path) as f:
            spec = json.load(f)

        vocab_size = len(word_to_id)

        category_to_ids: dict[str, list[int]] = {}
        missing: dict[str, list[str]] = {}

        # Findings (per-category)
        for cat, seeds in spec.get("findings", {}).items():
            ids = []
            cat_missing = []
            for w in seeds:
                tid = word_to_id.get(w)
                if tid is None:
                    cat_missing.append(w)
                else:
                    ids.append(tid)
            category_to_ids[cat] = sorted(set(ids))
            if cat_missing:
                missing[cat] = cat_missing

        # Negations
        negation_ids = []
        neg_missing = []
        for w in spec.get("negations", []):
            tid = word_to_id.get(w)
            if tid is None:
                neg_missing.append(w)
            else:
                negation_ids.append(tid)
        negation_ids = sorted(set(negation_ids))
        if neg_missing:
            missing["__negations__"] = neg_missing

        # Hedges
        hedge_ids = []
        hedge_missing = []
        for w in spec.get("hedges", []):
            tid = word_to_id.get(w)
            if tid is None:
                hedge_missing.append(w)
            else:
                hedge_ids.append(tid)
        hedge_ids = sorted(set(hedge_ids))
        if hedge_missing:
            missing["__hedges__"] = hedge_missing

        return cls(
            vocab_size=vocab_size,
            category_to_ids=category_to_ids,
            negation_ids=negation_ids,
            hedge_ids=hedge_ids,
            missing=missing,
        )

    # ------------------------------------------------------------------ #
    # Tensor producers (torch imported lazily)                           #
    # ------------------------------------------------------------------ #

    def weight_vector(
        self,
        alpha_finding: float = 3.0,
        alpha_negation: float = 2.0,
        alpha_hedge: float = 1.0,
        dtype=None,
    ):
        """Build a per-class weight vector for nn.CrossEntropyLoss.

        Returns a (vocab_size,) float tensor where:
          - finding token positions get `alpha_finding`
          - negation token positions get `alpha_negation`
          - hedge token positions get `alpha_hedge`
          - everything else gets 1.0
          - if a token is in multiple tiers, MAX of the assigned weights wins
          - PAD_ID gets 1.0 (irrelevant: ignore_index=0 masks it anyway)

        Args:
            alpha_finding: multiplier for finding tokens (default 3.0)
            alpha_negation: multiplier for negation tokens (default 2.0)
            alpha_hedge: multiplier for hedge tokens (default 1.0, i.e. off)
            dtype: torch dtype (default torch.float32)

        Returns:
            torch.Tensor of shape (vocab_size,)
        """
        import torch  # local import — keep this module torch-optional at import time
        if dtype is None:
            dtype = torch.float32

        w = torch.ones(self.vocab_size, dtype=dtype)

        # Apply in order from lowest to highest priority, then take MAX at overlaps.
        # Since hedges < negations < findings is the assumed priority, and we apply
        # MAX explicitly, ordering inside this function doesn't matter — but we
        # write it ordered for readability.

        for tid in self.hedge_ids:
            if tid in SPECIAL_IDS:
                continue
            w[tid] = max(float(w[tid]), float(alpha_hedge))

        for tid in self.negation_ids:
            if tid in SPECIAL_IDS:
                continue
            w[tid] = max(float(w[tid]), float(alpha_negation))

        for tid in self.all_finding_ids:
            if tid in SPECIAL_IDS:
                continue
            w[tid] = max(float(w[tid]), float(alpha_finding))

        return w

    def bias_mask(self, include_negations: bool = True, dtype=None):
        """Build a (vocab_size,) binary mask for inference-time logit bias.

        Positions of clinical tokens get 1.0, everything else 0.0.

        Args:
            include_negations: if True, negation tokens are also biased (recommended).
                Hedges are never included in the bias mask.
            dtype: torch dtype (default torch.float32)

        Returns:
            torch.Tensor of shape (vocab_size,)
        """
        import torch
        if dtype is None:
            dtype = torch.float32

        mask = torch.zeros(self.vocab_size, dtype=dtype)
        for tid in self.all_finding_ids:
            if tid in SPECIAL_IDS:
                continue
            mask[tid] = 1.0
        if include_negations:
            for tid in self.negation_ids:
                if tid in SPECIAL_IDS:
                    continue
                mask[tid] = 1.0
        return mask

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def summary(self, id_to_word: Optional[dict[int, str]] = None) -> str:
        """Return a human-readable coverage report. Use for verification scripts."""
        lines = []
        lines.append(f"ClinicalLexicon: vocab_size={self.vocab_size}")
        lines.append(f"  total finding token IDs: {len(self.all_finding_ids)}")
        lines.append(f"  negation token IDs: {len(self.negation_ids)}")
        lines.append(f"  hedge token IDs: {len(self.hedge_ids)}")
        lines.append("")
        lines.append("Per-category coverage:")

        for cat, ids in self.category_to_ids.items():
            if id_to_word is not None:
                tokens = [id_to_word.get(i, f"<id={i}>") for i in ids]
                tok_str = " ".join(repr(t) for t in tokens)
            else:
                tok_str = " ".join(str(i) for i in ids)
            line = f"  {cat:<32} ({len(ids):2d} tokens): {tok_str}"
            if cat in self.missing:
                line += f"  [missing: {self.missing[cat]}]"
            lines.append(line)

        if self.negation_ids:
            if id_to_word is not None:
                tokens = [id_to_word.get(i, f"<id={i}>") for i in self.negation_ids]
                tok_str = " ".join(repr(t) for t in tokens)
            else:
                tok_str = " ".join(str(i) for i in self.negation_ids)
            lines.append("")
            lines.append(f"  negations ({len(self.negation_ids)} tokens): {tok_str}")
            if "__negations__" in self.missing:
                lines.append(f"    [missing: {self.missing['__negations__']}]")

        if self.hedge_ids:
            if id_to_word is not None:
                tokens = [id_to_word.get(i, f"<id={i}>") for i in self.hedge_ids]
                tok_str = " ".join(repr(t) for t in tokens)
            else:
                tok_str = " ".join(str(i) for i in self.hedge_ids)
            lines.append(f"  hedges    ({len(self.hedge_ids)} tokens): {tok_str}")
            if "__hedges__" in self.missing:
                lines.append(f"    [missing: {self.missing['__hedges__']}]")

        # Empty categories (zero matched tokens) — caller should know
        empty = [c for c, ids in self.category_to_ids.items() if len(ids) == 0]
        if empty:
            lines.append("")
            lines.append(f"  WARNING: empty categories (no tokens in vocab): {empty}")

        return "\n".join(lines)