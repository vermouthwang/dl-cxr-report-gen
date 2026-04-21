"""
@Yousuf plz replace this with a real implementation using pycocoevalcap.
The training script calls compute_linguistic_metrics() and degrades gracefully
when the returned dict is empty.

Real implementation contract:
    compute_linguistic_metrics(
        hypotheses: list[str],   # generated reports
        references: list[str],   # ground-truth reports
    ) -> dict[str, float]

    Expected returned keys (suggested): bleu1, bleu2, bleu3, bleu4, meteor,
    rouge_l, cider. The training script will log whatever is returned; for
    early-stopping on bleu4, set config.early_stopping.metric = "val_bleu4".
"""
from __future__ import annotations


_WARNED_ONCE = False


def compute_linguistic_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Stub. Returns empty dict. Logs a one-time warning."""
    # @YousufQ7
    global _WARNED_ONCE
    if not _WARNED_ONCE:
        print("[WARN] linguistic_metrics is a stub — BLEU/METEOR/CIDEr not computed.")
        _WARNED_ONCE = True
    return {}