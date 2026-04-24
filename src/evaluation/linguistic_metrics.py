"""
@Yousuf plz replace this with a real implementation using pycocoevalcap.
The training script calls compute_linguistic_metrics() and degrades gracefully
when the returned dict is empty.

Real implementation contract:
    compute_linguistic_metrics(
        hypotheses: Dict[str,str],   # generated reports
        references: Dict[str,str],   # ground-truth reports
    ) -> dict[str, float]

    Expected returned keys (suggested): bleu1, bleu2, bleu3, bleu4, meteor,
    rouge_l, cider. The training script will log whatever is returned; for
    early-stopping on bleu4, set config.early_stopping.metric = "val_bleu4".

Based on:
  https://gist.github.com/solversa/1a2f18d9e880b4cbb9a5afa62222181c
  (Python-2 pycocoevalcap wrapper for non-COCO datasets)

Adapted for this project:
  - Follows interfaces.md: inputs are Dict[str, str] keyed by image_id
  - compute_linguistic_metrics: called every val epoch (BLEU/METEOR/CIDEr)
  - compute_clinical_metrics:   called only at end of training (CheXbert)
  - Keys match what the trainer logs to W&B:
      bleu1, bleu2, bleu3, bleu4, meteor, rouge_l, cider

Ensure dependency is installed (on PACE):
    pip install pycocoevalcap
"""
from typing import Dict, List
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# pycocoevalcap uses PTBTokenizer (needs Java) for METEOR/CIDEr.
# BLEU and ROUGE-L work without it, but we call tokenize() for all
# metrics so scoring is consistent with the standard CXR literature.


def compute_linguistic_metrics(
    predictions: Dict[str, str],   # image_id -> generated report
    references:  Dict[str, str],   # image_id -> ground-truth report
) -> dict[str, float]:
    """
    Compute BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr over a corpus.
    Called every validation epoch by the trainer.

    Args:
        predictions : dict of image_id -> generated report string
        references  : dict of image_id -> ground-truth report string

    Returns:
        dict with keys: bleu1, bleu2, bleu3, bleu4, meteor, rouge_l, cider
        All values are floats in [0, 1].

    Notes:
        - PTBTokenizer is required for METEOR and CIDEr.
          Make sure Java is available on PACE before running evaluation.
        - Each image_id must appear in both predictions and references.
    """
    assert set(predictions.keys()) == set(references.keys()), \
        "predictions and references must have the same image_id keys"

    keys       = list(predictions.keys())
    hypotheses = [predictions[k] for k in keys]
    refs       = [references[k]  for k in keys]

    #Build the dicts pycocoevalcap expects
    # Keys are integer image IDs (we use list indices).
    # gts: image_id -> [{"caption": str}, ...]   (list; allows multiple refs)
    # res: image_id -> [{"caption": str}]         (list; exactly one hyp)
    gts = {i: [{"caption": ref}]  for i, ref in enumerate(refs)}
    res = {i: [{"caption": hyp}]  for i, hyp in enumerate(hypotheses)}

    # PTB tokenization (lowercases + strips punctuation)
    # Keeps scoring consistent with published CXR report-gen benchmarks.
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Set up scorers
    scorers = [
        (Bleu(4),    ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),   "METEOR"),
        (Rouge(),    "ROUGE_L"),
        (Cider(),    "CIDEr"),
    ]

    try:
        raw: dict = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    raw[m] = s
            else:
                raw[method] = score
    except Exception as e:
        import warnings
        warnings.warn(
            f"[linguistic_metrics] Scoring failed - returning empty dict.\n"
            f"Error: {e}",
            RuntimeWarning,
        )
        return {}

    # Remap to the key names the trainer expects
    return {
        "bleu1":   raw["Bleu_1"],
        "bleu2":   raw["Bleu_2"],
        "bleu3":   raw["Bleu_3"],
        "bleu4":   raw["Bleu_4"],
        "meteor":  raw["METEOR"],
        "rouge_l": raw["ROUGE_L"],
        "cider":   raw["CIDEr"],
    }