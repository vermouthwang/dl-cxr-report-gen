"""
Compute CheXbert-based clinical F1 metrics for chest X-ray report generation.

Called once at end-of-training (not every val epoch) - CheXbert is slower than BLEU/METEOR.

Based on:
  https://github.com/rajpurkarlab/CXR-Report-Metric

Label convention (CheXpert/CheXbert):
    1.0  = Positive
    0.0  = Negative
   -1.0  = Uncertain
    NaN  = Blank / Not mentioned

F1 is computed as macro-average over the three non-blank classes
{positive, negative, uncertain} per condition. NaN entries in either
prediction or reference are excluded from that condition's computation.

Return keys (as specified in interfaces.md):
    chexbert_f1_micro (micro F1 across all conditions)
    chexbert_f1_macro (mean of per-condition F1s)
    chexbert_f1_<finding> (per-condition F1)
        e.g. chexbert_f1_cardiomegaly
             chexbert_f1_no_finding
             chexbert_f1_enlarged_cardiomediastinum
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Paths
# File is at:  <repo_root>/src/evaluation/clinical_metrics.py
# parents[0] = src/evaluation/
# parents[1] = src/
# parents[2] = <repo_root>/
_REPO_ROOT    = Path(__file__).resolve().parents[2]
_CHEXBERT_SRC = _REPO_ROOT / "external" / "CheXbert" / "src"
_CHECKPOINT   = _REPO_ROOT / "external" / "CheXbert" / "weights" / "chexbert.pth"

# 14 conditions in the exact order constants.py defines them.
# Do not reorder - label.py returns y_pred indexed by this order.
CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# Condition name -> metric key suffix
# e.g. "Pleural Effusion" -> "pleural_effusion"
# Matches interfaces.md: keys like "chexbert_f1_<finding>"
def _finding_key(name: str) -> str:
    return name.lower().replace(" ", "_")

# CheXbert labeling
def _load_label_module():
    """
    Load CheXbert's label.py via spec_from_file_location, bypassing the
    import cache.

    importlib.import_module("label") would return a stale or wrong module if
    anything else on sys.modules already holds that name. Loading by file path
    is unambiguous and safe to call repeatedly.
    """
    label_path = _CHEXBERT_SRC / "label.py"
    if not label_path.exists():
        raise FileNotFoundError(
            f"CheXbert label.py not found at {label_path}. "
        )

    # CheXbert's label.py has relative imports (models, datasets, constants,
    # utils, bert_tokenizer) that only resolve when src/ is on sys.path.
    src_str = str(_CHEXBERT_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    spec   = importlib.util.spec_from_file_location("chexbert_label", label_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label_reports(texts: list[str]) -> pd.DataFrame:
    """
    Run CheXbert on a list of report strings.

    Returns a DataFrame of shape (n_reports, 14), columns = CONDITIONS.
    Values: NaN = blank, 1.0 = positive, 0.0 = negative, -1.0 = uncertain.

    Raises on any error - callers catch and degrade gracefully.
    """
    if not _CHECKPOINT.exists():
        raise FileNotFoundError(
            f"CheXbert checkpoint not found at {_CHECKPOINT}. "
        )

    label_mod = _load_label_module()

    # CheXbert's label.py reads a CSV with a "Report Impression" column and
    # writes labeled_reports.csv to an output directory.
    with tempfile.TemporaryDirectory() as tmp:
        csv_in      = os.path.join(tmp, "reports.csv")
        csv_out_dir = os.path.join(tmp, "out")
        os.makedirs(csv_out_dir, exist_ok=True)

        pd.DataFrame({"Report Impression": texts}).to_csv(csv_in, index=False)

        y_pred = label_mod.label(str(_CHECKPOINT), csv_in)
        label_mod.save_preds(y_pred, csv_in, csv_out_dir)

        df = pd.read_csv(os.path.join(csv_out_dir, "labeled_reports.csv"))

    missing = [c for c in CONDITIONS if c not in df.columns]
    if missing:
        raise ValueError(f"CheXbert output missing conditions: {missing}")

    return df[CONDITIONS].reset_index(drop=True)


# F1 helpers

def _f1_for_condition(pred_col: pd.Series, ref_col: pd.Series) -> float:
    """
    Macro-averaged F1 over {positive=1, negative=0, uncertain=-1} for one
    condition. Samples where either side is NaN (blank) are excluded.

    Returns 0.0 when no non-blank samples exist (for W&B logging).
    """
    mask = pred_col.notna() & ref_col.notna()
    p = pred_col[mask].values.astype(float)
    r = ref_col[mask].values.astype(float)

    if len(p) == 0:
        return 0.0

    f1s = []
    for cls in (1.0, 0.0, -1.0):
        tp = float(((p == cls) & (r == cls)).sum())
        fp = float(((p == cls) & (r != cls)).sum())
        fn = float(((p != cls) & (r == cls)).sum())
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = prec + rec
        f1s.append(2 * prec * rec / denom if denom > 0 else 0.0)

    return float(np.mean(f1s))


def _micro_f1(pred_df: pd.DataFrame, ref_df: pd.DataFrame) -> float:
    """
    Micro-averaged F1 across all 14 conditions and all 3 classes.
    Pools TP/FP/FN globally before computing precision and recall.
    NaN entries are excluded per cell.
    """
    pred_arr = pred_df[CONDITIONS].values.astype(float)
    ref_arr  = ref_df[CONDITIONS].values.astype(float)
    valid    = ~(np.isnan(pred_arr) | np.isnan(ref_arr))  # (n, 14) bool mask

    tp_total = fp_total = fn_total = 0
    for cls in (1.0, 0.0, -1.0):
        p_cls = (pred_arr == cls) & valid
        r_cls = (ref_arr  == cls) & valid
        tp_total += int((p_cls & r_cls).sum())
        fp_total += int((p_cls & ~r_cls).sum())
        fn_total += int((~p_cls & r_cls).sum())

    prec  = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec   = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0


# Public API  (interfaces.md)


def compute_clinical_metrics(
    predictions: Dict[str, str],   # image_id → generated report
    references:  Dict[str, str],   # image_id → ground-truth report
) -> Dict[str, float]:
    """
    Compute CheXbert F1 scores for generated vs. ground-truth reports.

    Follows interfaces.md: same (predictions, references) signature as
    compute_linguistic_metrics. Called only at end of training.

    Returns a flat dict of metric name → score, all values in [0, 1]:
    chexbert_f1_micro (micro F1 across all conditions)
    chexbert_f1_macro (mean of per-condition F1s)
    chexbert_f1_<finding> (per-condition F1)
        e.g. chexbert_f1_cardiomegaly
             chexbert_f1_no_finding
             chexbert_f1_enlarged_cardiomediastinum

    Returns {} on any failure (with a warning) so a CheXbert crash never
    kills a training run.
    """
    shared_ids = sorted(set(predictions) & set(references))

    if not shared_ids:
        warnings.warn(
            "compute_clinical_metrics: no shared image_ids between predictions "
            "and references - returning {}.",
            stacklevel=2,
        )
        return {}

    # Warn (but don't crash) on partial key mismatch, consistent with how
    # linguistic_metrics handles this case.
    only_pred = set(predictions) - set(references)
    only_ref  = set(references)  - set(predictions)
    if only_pred or only_ref:
        warnings.warn(
            f"compute_clinical_metrics: key mismatch - "
            f"{len(only_pred)} ids only in predictions, "
            f"{len(only_ref)} only in references. "
            f"Evaluating on {len(shared_ids)} shared ids.",
            stacklevel=2,
        )

    pred_texts = [predictions[i] for i in shared_ids]
    ref_texts  = [references[i]  for i in shared_ids]

    # Label with CheXbert
    try:
        logger.info(
            "CheXbert: labeling %d prediction/reference pairs …",
            len(shared_ids),
        )
        pred_df = _label_reports(pred_texts)
        ref_df  = _label_reports(ref_texts)
    except Exception as exc:
        warnings.warn(
            f"compute_clinical_metrics: CheXbert labeling failed - {exc}. "
            "Returning {}.",
            stacklevel=2,
        )
        logger.exception("CheXbert labeling error")
        return {}

    # Compute metrics
    try:
        metrics: Dict[str, float] = {}

        # Per-condition F1 -> key: chexbert_f1_<finding>
        per_cond_f1s = []
        for cond in CONDITIONS:
            f1 = _f1_for_condition(pred_df[cond], ref_df[cond])
            metrics[f"chexbert_f1_{_finding_key(cond)}"] = f1
            per_cond_f1s.append(f1)

        metrics["chexbert_f1_macro"] = float(np.mean(per_cond_f1s))
        metrics["chexbert_f1_micro"] = _micro_f1(pred_df, ref_df)

        logger.info(
            "CheXbert done - micro=%.4f  macro=%.4f",
            metrics["chexbert_f1_micro"],
            metrics["chexbert_f1_macro"],
        )
        return metrics

    except Exception as exc:
        warnings.warn(
            f"compute_clinical_metrics: metric computation failed - {exc}. "
            "Returning {}.",
            stacklevel=2,
        )
        logger.exception("CheXbert metric computation error")
        return {}
