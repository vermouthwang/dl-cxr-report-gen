"""
Plot training and validation loss curves for the four headline models, side by side.

"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent

# Path to BOTH CSVs. Update these to match your filenames.
TRAIN_CSV = HERE / "headline_trainingLoss.csv"   # the "train/loss_step" file
VAL_CSV   = HERE / "headline_valLoss.csv"        # the "val_loss" file

OUT_PDF = HERE / "training_curves.pdf"
OUT_PNG = HERE / "training_curves.png"

# Smoothing for noisy train/loss_step curves (rolling-mean window in points).
# 0 = no smoothing. 10 is a good default.
TRAIN_SMOOTH_WINDOW = 10

# Column-name-in-CSV  ->  (display label, color, linestyle)
SERIES = {
    "transformer_6L_unfrozen": {
        "label":     "transformer_6L_unfrozen",
        "color":     "#1f77b4",
        "linestyle": "-",
    },
    "clinical_transformer_D": {
        "label":     "clinical_transformer_softguide_3L",
        "color":     "#ff7f0e",
        "linestyle": "-",
    },
    "clinical_transformer_D_6L": {
        "label":     "clinical_transformer_softguide_6L",
        "color":     "#d62728",
        "linestyle": "-",
    },
    "lstm_hierarchical_run10": {
        "label":     "lstm_hierarchical_optimized",
        "color":     "#8c564b",
        "linestyle": "-",
    },
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_csv(csv_path, metric_suffix, smooth_window=0):
    """Load a W&B-exported CSV. Returns {run_key: (steps, values)}.

    `metric_suffix` is the suffix W&B appended to the run name, e.g.
    'val_loss' or 'train/loss_step'.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for run_key in SERIES:
        col = f"{run_key} - {metric_suffix}"
        if col not in df.columns:
            raise KeyError(
                f"Column not found: {col!r}\nIn file: {csv_path}\n"
                f"Available columns: {list(df.columns)}"
            )
        sub = df[["Step", col]].dropna()
        steps = sub["Step"].values
        vals = sub[col].values
        if smooth_window > 1:
            vals = (
                pd.Series(vals)
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean()
                .values
            )
        out[run_key] = (steps, vals)
    return out


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot(train_data, val_data):
    plt.rcParams.update({
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "serif",
    })

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 3.8))

    # x-axis range covers both panels
    max_step = max(
        max(s[0].max() for s in train_data.values()),
        max(s[0].max() for s in val_data.values()),
    )

    # --- Training loss panel ---
    for run_key, cfg in SERIES.items():
        steps, vals = train_data[run_key]
        ax_train.plot(
            steps, vals,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=1.4,
        )
    ax_train.set_xlabel("Training step")
    ax_train.set_ylabel("Training loss")
    ax_train.set_title("Training loss")
    ax_train.set_xlim(0, max_step * 1.02)
    ax_train.set_ylim(0, 7.5)
    ax_train.xaxis.set_major_locator(MultipleLocator(2000))
    ax_train.yaxis.set_major_locator(MultipleLocator(1.0))
    ax_train.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)

    # --- Validation loss panel ---
    for run_key, cfg in SERIES.items():
        steps, vals = val_data[run_key]
        ax_val.plot(
            steps, vals,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=1.6,
        )
    ax_val.set_xlabel("Training step")
    ax_val.set_ylabel("Validation loss")
    ax_val.set_title("Validation loss")
    ax_val.set_xlim(0, max_step * 1.02)
    ax_val.set_ylim(1.4, 4.6)
    ax_val.xaxis.set_major_locator(MultipleLocator(2000))
    ax_val.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_val.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)

    # Single shared legend below both panels
    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        handlelength=2.5,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT_PDF}, {OUT_PNG}")


if __name__ == "__main__":
    train_data = load_csv(TRAIN_CSV, "train/loss_step", smooth_window=TRAIN_SMOOTH_WINDOW)
    val_data = load_csv(VAL_CSV, "val_loss")

    print("Training loss:")
    for key, (steps, vals) in train_data.items():
        print(f"  {SERIES[key]['label']}: {len(steps)} points, "
              f"step [{steps.min():.0f}, {steps.max():.0f}], "
              f"loss [{vals.min():.3f}, {vals.max():.3f}]")
    print("Validation loss:")
    for key, (steps, vals) in val_data.items():
        print(f"  {SERIES[key]['label']}: {len(steps)} points, "
              f"step [{steps.min():.0f}, {steps.max():.0f}], "
              f"loss [{vals.min():.3f}, {vals.max():.3f}]")
    plot(train_data, val_data)