
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

BLEU_CSV = HERE / "6L_5model_BLEU4.csv"
F1_CSV   = HERE / "6L_5model_f1macro.csv"

OUT_PDF = HERE / "ablation_6L_bleu4_vs_f1.pdf"
OUT_PNG = HERE / "ablation_6L_bleu4_vs_f1.png"

RUNS = [
    ("D_6L_none",              "No guidance"),
    ("D_6L_soft_finding_only", "LW only\n(soft)"),
    ("D_6L_soft_bias_only",    "DB only\n(soft)"),
    ("D_6L-test",              "Both\n(soft)"),       # the headline D_6L run
    ("D_6L_hardguidance",      "Both\n(hard)"),
]

BLEU_COLOR    = "#1f77b4"   # blue
F1_COLOR      = "#d62728"   # red
HEADLINE_LABEL = "Both\n(soft)"  # bold this x-tick to mark the headline


# ----------------------
def load_metric(csv_path, metric_suffix):
    """Return {run_substring: value} for the given metric suffix."""
    df = pd.read_csv(csv_path)
    out = {}
    for substr, _ in RUNS:
        match = [c for c in df.columns
                 if substr in c and c.endswith(metric_suffix)]
        if not match:
            raise KeyError(
                f"No column in {csv_path.name} matches substring "
                f"{substr!r} ending with {metric_suffix!r}.\n"
                f"Available: {list(df.columns)}"
            )
        if len(match) > 1:
            # D_6L-test substring matches both 'D_6L-test...' and
            # 'D_6L_soft_bias_only-test...'.  Disambiguate by picking the
            # column whose run section ends right before '-test'.
            match = [c for c in match if f"{substr}" in c.split(" - ")[0]]
            if len(match) > 1:
                # Final tiebreak: shortest run name (the bare 'D_6L-test')
                match.sort(key=lambda c: len(c.split(" - ")[0]))
                match = match[:1]
        col = match[0]
        out[substr] = df[col].dropna().iloc[0]
    return out


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot(bleu_vals, f1_vals):
    plt.rcParams.update({
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "font.family":       "serif",
    })

    labels   = [lab for _, lab in RUNS]
    bleu_arr = np.array([bleu_vals[s] for s, _ in RUNS])
    f1_arr   = np.array([f1_vals[s]   for s, _ in RUNS])

    x = np.arange(len(labels))
    bar_w = 0.36

    fig, ax_bleu = plt.subplots(figsize=(7.5, 4.0))
    ax_f1 = ax_bleu.twinx()
    ax_f1.spines["top"].set_visible(False)

    bars_bleu = ax_bleu.bar(
        x - bar_w / 2, bleu_arr, bar_w,
        label="BLEU-4",
        color=BLEU_COLOR, edgecolor="black", linewidth=0.6,
    )
    bars_f1 = ax_f1.bar(
        x + bar_w / 2, f1_arr, bar_w,
        label=r"CheXbert F1$_{\mathrm{macro}}$",
        color=F1_COLOR, edgecolor="black", linewidth=0.6,
    )

    for b in bars_bleu:
        ax_bleu.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                     f"{b.get_height():.3f}",
                     ha="center", va="bottom", fontsize=8, color=BLEU_COLOR)
    for b in bars_f1:
        ax_f1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                   f"{b.get_height():.3f}",
                   ha="center", va="bottom", fontsize=8, color=F1_COLOR)

    ax_bleu.set_xticks(x)
    ax_bleu.set_xticklabels(labels)

    for tick in ax_bleu.get_xticklabels():
        if tick.get_text() == HEADLINE_LABEL:
            tick.set_fontweight("bold")

    ax_bleu.set_ylabel("BLEU-4", color=BLEU_COLOR)
    ax_f1.set_ylabel(r"CheXbert F1$_{\mathrm{macro}}$", color=F1_COLOR)
    ax_bleu.tick_params(axis="y", labelcolor=BLEU_COLOR)
    ax_f1.tick_params(axis="y", labelcolor=F1_COLOR)

    # Y-axis ranges that emphasize the variation without exaggerating it
    ax_bleu.set_ylim(0, max(bleu_arr) * 1.15)
    ax_f1.set_ylim(0,   max(f1_arr)   * 1.15)

    ax_bleu.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.6)

    # Combined legend
    handles = [bars_bleu, bars_f1]
    labels_leg = [h.get_label() for h in handles]
    ax_bleu.legend(handles, labels_leg, loc="upper center",
                   bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT_PDF.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    bleu = load_metric(BLEU_CSV, "test/bleu4")
    f1   = load_metric(F1_CSV,   "test/chexbert_f1_macro")

    print("Loaded values:")
    for substr, label in RUNS:
        print(f"  {label.replace(chr(10), ' '):20s}  BLEU-4={bleu[substr]:.4f}  "
              f"CheXbert-macro={f1[substr]:.4f}")

    plot(bleu, f1)