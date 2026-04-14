# IU X-Ray Dataset Setup

This document walks through downloading, extracting, and verifying the
**R2Gen-preprocessed Indiana University Chest X-Ray (IU X-Ray)** dataset on
PACE-ICE. 

## Why the R2Gen version?

We use the version of IU X-Ray preprocessed and split by Chen et al. (R2Gen,
EMNLP 2020), not the raw OpenI download. Two reasons:

1. **Comparability.** The R2Gen split (2069 train / 296 val / 590 test) is the
   canonical split used by every recent chest X-ray report generation paper.
   Using it means our BLEU/CIDEr/clinical-F1 numbers can be directly compared
   to published baselines without disclaimers.
2. **One file, one download.** R2Gen ships the images, reports, and split
   assignments in a single Google Drive zip. The raw OpenI download requires
   joining XML metadata to image files manually and inventing your own split.

> **Citation (use in the final report's data section):**
> Chen, Z., Song, Y., Chang, T.-H., & Wan, X. (2020). Generating Radiology
> Reports via Memory-driven Transformer. *Proceedings of EMNLP 2020.*
> https://github.com/zhjohnchan/R2Gen

## Prerequisites

- PACE-ICE account, SSH access via GT VPN
- Project conda env at `~/scratch/dl-project/venv/` with Python 3.11
- Working directory: `~/scratch/dl-project/`

Activate the env in any new shell before running these commands:

```bash
module load anaconda3/2022.05.0.1
conda activate ~/scratch/dl-project/venv
```

## Step 1 — Install `gdown`

`gdown` handles Google Drive's download-confirmation flow from headless SSH
sessions. It's a one-time install into your conda env.

```bash
pip install gdown
which gdown   # should print: ~/scratch/dl-project/venv/bin/gdown
```

## Step 2 — Download in `tmux`

The dataset is ~1.1 GB. `tmux` ensures the download survives an SSH drop.

```bash
mkdir -p ~/scratch/dl-project/data/iu-xray
tmux new -s iuxray
```

Inside the tmux session (it's a fresh shell, so re-activate the env):

```bash
module load anaconda3/2022.05.0.1
conda activate ~/scratch/dl-project/venv
cd ~/scratch/dl-project/data/iu-xray

# File ID is from R2Gen's README:
#   https://github.com/zhjohnchan/R2Gen
gdown 1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg

ls -lh   # should show iu_xray.zip ~1.1G
```

**Tmux tips:** detach with `Ctrl-b` then `d`. Reattach later with
`tmux attach -t iuxray`.

**Possible failure:** Google Drive may say "too many users have downloaded
this file recently" if the quota is tripped. Wait an hour or use the
[Kaggle mirror](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
as a fallback (note: Kaggle's mirror is the *raw* OpenI version, not the R2Gen
split — only use it as a last resort and tell the team).

## Step 3 — Extract

```bash
cd ~/scratch/dl-project/data/iu-xray
unzip -q iu_xray.zip   # ~30 seconds, no output on success

# Quick sanity check before the real verification
ls iu_xray/
# Expected output: annotation.json  images
```

You can leave or kill the tmux session now (`tmux kill-session -t iuxray`).

## Step 4 — Verify with the inspection script

This is the authoritative check. If the script exits cleanly, your dataset is
ready. If it reports any failure, **do not proceed** — fix the dataset first.

```bash
cd ~/scratch/dl-project
python dl-cxr-report-gen/scripts/inspect_iu_xray.py
```

The script is safe to run on the login node (no GPU, no heavy compute, <5 s).
It verifies every check in the "Expected dataset state" table below.

## Expected dataset state

If your numbers don't match exactly, your dataset is wrong.

| Check | Expected value |
|---|---|
| `annotation.json` size | 1,008,692 bytes |
| Top-level keys | `train`, `val`, `test` |
| Train studies | 2,069 |
| Val studies | 296 |
| Test studies | 590 |
| **Total studies** | **2,955** |
| Images referenced in `annotation.json` | 5,910 |
| `.png` files on disk | 6,091 |
| Orphan images (on disk, not referenced) | 181 |
| Entry schema fields | `id`, `report`, `image_path`, `split` |

### About the 181 orphan images

The zip ships 6,091 PNG files but `annotation.json` only references 5,910 of
them. The remaining 181 are studies R2Gen dropped during preprocessing
(missing findings sections, unusable views) but kept in the bundle. Your data
loaders should iterate over `annotation.json`, never over the filesystem
directly, so the orphans are harmless.

### About `XXXX` anonymization tokens

Roughly **40% of reports contain `XXXX`**, which is the IU X-Ray placeholder
for de-identified content (dates, patient names, descriptors). Prevalence by
split:

| Split | Reports with `XXXX` |
|---|---|
| Train | 907 / 2069 (43.8%) |
| Val | 108 / 296 (36.5%) |
| Test | 212 / 590 (35.9%) |

**Decision: leave them in.** Reasons:

1. Every published baseline on this dataset has them in. Removing them would
   break direct comparability.
2. CheXbert's BERT tokenizer splits `XXXX` into low-information subword pieces
   that don't fire any of the 14 finding labels. The findings around the
   `XXXX` are still detected normally.
3. If we ever decide otherwise, stripping is a one-line preprocessing change
   applied uniformly at load time.

This is a **known quirk to mention briefly in the final report's data
section**, not a bug.

## Annotation JSON schema reference

Each entry in `train`, `val`, or `test` has this shape:

```python
{
    "id": "CXR2384_IM-0942",                          # study ID, matches images/CXR2384_IM-0942/
    "report": "The heart size and pulmonary ...",     # full findings/impression text
    "image_path": [                                   # 1 to 3 views per study, mostly 2 (frontal + lateral)
        "CXR2384_IM-0942/0.png",
        "CXR2384_IM-0942/1.png"
    ],
    "split": "train"                                  # redundant with the top-level grouping
}
```

**Multi-view handling:** most studies have exactly 2 views (frontal + lateral).
A few have 1 or 3. The image encoders for our three models will need to
decide how to combine multi-view features (R2Gen concatenates the encoded
features from both views). Coordinate with the team before each model
implementation diverges.

## Storage

After extraction:

```
~/scratch/dl-project/data/iu-xray/
└── iu_xray/
    ├── annotation.json     (986 KB)
    └── images/             (~1.1 GB, 6091 PNG files in 2955 study folders)
```

Total disk: **~1.1 GB**. Living under `scratch/`, never under `home/`.

After verifying the dataset is intact, you can delete the zip:

```bash
rm ~/scratch/dl-project/data/iu-xray/iu_xray.zip
```

## Troubleshooting

**`gdown` says "Permission denied" or "File not found":** the Google Drive
file may have been moved. Check the R2Gen README for the current link.

**`unzip: cannot find or open iu_xray.zip`:** you're not in the right
directory. `cd ~/scratch/dl-project/data/iu-xray` first.

**Inspection script reports image count mismatch:** check `df -h
~/scratch/dl-project/` and `pace-quota` first to rule out a silent
out-of-space failure during extraction. Re-run `unzip -n iu_xray.zip` (no
overwrite) to fill in any missing files.

**Inspection script reports split size mismatch:** your `annotation.json` is
not the R2Gen version. You probably downloaded a different file from
somewhere. Re-run Step 2 with the file ID from this doc.

**Don't contact PACE support directly.** Per course policy, all PACE-specific
issues go to the class Ed Discussion thread.