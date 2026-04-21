# CheXbert Setup

This document walks through installing CheXbert on PACE-ICE and verifying
it produces the same labels as Stanford's reference output. CheXbert is the
labeler we use to compute clinical evaluation metrics (precision/recall/F1
over 14 chest X-ray findings) for all three of our report-generation models.

## What is CheXbert and why are we using it?

CheXbert is a BERT-based labeler that reads a free-text radiology report
and outputs a 14-dimensional label vector indicating which clinical findings
are mentioned and whether they are positive, negative, uncertain, or
unmentioned. The 14 categories are:

1. Enlarged Cardiomediastinum
2. Cardiomegaly
3. Lung Opacity
4. Lung Lesion
5. Edema
6. Consolidation
7. Pneumonia
8. Atelectasis
9. Pneumothorax
10. Pleural Effusion
11. Pleural Other
12. Fracture
13. Support Devices
14. No Finding

Label class convention (same as CheXpert):

| Value | Meaning |
|---|---|
| `0` (or blank) | Not mentioned |
| `1.0` | Positive |
| `0.0` | Negative |
| `-1.0` | Uncertain (hedged language like "may reflect", "possible") |

We use CheXbert (not the original Stanford CheXpert rule-based labeler)
because (a) it's faster, more accurate, and easier to install, and (b) the
original CheXpert labeler depends on Java + NegBio + Python 2, which is a
nightmare on PACE-ICE.

> **Citation (for the final report's evaluation methodology section):**
> Smit, A., Jain, S., Rajpurkar, P., Pareek, A., Ng, A. Y., & Lungren, M. P.
> (2020). CheXbert: Combining Automatic Labelers and Expert Annotations for
> Accurate Radiology Report Labeling Using BERT. *arXiv:2004.09167.*
> https://github.com/stanfordmlgroup/CheXbert

## Prerequisites

- Working `~/scratch/dl-project/venv/` conda env (Python 3.11, torch 2.11.0+cu129)
- Active GT VPN, SSH access to PACE-ICE
- Successfully completed `setup_iu_xray.md` (so you know the env activation pattern works)

In any new shell:
```bash
module load anaconda3/2022.05.0.1
conda activate ~/scratch/dl-project/venv
```

## Step 1 — Clone the CheXbert repo (source only)

We clone the source code but **do NOT install via their `requirements.txt`** —
their pinned versions (`torch==1.4.0`, `transformers==2.5.1`) are 2020-era
and would downgrade your entire environment. We run their code directly on
modern PyTorch with one small patch documented in `PATCHES.md`.

```bash
mkdir -p ~/scratch/dl-project/external
cd ~/scratch/dl-project/external

# Shallow clone — we only need the source, not the full git history.
git clone --depth 1 https://github.com/stanfordmlgroup/CheXbert.git
```

> ⚠️ **Do NOT `git pull` this repo later.** The upstream is a dead research
> codebase. Our local checkout has been patched for compatibility with
> modern `transformers` (see `external/CheXbert/PATCHES.md`); pulling would
> revert the patch.

## Step 2 — Apply the `transformers` 5.x compatibility patch

Modern `transformers` (5.x) removed `BertTokenizer.encode_plus()`, which
CheXbert's tokenizer uses. The fix is a one-line replacement.

```bash
cd ~/scratch/dl-project/external/CheXbert/src

# Verify the broken line is present (you should see one line containing encode_plus)
grep -n "encode_plus" bert_tokenizer.py

# Apply the patch (creates a .bak backup automatically)
sed -i.bak \
  "s|res = tokenizer.encode_plus(tokenized_imp)\['input_ids'\]|res = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenized_imp) + [tokenizer.sep_token_id]|" \
  bert_tokenizer.py

# Verify the patch landed (the encode_plus grep should now find nothing)
grep -n "encode_plus" bert_tokenizer.py
grep -n "cls_token_id" bert_tokenizer.py
```

After patching:
- `grep encode_plus` should return **no matches**.
- `grep cls_token_id` should return **two matches** (line 22, our patch; line 28, original code path for empty reports).

The patch is fully documented in `external/CheXbert/PATCHES.md`. Read it if
you want to understand why the replacement is byte-faithful to what
`encode_plus` was doing internally.

## Step 3 — Install the missing runtime dependency

CheXbert's `utils.py` imports `statsmodels` at the top of the file, so it
needs to be importable even though we won't use the function that requires
it. Just install it.

```bash
pip install statsmodels

# Verify
python -c "from statsmodels.stats.inter_rater import cohens_kappa; print('OK')"
```

The other runtime dependencies (`torch`, `transformers`, `pandas`, `numpy`,
`tqdm`, `scikit-learn`) should already be in the venv from the initial setup.

## Step 4 — Download the pretrained checkpoint

Stanford's original Box link sometimes 404s for non-browser clients. We use
the HuggingFace mirror under the official **StanfordAIMI** institutional
account, which has the same checkpoint with a verifiable SHA256.

```bash
mkdir -p ~/scratch/dl-project/external/CheXbert/weights

python <<'PY'
from huggingface_hub import hf_hub_download
import os, hashlib

local_path = hf_hub_download(
    repo_id="StanfordAIMI/RRG_scorers",
    filename="chexbert.pth",
)
print("downloaded to HF cache:", local_path)
print("size:", os.path.getsize(local_path), "bytes")

# Verify SHA256 matches the canonical checkpoint
expected_sha = "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"
h = hashlib.sha256()
with open(local_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
actual_sha = h.hexdigest()
print("expected sha256:", expected_sha)
print("actual sha256  :", actual_sha)
assert actual_sha == expected_sha, "SHA mismatch — download is corrupt!"
print("SHA verified ✅")

# Symlink into external/CheXbert/weights/ so other scripts can find it
# without duplicating 1.3 GB
target = os.path.expanduser("~/scratch/dl-project/external/CheXbert/weights/chexbert.pth")
if os.path.exists(target) or os.path.islink(target):
    os.remove(target)
os.symlink(local_path, target)
print("symlink created:", target)
PY
```

Expected output:
- A download progress bar (~1.3 GB, takes 1–3 minutes on PACE)
- `size: 1314414442 bytes` (≈1.31 GB)
- `SHA verified ✅`

The download lands in `~/scratch/dl-project/.cache/huggingface/` because
your `HF_HOME` env var points there (per the project setup). We symlink it
to `external/CheXbert/weights/chexbert.pth` so scripts can reference a
clean, predictable path.

## Step 5 — Smoke test on a GPU compute node

This is the **definitive verification** that CheXbert is working on your
stack. We run their `label.py` CLI on the 4 sample reports they bundled,
then `diff` against their reference output. Identical = working.

> **Do not run this on the login node.** GPU work always goes inside `salloc`
> (or `sbatch`). Your TA explicitly bans GPU code on the login node.

Request a GPU:
```bash
salloc --gres=gpu:1 --time=01:00:00 --mem=16G --cpus-per-task=4
```

Inside the salloc session (it's a fresh shell on a compute node, so
re-activate the env):
```bash
module load anaconda3/2022.05.0.1
conda activate ~/scratch/dl-project/venv

# Verify the GPU is real
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run the labeler on Stanford's bundled sample reports
cd ~/scratch/dl-project/external/CheXbert/src
mkdir -p ~/scratch/dl-project/outputs/chexbert_smoke

python label.py \
  -d=sample_reports.csv \
  -o=$HOME/scratch/dl-project/outputs/chexbert_smoke \
  -c=$HOME/scratch/dl-project/external/CheXbert/weights/chexbert.pth

# Diff our output against Stanford's reference — exit code 0 means identical
diff ~/scratch/dl-project/outputs/chexbert_smoke/labeled_reports.csv \
     ~/scratch/dl-project/external/CheXbert/src/labeled_reports.csv
echo "diff exit code: $?"
```

**Success criterion:** `diff exit code: 0` with no output. This means your
labels match Stanford's reference byte-for-byte across all 4 reports × 14
columns.

When done, release the GPU:
```bash
exit
```

Your prompt should change back to a login node (`login-ice-N`).

### Expected warnings (harmless, do not panic)

These appear during the smoke test and are all expected:

1. `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...`
   — HuggingFace nudging you to log in. Public files don't require it.

2. `BertModel LOAD REPORT from: bert-base-uncased ... cls.predictions.* | UNEXPECTED`
   — `bert-base-uncased` includes pretraining heads (MLM, NSP) that
   `BertModel` doesn't use. Reported on every BERT load everywhere. The
   warning literally says *"can be ignored when loading from different
   task/architecture"* — that's our case.

## Troubleshooting

**`ModuleNotFoundError: No module named 'statsmodels'`** — you skipped Step 3.
Run `pip install statsmodels`.

**`AttributeError: 'BertTokenizer' object has no attribute 'encode_plus'`** —
the patch in Step 2 didn't apply. Re-run the `sed` command and re-check
with `grep -n encode_plus bert_tokenizer.py` (should return nothing).

**`FileNotFoundError: ...chexbert.pth`** — the symlink in Step 4 didn't get
created or got removed. Re-run the Step 4 Python block.

**SHA mismatch in Step 4** — your download is corrupt. Delete the cached
file and re-run:
```bash
rm ~/scratch/dl-project/.cache/huggingface/hub/models--StanfordAIMI--RRG_scorers/snapshots/*/chexbert.pth
# then re-run the Step 4 Python block
```

**`diff` shows differences in Step 5** — something is fundamentally wrong.
Most likely the patch in Step 2 was applied incorrectly (different
tokenization → different model output). Check `external/CheXbert/PATCHES.md`
and verify line 22 of `bert_tokenizer.py` matches the patched form exactly.
If you can't resolve it, post in the team channel with the diff output
attached — don't contact PACE support.

**CUDA OOM during smoke test** — extremely unlikely (4 reports × BERT-base
fits in ~2 GB). If it happens, you got a GPU with <4 GB free. `exit` and
`salloc` again, or reduce `BATCH_SIZE` in `external/CheXbert/src/constants.py`
from 18 to 4.

**Don't contact PACE support directly.** Per course policy, all PACE-specific
issues go to the class Ed Discussion thread.

## Storage summary

After successful setup:

```
~/scratch/dl-project/
├── external/
│   └── CheXbert/
│       ├── src/                    (~70 KB, patched)
│       ├── weights/
│       │   └── chexbert.pth        (symlink → HF cache, 1.31 GB on disk)
│       ├── PATCHES.md              (records local modifications)
│       └── README.md, etc.         (Stanford's original)
└── .cache/huggingface/hub/
    └── models--StanfordAIMI--RRG_scorers/
        └── snapshots/<hash>/
            └── chexbert.pth        (1.31 GB, the actual file)
```

Total disk usage for CheXbert: **~1.4 GB**, all on scratch.

## What's next

After CheXbert is verified working, the next setup step is the reusable
Python wrapper module (`src/evaluation/chexbert_labeler.py`) that loads
the model once and exposes a clean API for the rest of the project to call.
That's documented in the project README's Section 2, not here.