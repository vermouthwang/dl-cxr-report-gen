# dl-cxr-report-gen

Chest X-ray report generation: comparing three architectures on the Indiana University Chest X-Ray dataset.

**Course project — Georgia Tech, Deep Learning, Spring 2026.**

## Project overview

We build and compare three models for generating radiology reports from chest X-ray images:

1. **Hierarchical LSTM baseline** — encoder-decoder with a two-level LSTM decoder (sentence-level + word-level).
2. **Vanilla Transformer** — CNN encoder + Transformer decoder, no clinical priors.
3. **Clinical-term-guided Transformer** — same as (2) but conditioned on predicted clinical terms (tags) to improve factual accuracy.

All three are trained and evaluated on the **Indiana University Chest X-Ray dataset** (~7,470 images, open access).

### Evaluation

- **Linguistic metrics:** BLEU-1/2/3/4, METEOR, CIDEr (via `pycocoevalcap`)
- **Clinical metrics:** CheXbert F1 over 14 findings (measures whether the generated report mentions the right pathologies, not just fluent-sounding text)


## Repository structure

dl-cxr-report-gen/
├── src/             # Model code, data loaders, training utilities
├── configs/         # YAML configs for experiments (one per run)
├── scripts/         # Setup scripts + SLURM batch scripts
├── notebooks/       # Exploratory / sanity-check notebooks (not for training)
├── docs/            # Design docs, interface specs, diagrams
├── README.md
├── requirements.txt
└── .gitignore


## Getting started

### Prerequisites

- Georgia Tech PACE-ICE account ([request here](https://pace.gatech.edu/participant-information))
- GT VPN connection ([setup instructions](https://vpn.gatech.edu))
- GitHub account with access to this repo (ask a teammate to add you as a collaborator)
- Weights & Biases account (free for students) — join the team project at `https://wandb.ai/<team-name>/dl-cxr-report-gen` (link to be added once created)

### First-time setup on PACE

Full instructions live in [`docs/pace_setup.md`](docs/pace_setup.md) (to be added). Short version:
```bash
# 1. Connect to GT VPN, then SSH to PACE
ssh <gt-username>@login-ice.pace.gatech.edu

# 2. Clone the repo into scratch (NOT home directory — quota is tiny)
cd ~/scratch
mkdir -p dl-project && cd dl-project
git clone git@github.com:<org-or-user>/dl-cxr-report-gen.git
cd dl-cxr-report-gen

# 3. Run the setup script (creates venv, installs deps, sets env vars)
bash scripts/setup_pace.sh

# 4. Smoke-test GPU access via SLURM
sbatch scripts/test_gpu.sbatch
```

### Critical rules for working on PACE

- **Never run training on the login node.** Always use `salloc` (interactive) or `sbatch` (batch).
- **Never install packages or store data in `$HOME`.** The home quota is tiny (~10GB). Everything goes in `~/scratch/dl-project/`.
- **Never commit data, checkpoints, or W&B logs.** The `.gitignore` should prevent this, but double-check your `git status` before committing.
- **If something breaks with PACE itself, ask in the class Ed Discussion thread.** Do not contact PACE support directly (TA instruction).

## Conventions

### Branching strategy

We use a lightweight feature-branch workflow:

- `main` — always works. Protected; no direct pushes.
- `feat/<person>-<short-desc>` — feature branches (e.g., `feat/a-lstm-decoder`, `feat/b-transformer-trainer`).
- `fix/<short-desc>` — bug fixes.
- `exp/<person>-<experiment>` — long-running experiment branches that may or may not merge.

**Workflow:**
1. Pull latest `main` before starting: `git pull origin main`
2. Create a feature branch: `git checkout -b feat/a-data-pipeline`
3. Commit often with descriptive messages.
4. Open a pull request when ready; request review from at least one teammate.
5. Squash-merge into `main` once approved.

### Commit message style

Keep them short and descriptive. Prefix with a tag when useful:

- **feat: add IU X-Ray tokenizer
- **fix: handle missing report field in JSON
- **docs: update interface spec for decoder
- **exp: try larger batch size on LSTM

### W&B run naming

All runs are logged to the shared W&B project. Use this naming convention so runs are sortable and attributable:
<model><person><short-tag>_<YYYYMMDD>

Examples:
- `lstm_a_baseline_20260410`
- `transformer_b_bs32_20260412`
- `clinical-tf_c_tagloss-0.5_20260415`

Tag runs with the model family (`lstm`, `transformer`, `clinical-tf`) so the W&B dashboard can filter cleanly.

### Shared code interfaces

To let all three models share the same training script and data loader, we've locked down the model and batch interfaces on Day 1. See [`docs/interfaces.md`](docs/interfaces.md) for the spec. **Do not deviate without team agreement** — breaking the interface breaks everyone's code.

## Communication

- **Daily / async:** Discord (channel: `#dl-project`)
- **Weekly sync:** Fridays, 4pm ET (video call)
- **Task tracking:** GitHub Issues (one issue per task, assigned to a person, tagged by week)
- **Experiment results:** shared W&B dashboard

## Timeline (rough)

| Week | Milestone |
|------|-----------|
| 1 | Environment setup, data pipeline, LSTM baseline skeleton |
| 2 | All 3 models trainable end-to-end; first full training runs |
| 3 | Hyperparameter tuning, final runs (burst to GCP if needed), evaluation |
| 4 | Ablations, report writing, final presentation |

## Acknowledgements

- Indiana University Chest X-Ray dataset (Demner-Fushman et al., 2016)
- CheXbert labeler (Smit et al., 2020)
- `pycocoevalcap` for BLEU/METEOR/CIDEr implementations



