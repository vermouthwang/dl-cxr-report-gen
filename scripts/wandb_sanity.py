"""
scripts/wandb_sanity.py
-----------------------
Minimal "hello world" W&B run. Logs 10 steps of fake metrics to verify that:
  1. wandb is installed and importable
  2. the API key is configured (via `wandb login`)
  3. runs show up in the shared team project

Run with:
    python scripts/wandb_sanity.py

No GPU required. Takes ~5 seconds.
"""

import math
import random
import time

import wandb

WANDB_ENTITY = "yinghou-georgia-institute-of-technology"
WANDB_PROJECT = "dl-cxr-report-gen"


def main() -> None:
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"sanity_check_{int(time.time())}",
        tags=["sanity", "setup", "day1"],
        notes="Day 1 sanity check — verifies W&B auth and the team project are wired up.",
        config={
            "script": "wandb_sanity.py",
            "purpose": "verify wandb logging works end-to-end",
            "fake_lr": 3e-4,
            "fake_batch_size": 32,
        },
    )

    print(f"Started run: {run.name}")
    print(f"View at:     {run.get_url()}")

    # Log 10 steps of deliberately fake, clearly-not-real metrics.
    for step in range(10):
        fake_loss = 2.0 * math.exp(-step / 3.0) + random.uniform(-0.05, 0.05)
        fake_bleu = 0.1 + 0.03 * step + random.uniform(-0.01, 0.01)
        wandb.log(
            {
                "train/loss": fake_loss,
                "val/bleu_4": fake_bleu,
                "step": step,
            }
        )
        time.sleep(0.2)  # pretend we're doing work

    wandb.finish()
    print("Sanity check complete. Go check the dashboard.")


if __name__ == "__main__":
    main()