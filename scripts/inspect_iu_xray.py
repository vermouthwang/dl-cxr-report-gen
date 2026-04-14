#!/usr/bin/env python3
"""
This is a script for inspecting and verifying the R2Gen-preprocessed IU X-Ray dataset.
You should expect printed all the checks to pass.
  ✅ train: 2069 studies
  ✅ val: 296 studies
  ✅ test: 590 studies
  ✅ total: 2955 studies
Verifies:
  1. annotation.json exists, parses, and has the expected top-level structure
  2. Train/val/test split sizes match the canonical R2Gen split
  3. Every study referenced in annotation.json has its image files on disk
  4. The schema of each entry is what downstream code expects
  5. Reports a few random ground-truth reports for visual inspection
  6. Reports the prevalence of 'XXXX' anonymization tokens

Safe to run on the PACE login node: no GPU, no heavy compute, <5 seconds.

Usage:
    python scripts/inspect_iu_xray.py
    python scripts/inspect_iu_xray.py --data-root /custom/path/to/iu_xray
    python scripts/inspect_iu_xray.py --num-samples 5 --seed 0

Exits 0 on success, 1 on any verification failure. 
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical R2Gen split for IU X-Ray (Chen et al., EMNLP 2020).
# These are the ground-truth numbers we verify against.
# Source: https://github.com/zhjohnchan/R2Gen + verified on disk 2026-04-13.
# ---------------------------------------------------------------------------
EXPECTED_SPLITS = {"train": 2069, "val": 296, "test": 590}
EXPECTED_TOTAL_STUDIES = sum(EXPECTED_SPLITS.values())  # 2955
EXPECTED_TOTAL_IMAGES = 6091  # all .png files under images/, all splits
REQUIRED_FIELDS = {"id", "report", "image_path", "split"}

# Default data root, relative to the project root. Override with --data-root.
DEFAULT_DATA_ROOT = Path("data/iu-xray/iu_xray")


def fail(msg: str) -> None:
    """Print an error and exit non-zero."""
    print(f"  ❌ FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def load_annotation(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    ann_path = data_root / "annotation.json"
    if not ann_path.exists():
        fail(f"annotation.json not found at {ann_path}")
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"annotation.json is not valid JSON: {e}")
    ok(f"annotation.json loaded ({ann_path.stat().st_size:,} bytes)")
    return ann


def check_top_level_structure(ann: dict) -> None:
    keys = set(ann.keys())
    expected = set(EXPECTED_SPLITS.keys())
    if keys != expected:
        fail(f"top-level keys are {sorted(keys)}, expected {sorted(expected)}")
    ok(f"top-level keys: {sorted(keys)}")


def check_split_sizes(ann: dict) -> None:
    for split, expected_n in EXPECTED_SPLITS.items():
        actual_n = len(ann[split])
        if actual_n != expected_n:
            fail(f"{split} has {actual_n} entries, expected {expected_n}")
        ok(f"{split}: {actual_n} studies")
    total = sum(len(ann[s]) for s in ann)
    if total != EXPECTED_TOTAL_STUDIES:
        fail(f"total studies = {total}, expected {EXPECTED_TOTAL_STUDIES}")
    ok(f"total: {total} studies")


def check_entry_schema(ann: dict) -> None:
    """Validate that every entry has the fields downstream code expects."""
    for split, entries in ann.items():
        for i, entry in enumerate(entries):
            missing = REQUIRED_FIELDS - set(entry.keys())
            if missing:
                fail(f"{split}[{i}] (id={entry.get('id', '?')}) missing fields: {missing}")
            if not isinstance(entry["image_path"], list) or not entry["image_path"]:
                fail(f"{split}[{i}] (id={entry['id']}) has non-list or empty image_path")
            if not isinstance(entry["report"], str) or not entry["report"].strip():
                fail(f"{split}[{i}] (id={entry['id']}) has empty or non-string report")
    ok(f"all {EXPECTED_TOTAL_STUDIES} entries have required fields {sorted(REQUIRED_FIELDS)}")


def check_images_on_disk(ann: dict, data_root: Path) -> None:
    """Verify every image referenced in annotation.json exists on disk."""
    images_dir = data_root / "images"
    if not images_dir.is_dir():
        fail(f"images directory not found at {images_dir}")

    referenced: set[Path] = set()
    for entries in ann.values():
        for entry in entries:
            for rel in entry["image_path"]:
                referenced.add(images_dir / rel)

    missing = [p for p in referenced if not p.exists()]
    if missing:
        fail(f"{len(missing)} images referenced in annotation.json are missing on disk. "
             f"First 3: {missing[:3]}")
    ok(f"all {len(referenced)} referenced images exist on disk")

    # Also report orphan images (on disk but not referenced) — informational.
    on_disk = set(images_dir.rglob("*.png"))
    orphans = on_disk - referenced
    if orphans:
        print(f"  ℹ️  {len(orphans)} images on disk are not referenced in annotation.json "
              f"(harmless, just unused)")
    if len(on_disk) != EXPECTED_TOTAL_IMAGES:
        print(f"  ℹ️  found {len(on_disk)} images on disk, "
              f"expected {EXPECTED_TOTAL_IMAGES} (informational)")


def show_random_reports(ann: dict, num: int, seed: int) -> None:
    rng = random.Random(seed)
    samples = rng.sample(ann["train"], num)
    for s in samples:
        print(f"\n  --- {s['id']} (views: {len(s['image_path'])}) ---")
        # Wrap long reports for readability
        report = s["report"]
        print(f"  {report}")


def report_xxxx_prevalence(ann: dict) -> None:
    for split, entries in ann.items():
        n_xxxx = sum(1 for e in entries if "XXXX" in e["report"])
        pct = 100 * n_xxxx / len(entries)
        print(f"  {split}: {n_xxxx} / {len(entries)} ({pct:.1f}%) reports contain 'XXXX'")
    print("  Note: XXXX is the IU X-Ray anonymization placeholder. Left in by")
    print("  default for comparability with published baselines (R2Gen et al.).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Path to the iu_xray/ directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="How many random reports to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    if not args.data_root.is_dir():
        fail(f"--data-root {args.data_root} is not a directory")

    print(f"Inspecting IU X-Ray dataset at: {args.data_root.resolve()}")

    section("1. Load annotation.json")
    ann = load_annotation(args.data_root)

    section("2. Top-level structure")
    check_top_level_structure(ann)

    section("3. Split sizes (vs. canonical R2Gen split)")
    check_split_sizes(ann)

    section("4. Entry schema")
    check_entry_schema(ann)

    section("5. Image files on disk")
    check_images_on_disk(ann, args.data_root)

    section(f"6. Random sample of {args.num_samples} train reports")
    show_random_reports(ann, args.num_samples, args.seed)

    section("7. XXXX anonymization prevalence")
    report_xxxx_prevalence(ann)

    print("\n✅ All checks passed. Dataset is ready to use.")


if __name__ == "__main__":
    main()