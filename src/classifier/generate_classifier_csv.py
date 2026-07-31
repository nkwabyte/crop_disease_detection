#!/usr/bin/env python3
"""
generate_classifier_csv.py — Build crop-type classification CSVs from the YOLO dataset.

Reads every YOLO label file in train / valid / test splits and maps the
fine-grained disease class IDs to the three parent crop types:

  Corn   (classes 0–4)
  Pepper (classes 5–14)
  Tomato (classes 15–22)

Output files (written to dataset/):
  classifier_train.csv
  classifier_valid.csv
  classifier_test.csv

Each CSV has columns:
  image_path   — path relative to project root
  crop_label   — "Corn", "Pepper", or "Tomato"
  crop_id      — 0, 1, or 2 (integer label)
  split        — "train", "valid", or "test"

Images whose label files are missing or empty (hard negatives) are skipped.
If a single image has bounding boxes from more than one crop type (rare edge
case) the majority vote wins; ties go to the first crop found.

Usage
-----
  python generate_classifier_csv.py
  python generate_classifier_csv.py --data-dir data/main --out-dir dataset
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

# ── Class-ID → Crop mapping ─────────────────────────────────────────────────

YOLO_CLASSES = [
    "Corn_Cercospora_Leaf_Spot",   # 0
    "Corn_Common_Rust",            # 1
    "Corn_Healthy",                # 2
    "Corn_Northern_Leaf_Blight",   # 3
    "Corn_Streak",                 # 4
    "Pepper_Bacterial_Spot",       # 5
    "Pepper_Cercospora",           # 6
    "Pepper_Early_Blight",         # 7
    "Pepper_Fusarium",             # 8
    "Pepper_Healthy",              # 9
    "Pepper_Late_Blight",          # 10
    "Pepper_Leaf_Blight",          # 11
    "Pepper_Leaf_Curl",            # 12
    "Pepper_Leaf_Mosaic",          # 13
    "Pepper_Septoria",             # 14
    "Tomato_Bacterial_Spot",       # 15
    "Tomato_Early_Blight",         # 16
    "Tomato_Fusarium",             # 17
    "Tomato_Healthy",              # 18
    "Tomato_Late_Blight",          # 19
    "Tomato_Leaf_Curl",            # 20
    "Tomato_Mosaic",               # 21
    "Tomato_Septoria",             # 22
]

# Derived automatically from the class name prefix
ID_TO_CROP: dict[int, str] = {
    i: name.split("_")[0] for i, name in enumerate(YOLO_CLASSES)
}

CROP_TO_ID: dict[str, int] = {"Corn": 0, "Pepper": 1, "Tomato": 2}


# ── Core logic ───────────────────────────────────────────────────────────────

def crop_from_label_file(label_path: Path) -> str | None:
    """Return the majority-vote crop type for one YOLO label file, or None."""
    if not label_path.exists():
        return None

    votes: list[str] = []
    with label_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cls_id = int(line.split()[0])
                crop = ID_TO_CROP.get(cls_id)
                if crop:
                    votes.append(crop)
            except (ValueError, IndexError):
                continue

    if not votes:
        return None

    # Majority vote; Counter.most_common preserves insertion order on tie
    return Counter(votes).most_common(1)[0][0]


def process_split(
    images_dir: Path,
    labels_dir: Path,
    split: str,
    project_root: Path,
) -> list[dict]:
    rows: list[dict] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        crop = crop_from_label_file(label_path)
        if crop is None:
            continue  # skip hard negatives and unlabelled images

        rows.append({
            "image_path": str(img_path.relative_to(project_root)),
            "crop_label": crop,
            "crop_id":    CROP_TO_ID[crop],
            "split":      split,
        })

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "crop_label", "crop_id", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict], split: str) -> None:
    counts = Counter(r["crop_label"] for r in rows)
    total  = len(rows)
    print(f"\n  {split:6s}  ({total} images)")
    for crop in ["Corn", "Pepper", "Tomato"]:
        n = counts.get(crop, 0)
        bar = "█" * (n * 30 // max(total, 1))
        print(f"    {crop:<8}  {n:5d}  {bar}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/main",
        help="Root of the YOLO dataset (contains train/, valid/, test/)",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset",
        help="Directory to write the classifier CSV files into",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir     = project_root / args.data_dir
    out_dir      = project_root / args.out_dir

    splits = [
        ("train", "train"),
        ("valid", "valid"),
        ("test",  "test"),
    ]

    print("Generating classifier CSVs …")
    all_rows: list[dict] = []

    for split_name, folder in splits:
        images_dir = data_dir / folder / "images"
        labels_dir = data_dir / folder / "labels"

        if not images_dir.exists():
            print(f"  [WARN] {images_dir} not found — skipping {split_name}")
            continue

        rows = process_split(images_dir, labels_dir, split_name, project_root)
        out_path = out_dir / f"classifier_{split_name}.csv"
        write_csv(rows, out_path)
        print_stats(rows, split_name)
        print(f"  → {out_path}")
        all_rows.extend(rows)

    print(f"\nDone. {len(all_rows)} total labelled images written.")


if __name__ == "__main__":
    main()
