#!/usr/bin/env python3
"""
calibrate_threshold.py — pick CONF_DEFAULT from data instead of guessing it.

The classifier is a 3-class softmax (Corn / Pepper / Tomato) with no "not a crop"
class, so rejection is done purely by thresholding the top softmax probability:

    if conf < CONF_DEFAULT: return "unknown"

CONF_DEFAULT is currently 0.55, which was never measured. This script sweeps the
threshold and reports the two rates that matter, so the value can be chosen against
a real operating point:

  crop retention  — share of genuine crop images still accepted AND classified
                    correctly. Raising the threshold costs you these.
  OOD rejection   — share of non-crop images correctly sent to "unknown". Raising
                    the threshold wins you these.

Note on what this can and cannot tell you: the negatives in data/negatives/ood are
real-world photographs (objects, interiors, people, landscapes), so they measure
rejection of obviously-not-a-leaf inputs. They do NOT contain the near-miss case —
a different crop's leaf shot in a field — which is the harder and more realistic
failure. Treat the OOD rejection number as an upper bound on real-world behaviour.

Usage
-----
  python -m src.classifier.calibrate_threshold
  python -m src.classifier.calibrate_threshold --ood-dir data/negatives/ood
  python -m src.classifier.calibrate_threshold --checkpoint outputs/classifier_output/best.pth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.classifier.config import (CONF_DEFAULT, CROP_CLASSES, DATASET_DIR,
                                   IMG_SIZE, OUTPUT_DIR, PROJECT_ROOT)
from src.classifier.train_classifier import build_model

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageListDataset(Dataset):
    """Plain image list; label is -1 for out-of-distribution images."""

    def __init__(self, paths: list[Path], labels: list[int]):
        self.paths, self.labels = paths, labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return EVAL_TF(img), self.labels[i]


def load_test_split(csv_path: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            p = PROJECT_ROOT / row["image_path"]
            if p.exists():
                paths.append(p)
                labels.append(int(row["crop_id"]))
    return paths, labels


@torch.no_grad()
def confidences(model, paths, labels, device, batch=128):
    """Return (top_confidence, predicted_class, true_label) per image."""
    ld = DataLoader(ImageListDataset(paths, labels), batch_size=batch,
                    shuffle=False, num_workers=8, pin_memory=(device.type == "cuda"))
    conf_all, pred_all, true_all = [], [], []
    for imgs, lbls in ld:
        imgs = imgs.to(device, non_blocking=True)
        probs = torch.softmax(model(imgs), dim=1)
        c, p = probs.max(dim=1)
        conf_all += c.cpu().tolist()
        pred_all += p.cpu().tolist()
        true_all += lbls.tolist()
    return conf_all, pred_all, true_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(OUTPUT_DIR / "best.pth"))
    ap.add_argument("--ood-dir", default="data/negatives/ood")
    ap.add_argument("--test-csv", default=str(DATASET_DIR / "classifier_test.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")

    # ── In-distribution: the real test split ─────────────────────────────────
    paths, labels = load_test_split(Path(args.test_csv))
    if not paths:
        raise SystemExit(f"  x no test images found via {args.test_csv}")
    id_conf, id_pred, id_true = confidences(model, paths, labels, device)
    print(f"  crop test  : {len(paths):,} images")

    # ── Out-of-distribution: non-crop photographs ────────────────────────────
    ood_dir = PROJECT_ROOT / args.ood_dir
    ood_paths = sorted(p for p in ood_dir.glob("*.jpg") if not p.name.startswith("._"))
    if not ood_paths:
        raise SystemExit(f"  x no OOD images in {ood_dir} — run scripts/fetch_ood_negatives.py")
    ood_conf, ood_pred, _ = confidences(model, ood_paths, [-1] * len(ood_paths), device)
    print(f"  OOD images : {len(ood_paths):,}")

    # What the model *thinks* the non-crop images are — a 3-class softmax has to
    # put them somewhere, so this shows which class absorbs them.
    print()
    print("  Non-crop images are forced into these classes:")
    for i, name in enumerate(CROP_CLASSES):
        n = sum(1 for p in ood_pred if p == i)
        print(f"    {name:<8} {n:5,}  ({n / len(ood_pred) * 100:5.1f}%)")
    over_half = sum(1 for c in ood_conf if c >= 0.5)
    print(f"  ...and {over_half:,} of {len(ood_conf):,} "
          f"({over_half / len(ood_conf) * 100:.1f}%) do so with >=50% confidence.")

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print()
    print("  threshold   crop retention   OOD rejection")
    print("  ---------   --------------   -------------")
    best = None
    for thr in [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                0.80, 0.85, 0.90, 0.95, 0.99]:
        keep = sum(1 for c, p, t in zip(id_conf, id_pred, id_true)
                   if c >= thr and p == t) / len(id_conf)
        rej = sum(1 for c in ood_conf if c < thr) / len(ood_conf)
        mark = "   <- current" if abs(thr - CONF_DEFAULT) < 1e-9 else ""
        print(f"     {thr:.2f}        {keep * 100:6.2f}%          {rej * 100:6.2f}%{mark}")
        # Balanced pick: maximise retention + rejection jointly.
        score = keep + rej
        if best is None or score > best[0]:
            best = (score, thr, keep, rej)

    _, thr, keep, rej = best
    print()
    print(f"  Best balanced threshold: {thr:.2f} "
          f"(retains {keep * 100:.1f}% of crops, rejects {rej * 100:.1f}% of non-crops)")
    cur_keep = sum(1 for c, p, t in zip(id_conf, id_pred, id_true)
                   if c >= CONF_DEFAULT and p == t) / len(id_conf)
    cur_rej = sum(1 for c in ood_conf if c < CONF_DEFAULT) / len(ood_conf)
    print(f"  Current CONF_DEFAULT={CONF_DEFAULT}: "
          f"retains {cur_keep * 100:.1f}%, rejects {cur_rej * 100:.1f}%")


if __name__ == "__main__":
    main()
