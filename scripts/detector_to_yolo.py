#!/usr/bin/env python3
"""
detector_to_yolo.py — convert data/detector/ into YOLO format, non-destructively.

data/detector/ is the larger, self-collected dataset (58,361 images) stored as
per-crop folders plus CSVs with absolute XYXY boxes. data/yolo/ is the 5,239-image
Roboflow export in Ultralytics layout. Converting the former lets YOLO and RT-DETR
train on ~11x more data.

The source is never modified: images are symlinked by default (add --copy to
duplicate), and data/detector/ keeps its own CSVs and folder structure.

  data/detector/train/Tomato/<uuid>.jpg      ->  data/detector_yolo/train/images/<uuid>.jpg
  final_train_labels.csv (XYXY absolute)     ->  data/detector_yolo/train/labels/<uuid>.txt

**Class ids are mapped by NAME, never by arithmetic.** The two datasets order their
classes differently — data/detector's label_map.json has 4=Corn Streak while the
YOLO class list has 4=Corn_Streak at a different index — and a naive
`integer_label - 1` mislabels **17 of the 23 classes**. Any class that cannot be
matched by name is a hard error rather than a guess.

Usage
-----
  python scripts/detector_to_yolo.py --dry-run
  python scripts/detector_to_yolo.py
  python scripts/detector_to_yolo.py --copy          # real files instead of symlinks
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# data/detector split name -> Ultralytics split name
SPLITS = {"train": "train", "validate": "valid", "test": "test"}
CSVS = {"train": "final_train_labels.csv",
        "validate": "final_validate_labels.csv",
        "test": "final_test_labels.csv"}


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def build_class_map(src: Path, class_names: list[str]) -> dict[str, int]:
    """detector class name -> YOLO 0-indexed id, matched on name."""
    by_name = {_norm(n): i for i, n in enumerate(class_names)}
    lm = json.loads((src / "label_map.json").read_text())
    mapping, unmapped = {}, []
    for name in lm:
        idx = by_name.get(_norm(name))
        if idx is None:
            unmapped.append(name)
        else:
            mapping[name] = idx
    if unmapped:
        raise SystemExit(f"  x cannot map these classes by name: {unmapped}\n"
                         f"    refusing to guess — fix label_map.json or CLASS_NAMES")
    if len(set(mapping.values())) != len(class_names):
        raise SystemExit("  x mapping is not a bijection onto the YOLO class list")
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/detector")
    ap.add_argument("--dst", default="data/detector_yolo")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (uses ~3.6 GB more disk)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.yolo.config import CLASS_NAMES

    src = PROJECT_ROOT / args.src
    dst = PROJECT_ROOT / args.dst
    if not src.exists():
        raise SystemExit(f"  x source not found: {src}")

    cmap = build_class_map(src, CLASS_NAMES)
    naive_wrong = sum(1 for n, i in cmap.items()
                      if i != json.loads((src / "label_map.json").read_text())[n] - 1)
    print(f"  class map built by name: {len(cmap)} classes "
          f"({naive_wrong} would be wrong under a naive id-1)")

    totals, per_class = {}, Counter()
    plan: list[tuple[Path, Path, str]] = []   # (image_src, image_dst, label_text)
    problems: list[str] = []

    for split, out_split in SPLITS.items():
        rows = list(csv.DictReader((src / CSVS[split]).open()))
        by_img: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_img[r["fname"]].append(r)

        for fname, recs in by_img.items():
            crop = recs[0]["crop"]
            img_src = src / split / crop / fname
            if not img_src.exists():
                problems.append(f"missing image {img_src}")
                continue
            lines = []
            for r in recs:
                w, h = float(r["width"]), float(r["height"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                x2, y2 = float(r["x2"]), float(r["y2"])
                # clamp to the image, then convert XYXY(abs) -> cxcywh(normalised)
                x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    problems.append(f"degenerate box in {fname}")
                    continue
                cid = cmap[r["class"]]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    problems.append(f"out-of-range box in {fname}")
                    continue
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                per_class[cid] += 1
            if lines:
                plan.append((img_src, dst / out_split / "images" / fname, "\n".join(lines)))
        totals[out_split] = sum(1 for p in plan if f"/{out_split}/" in str(p[1]))

    print("  images per split:", {k: v for k, v in totals.items()})
    if problems:
        print(f"  ! {len(problems)} problem rows (first 5):")
        for p in problems[:5]:
            print(f"      {p}")

    if args.dry_run:
        print(f"  dry run — would write {len(plan):,} images + labels to {dst}")
        return 0

    for out_split in SPLITS.values():
        (dst / out_split / "images").mkdir(parents=True, exist_ok=True)
        (dst / out_split / "labels").mkdir(parents=True, exist_ok=True)

    n = 0
    for img_src, img_dst, label in plan:
        if not img_dst.exists():
            if args.copy:
                shutil.copy2(img_src, img_dst)
            else:
                img_dst.symlink_to(img_src.resolve())
        lbl = img_dst.parent.parent / "labels" / (img_dst.stem + ".txt")
        lbl.write_text(label + "\n")
        n += 1
        if n % 10000 == 0:
            print(f"    {n:,}/{len(plan):,}")

    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        f"path: {dst}\n"
        f"train: {dst / 'train' / 'images'}\n"
        f"val: {dst / 'valid' / 'images'}\n"
        f"test: {dst / 'test' / 'images'}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n" + "".join(f"  - {n}\n" for n in CLASS_NAMES)
    )

    print(f"  wrote {n:,} image/label pairs to {dst}")
    print(f"  wrote {yaml_path}")
    print(f"  source {src} untouched "
          f"({'copies' if args.copy else 'symlinks'} used for images)")
    print("\n  class distribution (YOLO ids):")
    for cid, c in sorted(per_class.items()):
        print(f"    {cid:2d} {CLASS_NAMES[cid]:<30} {c:6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
