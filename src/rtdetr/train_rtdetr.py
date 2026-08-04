#!/usr/bin/env python3
"""
train_rtdetr.py — RT-DETR crop-disease detector (Ultralytics, transformer query head).

RT-DETR is a query-based transformer detector: no anchors and no NMS. Unlike the
region-based Faster R-CNN heads (used by the ViT and Swin detectors here), its
inference is static-shape, which makes it the most ExecuTorch-friendly *full model*
of the transformer detectors — a strong candidate for the mobile app.

It trains on the same YOLO-format dataset (data/yolo/) as YOLO26, with the same
0-indexed class order, so it slots straight into the benchmark on mAP@0.5.

This file is SELF-CONTAINED: the data-yaml writer and hard-negative staging are
duplicated here rather than imported from src/yolo, so the packages stay decoupled.

Usage
-----
  python -m src.rtdetr.train_rtdetr                    # full pipeline
  python -m src.rtdetr.train_rtdetr --dry-run          # 1-epoch timing estimate
  python -m src.rtdetr.train_rtdetr --skip-negatives   # negatives already staged
  python -m src.rtdetr.train_rtdetr --epochs 50        # override epoch count
  python -m src.rtdetr.train_rtdetr --batch-size 8     # override batch (e.g. on CUDA)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from src.rtdetr.config import (
    PROJECT_ROOT, DATA_DIR, NEG_DIR, RTDETR_YAML, RUNS_DIR, EXP_NAME,
    BENCHMARK_DIR, MODEL_SIZE, IMG_SIZE, BASE_BATCH, CUDA_BATCH, EPOCHS_DEFAULT, PATIENCE,
    CONF_THRESHOLD, IOU_THRESHOLD, NUM_NEGATIVES, CLASS_NAMES,
)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — data yaml
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data_yaml() -> None:
    """Write data_rtdetr.yaml with absolute paths. Safe to call repeatedly."""
    cfg = {
        "path":  str(DATA_DIR),
        "train": str(DATA_DIR / "train" / "images"),
        "val":   str(DATA_DIR / "valid" / "images"),
        "test":  str(DATA_DIR / "test"  / "images"),
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(RTDETR_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  data_rtdetr.yaml → {RTDETR_YAML}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Hard negatives
# ══════════════════════════════════════════════════════════════════════════════

def prepare_hard_negatives(num: int = NUM_NEGATIVES, skip: bool = False) -> None:
    """Download diverse non-crop images and stage them (empty labels) in the training
    split so RT-DETR learns zero detections on OOD imagery. Fully resumable."""
    if skip:
        print("  Hard-negative preparation skipped (--skip-negatives).")
        return

    neg_img_dir = NEG_DIR / "images"
    neg_img_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    seeds = random.sample(range(1, 2000), num)

    pending = [
        (seed, neg_img_dir / f"negative_{seed:04d}.jpg")
        for seed in seeds
        if not (neg_img_dir / f"negative_{seed:04d}.jpg").exists()
    ]
    already = num - len(pending)
    print(f"  Hard negatives: {already}/{num} cached, {len(pending)} to download")

    if pending:
        def _fetch(args):
            seed, dest = args
            url = f"https://picsum.photos/seed/{seed}/640/640"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    dest.write_bytes(r.read())
                return seed, None
            except Exception as exc:
                return seed, str(exc)

        ok = err = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch, item): item for item in pending}
            for done, future in enumerate(as_completed(futures), 1):
                _, exc = future.result()
                ok += exc is None
                err += exc is not None
                if done % 50 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} fetched  (ok={ok}, errors={err})")
        print(f"  Download complete: {ok} new, {err} failed")

    train_img_dir = DATA_DIR / "train" / "images"
    train_lbl_dir = DATA_DIR / "train" / "labels"
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in neg_img_dir.glob("*.jpg"):
        dst = train_img_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
            (train_lbl_dir / img.with_suffix(".txt").name).touch()
            copied += 1
    staged = len(list(train_img_dir.glob("negative_*.jpg")))
    print(f"  Training split: {staged} negatives staged ({copied} newly added)")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device():
    import torch
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_last_ckpt() -> Path | None:
    last = RUNS_DIR / EXP_NAME / "weights" / "last.pt"
    return last if last.exists() else None


def write_benchmark_summary(model, metrics) -> None:
    """Persist RT-DETR final metrics for the cross-model benchmark aggregator."""
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        n_params = None
    map50 = None
    per_class = {}
    try:
        map50 = float(metrics.box.map50)
        names = metrics.names if hasattr(metrics, "names") else {i: n for i, n in enumerate(CLASS_NAMES)}
        ap50 = getattr(metrics.box, "ap50", None)
        if ap50 is not None:
            for i, ap in enumerate(list(ap50)):
                per_class[names.get(i, str(i))] = round(float(ap), 5)
    except Exception as exc:
        print(f"  [WARN]  could not read RT-DETR metrics: {exc}")

    payload = {
        "model_name": "rtdetr_l",
        "architecture": "RT-DETR-L (Ultralytics, transformer query head)",
        "split": "val",
        "map50": None if map50 is None else round(map50, 5),
        "num_params": n_params,
        "per_class_ap": per_class,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "final_eval.json", "w") as f:
        json.dump(payload, f, indent=2)
    try:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_DIR / "rtdetr_l.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"  [WARN]  could not write shared benchmark summary: {exc}")
    print(f"  Final eval saved → {RUNS_DIR / 'final_eval.json'}  (mAP@0.5={payload['map50']})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETR crop-disease detector")
    parser.add_argument("--dry-run", action="store_true", help="1-epoch timing estimate")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--skip-negatives", action="store_true", help="skip hard-negative staging")
    args = parser.parse_args()

    from ultralytics import RTDETR

    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    epochs  = args.epochs if args.epochs is not None else (1 if dry_run else EPOCHS_DEFAULT)
    device  = resolve_device()
    is_cuda = device == 0 or (isinstance(device, str) and device.startswith("cuda"))
    batch   = args.batch_size if args.batch_size is not None else (CUDA_BATCH if is_cuda else BASE_BATCH)
    workers = 0 if device == "mps" else min(16 if is_cuda else 8, os.cpu_count() or 1)

    print("Step 1 — data yaml");        prepare_data_yaml()
    print("Step 2 — hard negatives");   prepare_hard_negatives(skip=args.skip_negatives)

    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  Model      : RT-DETR ({MODEL_SIZE})  — transformer query head, no NMS")
    print(f"  Device     : {device}   workers={workers}")
    print(f"  Batch      : {batch}    Image size: {IMG_SIZE}")
    print(f"  Epochs     : {epochs}" + ("  ← DRY RUN" if dry_run else ""))
    print(f"{sep}\n")

    last = _find_last_ckpt()
    t0 = time.perf_counter()
    if last:
        print(f"  Resuming from {last}")
        model = RTDETR(str(last))
        model.train(resume=True)
    else:
        model = RTDETR(f"{MODEL_SIZE}.pt")   # downloads pretrained weights on first run
        model.train(
            data=str(RTDETR_YAML), imgsz=IMG_SIZE, batch=batch, epochs=epochs,
            patience=PATIENCE, device=device, workers=workers,
            project=str(RUNS_DIR), name=EXP_NAME, exist_ok=True, seed=42,
        )
    print(f"\n  Training complete in {(time.perf_counter() - t0) / 60:.1f} min")

    # Ultralytics auto-saves training curves + confusion matrix + PR curves under
    # the run directory; we add the cross-model benchmark summary.
    print("\n  Validating best checkpoint …")
    metrics = model.val(data=str(RTDETR_YAML), imgsz=IMG_SIZE, device=device)
    write_benchmark_summary(model, metrics)
    print("\n[OK]  RT-DETR pipeline complete. Export with:  python -m src.rtdetr.export_rtdetr")


if __name__ == "__main__":
    main()
