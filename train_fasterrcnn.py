#!/usr/bin/env python3
"""
train_fasterrcnn.py — Faster RCNN v2 crop-disease training pipeline.

Steps performed in order:
  1.  Stage hard negatives      — downloads 200 diverse non-crop images,
                                  added as empty-annotation training images
                                  so the model learns zero-detection on OOD inputs
  2.  Train FasterRCNN v2       — resume-aware, MPS/CUDA/CPU, dry-run capable
  3.  Generate figures          — 11 publication-quality PNGs → outputs/fasterrcnn_output/
  4.  Export                    — ExecuTorch (.pte) + ONNX + TorchScript mobile
                                  → outputs/fasterrcnn_output/models/

Architecture upgrade from detector-torch.ipynb:
  • fasterrcnn_resnet50_fpn  (v1) → fasterrcnn_resnet50_fpn_v2  (v2)
  • FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT  (improved COCO pretraining)
  • num_classes 23 (wrong, 0-indexed) → 24 (correct: 23 diseases + background)
  • integer_label subtracted by 1 (bug) → kept 1-23 as-is (FasterRCNN needs ≥1)
  • Fixed epochs 4 → 30 with early stopping + resume support
  • Manual SGD → SGD + cosine-warmup LR schedule
  • No augmentation → torchvision.transforms.v2 augmentation pipeline
  • No OOD guard → 200 hard-negative images (empty annotation)
  • No export → ExecuTorch + ONNX + TorchScript mobile export

Usage
-----
  python train_fasterrcnn.py                     # full pipeline (steps 1–4)
  python train_fasterrcnn.py --dry-run           # 2-epoch timing + estimate
  python train_fasterrcnn.py --skip-negatives    # skip hard-negative download
  python train_fasterrcnn.py --figures-only      # regenerate figures only
  python train_fasterrcnn.py --export-only       # re-export best checkpoint
  python train_fasterrcnn.py --no-figures        # train without figures
  DRY_RUN=1 python train_fasterrcnn.py           # dry-run via env var
"""

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "fasterrcnn_output"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
MODELS_DIR   = OUTPUT_DIR / "models"
METRICS_FILE = OUTPUT_DIR / "metrics_history.json"

TRAIN_CSV = DATASET_DIR / "final_train_labels.csv"
VAL_CSV   = DATASET_DIR / "final_validate_labels.csv"
TEST_CSV  = DATASET_DIR / "final_test_labels.csv"

TRAIN_IMG_DIR = DATASET_DIR / "train"
VAL_IMG_DIR   = DATASET_DIR / "validate"
TEST_IMG_DIR  = DATASET_DIR / "test"

# ── Model ─────────────────────────────────────────────────────────────────────
# FasterRCNN labels: 0 = background (reserved), 1-23 = disease classes.
# num_classes must include the background class → 23 + 1 = 24.
NUM_CLASSES = 24
IMG_SIZE    = 640

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS_DEFAULT        = 30
PATIENCE              = 8
BATCH_SIZE            = 4    # safe on 24 GB unified memory; Faster RCNN is heavy
LR0                   = 5e-3
WEIGHT_DECAY          = 5e-4
MOMENTUM              = 0.9
WARMUP_EPOCHS         = 3
FREEZE_BACKBONE_EPOCHS= 5    # lock backbone for first N epochs (same spirit as freeze=10 in YOLO)
GRAD_CLIP             = 10.0
EVAL_EVERY            = 5    # run val mAP every N epochs (expensive on 40k images)

# ── Hard negatives ────────────────────────────────────────────────────────────
NUM_NEGATIVES = 200   # ~0.5 % of 40 k train images — proportional to YOLO approach

# ── Class map  (1-indexed, matching dataset/label_map.json exactly) ───────────
# 0 = background (Faster RCNN convention, not a disease)
CLASS_NAMES = [
    "",                              # 0  background
    "Corn Cercospora Leaf Spot",     # 1
    "Corn Common Rust",              # 2
    "Corn Healthy",                  # 3
    "Corn Streak",                   # 4
    "Corn Northern Leaf Blight",     # 5
    "Pepper Leaf Curl",              # 6
    "Pepper Cercospora",             # 7
    "Pepper Leaf Blight",            # 8
    "Pepper Bacterial Spot",         # 9
    "Pepper Leaf Mosaic",            # 10
    "Pepper Healthy",                # 11
    "Pepper Fusarium",               # 12
    "Pepper Septoria",               # 13
    "Pepper Late Blight",            # 14
    "Pepper Early Blight",           # 15
    "Tomato Late Blight",            # 16
    "Tomato Early Blight",           # 17
    "Tomato Bacterial Spot",         # 18
    "Tomato Septoria",               # 19
    "Tomato Fusarium",               # 20
    "Tomato Leaf Curl",              # 21
    "Tomato Healthy",                # 22
    "Tomato Mosaic",                 # 23
]
# Convenience list for plot labels (0-indexed display, skips background)
CLASS_NAMES_DISPLAY = CLASS_NAMES[1:]   # len == 23


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Hard negatives (OOD guard)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_hard_negatives(num: int = NUM_NEGATIVES, skip: bool = False) -> list:
    """
    Download diverse non-crop images and return their paths.

    Images are saved to data/negatives/images/.  Already-downloaded files are
    skipped, making this fully resumable.  A fixed seed ensures the same image
    IDs are chosen every run.

    Returns a list of Path objects for the staged negative images.
    """
    if skip:
        print("  Hard-negative preparation skipped (--skip-negatives).")
        neg_img_dir = NEG_DIR / "images"
        return sorted(neg_img_dir.glob("*.jpg"))[:num] if neg_img_dir.exists() else []

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
                seed, exc = future.result()
                if exc:
                    print(f"    ⚠  seed {seed}: {exc}")
                    err += 1
                else:
                    ok += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} fetched  (ok={ok}, errors={err})")
        print(f"  Download complete: {ok} new, {err} failed")

    all_negs = sorted(neg_img_dir.glob("*.jpg"))[:num]
    print(f"  Hard negatives ready: {len(all_negs)} images  ({neg_img_dir})")
    print("  ✅  Hard-negative setup complete")
    return all_negs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load an annotation CSV and compute the img_id column."""
    df = pd.read_csv(csv_path)
    # Validate and filter degenerate boxes
    df = df[(df["x1"] < df["x2"]) & (df["y1"] < df["y2"])].copy()
    df["img_id"] = df["fname"].apply(lambda x: x.rsplit(".", 1)[0])
    return df


class CropDiseaseDataset(Dataset):
    """
    CSV-based detection dataset for the FasterRCNN pipeline.

    Positive samples: grouped by img_id from the annotation CSV.
    Negative samples: hard-negative image paths with no annotations.

    integer_label values are kept 1-indexed (matching label_map.json).
    Faster RCNN treats label 0 as background; disease classes start at 1.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform=None,
        neg_paths: Optional[list] = None,
    ):
        self.image_ids = df["img_id"].unique()
        self.df        = df
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.neg_paths = neg_paths or []
        self._n_pos    = len(self.image_ids)

    def __len__(self) -> int:
        return self._n_pos + len(self.neg_paths)

    def __getitem__(self, idx: int):
        if idx < self._n_pos:
            return self._get_positive(idx)
        return self._get_negative(idx - self._n_pos)

    # ── Positive sample ───────────────────────────────────────────────────────
    def _get_positive(self, idx: int):
        img_id  = self.image_ids[idx]
        records = self.df[self.df["img_id"] == img_id]

        img_path = self.image_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_t = v2.functional.to_image(img)           # uint8 [C,H,W]
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes  = records[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        labels = records["integer_label"].values.astype(np.int64)  # 1-indexed

        boxes_tv = tv_tensors.BoundingBoxes(
            torch.as_tensor(boxes),
            format="XYXY",
            canvas_size=(h, w),
        )
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            img_t, boxes_tv = self.transform(img_t, boxes_tv)
        else:
            img_t = v2.functional.to_dtype(img_t, torch.float32, scale=True)

        boxes_out = torch.as_tensor(boxes_tv, dtype=torch.float32)
        boxes_out, labels_t = _sanitise_boxes(boxes_out, labels_t, h, w)

        return img_t, _make_target(boxes_out, labels_t, idx)

    # ── Negative (hard negative) sample ───────────────────────────────────────
    def _get_negative(self, neg_idx: int):
        img_path = self.neg_paths[neg_idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        img_t = v2.functional.to_image(img)
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes_tv = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            format="XYXY",
            canvas_size=(h, w),
        )

        if self.transform:
            img_t, boxes_tv = self.transform(img_t, boxes_tv)
        else:
            img_t = v2.functional.to_dtype(img_t, torch.float32, scale=True)

        empty_boxes  = torch.zeros((0, 4), dtype=torch.float32)
        empty_labels = torch.zeros((0,),   dtype=torch.int64)
        return img_t, _make_target(empty_boxes, empty_labels, self._n_pos + neg_idx)


def _sanitise_boxes(
    boxes: torch.Tensor, labels: torch.Tensor, h: int, w: int
) -> tuple:
    """Clamp to image boundary and remove degenerate boxes."""
    if boxes.numel() == 0:
        return boxes, labels
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], labels[keep]


def _make_target(
    boxes: torch.Tensor, labels: torch.Tensor, idx: int
) -> dict:
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return {
        "boxes":    boxes,
        "labels":   labels,
        "image_id": torch.tensor([idx]),
        "area":     area,
        "iscrowd":  torch.zeros(labels.shape[0], dtype=torch.int64),
    }


def collate_fn(batch):
    return tuple(zip(*batch))


# ══════════════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transform():
    """
    Joint image + bounding-box augmentation via torchvision.transforms.v2.
    No external dependencies required.
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        v2.ToDtype(torch.float32, scale=True),   # uint8 → float32 [0, 1]
    ])


def get_val_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    FasterRCNN ResNet-50 FPN v2.

    v2 uses the same ResNet-50 + FPN architecture but was re-trained with
    multi-scale training, improved data augmentation, and longer schedules.
    COCO box AP improves from 37.0 (v1) to 46.7 (v2).
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


# ══════════════════════════════════════════════════════════════════════════════
# Device resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_startup(device, n_train, n_val, epochs, dry_run):
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  Model        : FasterRCNN-ResNet50-FPN-v2")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {BATCH_SIZE}  (workers={'0 (MPS)' if device.type == 'mps' else min(8, os.cpu_count() or 1)})")
    print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Train images : {n_train:,}  (incl. {NUM_NEGATIVES} hard-negatives)")
    print(f"  Val images   : {n_val:,}")
    print(f"  Epochs       : {epochs}" + ("  ← DRY RUN" if dry_run else ""))
    print(f"  LR           : {LR0}  (cosine, warmup {WARMUP_EPOCHS} ep)")
    print(f"  Backbone     : frozen for first {FREEZE_BACKBONE_EPOCHS} epochs")
    print(f"  num_classes  : {NUM_CLASSES}  (23 diseases + background)")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# LR schedule — linear warmup + cosine decay
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_map: float,
    metrics_history: dict,
    is_best: bool,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":             epoch,
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map":          best_map,
        "metrics_history":   metrics_history,
    }
    last_path = CKPT_DIR / "last.pth"
    torch.save(state, last_path)
    if is_best:
        shutil.copy2(last_path, CKPT_DIR / "best.pth")
    if epoch % 10 == 0:
        shutil.copy2(last_path, CKPT_DIR / f"epoch_{epoch:04d}.pth")


def _is_resumable(ckpt_path: Path) -> bool:
    """Return True only when checkpoint contains training state (not a weights-only export)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return isinstance(ckpt, dict) and all(
            k in ckpt for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict")
        )
    except Exception:
        return False


def load_checkpoint(ckpt_path: Path, model, optimizer, scheduler):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_map", 0.0), ckpt.get("metrics_history", {})


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler=None,        # torch.amp.GradScaler for CUDA; None on MPS / CPU
) -> dict:
    model.train()
    totals = {"total": 0.0, "classifier": 0.0, "box_reg": 0.0,
              "objectness": 0.0, "rpn_box_reg": 0.0}
    n = len(loader)

    # autocast is only safe + beneficial on CUDA; skip on MPS (fp16 index quirks)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if scaler is not None
        else contextlib.nullcontext()
    )

    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in targets]

        with autocast_ctx:
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        totals["total"] += losses.item()
        for key, val in loss_dict.items():
            short = key.replace("loss_", "")
            if short in totals:
                totals[short] += val.item()

        if (batch_idx + 1) % max(1, n // 5) == 0 or batch_idx == n - 1:
            lr  = optimizer.param_groups[0]["lr"]
            pct = (batch_idx + 1) / n * 100
            print(f"    ep {epoch:3d}  [{pct:5.1f}%]  "
                  f"loss={totals['total']/(batch_idx+1):.4f}  lr={lr:.2e}")

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 23,
) -> dict:
    """
    Compute VOC-style mAP@0.5 and per-class AP.

    Uses pure NumPy — no torchmetrics or pycocotools needed.
    """
    model.eval()

    class_dets: dict = defaultdict(list)   # class_id → [(score, is_tp), ...]
    class_ngt:  dict = defaultdict(int)    # class_id → n ground-truth boxes

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, gt in zip(preds, targets):
            gt_boxes  = gt["boxes"].cpu().numpy()
            gt_labels = gt["labels"].cpu().numpy()

            p_boxes  = pred["boxes"].cpu().numpy()
            p_scores = pred["scores"].cpu().numpy()
            p_labels = pred["labels"].cpu().numpy()

            for c in range(1, num_classes + 1):
                gt_mask = gt_labels == c
                gt_c    = gt_boxes[gt_mask]
                class_ngt[c] += int(gt_mask.sum())

                det_mask = p_labels == c
                if not det_mask.any():
                    continue

                c_boxes  = p_boxes[det_mask]
                c_scores = p_scores[det_mask]
                order    = c_scores.argsort()[::-1]

                matched = set()
                for i in order:
                    best_iou, best_j = 0.0, -1
                    for j, gb in enumerate(gt_c):
                        if j in matched:
                            continue
                        iou = _box_iou(c_boxes[i], gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    is_tp = int(best_iou >= 0.5 and best_j >= 0)
                    if is_tp:
                        matched.add(best_j)
                    class_dets[c].append((c_scores[i], is_tp))

    # Per-class AP (11-point interpolation, VOC style)
    aps = {}
    for c in range(1, num_classes + 1):
        ngt = class_ngt[c]
        if ngt == 0:
            aps[c] = float("nan")
            continue
        dets = sorted(class_dets.get(c, []), key=lambda x: -x[0])
        if not dets:
            aps[c] = 0.0
            continue
        tp = np.array([d[1] for d in dets], dtype=np.float32)
        fp = 1 - tp
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec  = tp_c / ngt
        prec = tp_c / (tp_c + fp_c)
        ap   = 0.0
        for thresh in np.linspace(0, 1, 11):
            p = prec[rec >= thresh]
            ap += float(np.max(p)) if len(p) > 0 else 0.0
        aps[c] = ap / 11.0

    valid_aps = [v for v in aps.values() if not math.isnan(v)]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return {"map50": map50, "per_class_ap": aps}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / (union + 1e-6)


def _load_metrics() -> dict:
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {"epoch": [], "train_total": [], "train_cls": [], "train_box_reg": [],
            "train_obj": [], "train_rpn": [], "val_map50": [], "lr": []}


def _save_metrics(history: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Publication figures
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_df(split: str) -> pd.DataFrame:
    """Load annotation CSV for one split and add derived columns."""
    path_map = {"train": TRAIN_CSV, "valid": VAL_CSV, "test": TEST_CSV}
    df = pd.read_csv(path_map[split])
    df = df[(df["x1"] < df["x2"]) & (df["y1"] < df["y2"])].copy()
    df["split"]       = split
    df["img_id"]      = df["fname"].apply(lambda x: x.rsplit(".", 1)[0])
    df["class_name"]  = df["class"]
    df["crop"]        = df["class"].apply(lambda x: x.split()[0])
    df["cx"]          = ((df["x1"] + df["x2"]) / 2) / df["width"]
    df["cy"]          = ((df["y1"] + df["y2"]) / 2) / df["height"]
    df["bw"]          = (df["x2"] - df["x1"]) / df["width"]
    df["bh"]          = (df["y2"] - df["y1"]) / df["height"]
    df["area"]        = df["bw"] * df["bh"]
    return df


def generate_figures(per_class_ap: Optional[dict] = None) -> None:
    """
    Generate publication figures and save to outputs/fasterrcnn_output/.

    per_class_ap : dict {class_id (1-23) → float} or None.
        When provided, Fig 11 (per-class AP) is generated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        print(f"  ⚠  Figure generation skipped (missing: {exc})")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Publication rcParams ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.framealpha": 0.9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches":0.15,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.axisbelow":    True,
    })

    CROP_PAL  = {"Corn": "#E8973A", "Pepper": "#27AE60", "Tomato": "#C0392B"}
    SPLIT_PAL = {"train": "#2980B9", "valid": "#27AE60", "test": "#E74C3C"}
    HEALTHY   = "#3498DB"

    def cls_color(name):
        if "Healthy" in name:
            return HEALTHY
        for crop, col in CROP_PAL.items():
            if name.startswith(crop):
                return col
        return "#95A5A6"

    CLS_COLORS = [cls_color(c) for c in CLASS_NAMES_DISPLAY]

    # ── Load data once ────────────────────────────────────────────────────────
    print("  Loading annotation data …")
    dfs    = {s: _load_split_df(s) for s in ["train", "valid", "test"]}
    df_all = pd.concat(dfs.values(), ignore_index=True)
    df_box = df_all.copy()
    n_imgs = {s: len(list((DATASET_DIR / (s if s != "valid" else "validate")).glob("*.jpg")))
              for s in ["train", "valid", "test"]}
    for s, df in dfs.items():
        print(f"    {s:6s}: {n_imgs[s]:5d} imgs | {len(df):6d} boxes")

    saved = []

    # ── Fig 01: Dataset Split Overview ───────────────────────────────────────
    splits  = ["train", "valid", "test"]
    n_img_v = [n_imgs[s]  for s in splits]
    n_box_v = [len(dfs[s]) for s in splits]
    clrs    = [SPLIT_PAL[s] for s in splits]
    total_i = sum(n_img_v)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset Split Overview", fontsize=15, fontweight="bold", y=1.02)

    ax = axes[0]
    bars = ax.bar(splits, n_img_v, color=clrs, width=0.5, zorder=3)
    for bar, n in zip(bars, n_img_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{n:,}\n({n/total_i*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Images per Split"); ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(n_img_v) * 1.3)

    n_ann = n_img_v[0]
    ax = axes[1]
    _, _, ats = ax.pie(
        [n_ann, NUM_NEGATIVES],
        labels=["Annotated", "Hard Negatives"],
        colors=["#2980B9", "#BDC3C7"], autopct="%1.1f%%", startangle=90,
        explode=(0.04, 0.04), textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in ats:
        at.set_fontweight("bold")
    ax.set_title("Train: Annotated vs Hard Negatives")

    ax = axes[2]
    bars = ax.bar(splits, n_box_v, color=clrs, width=0.5, zorder=3)
    for bar, n in zip(bars, n_box_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                f"{n:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Annotation Boxes per Split"); ax.set_ylabel("Box Count")
    ax.set_ylim(0, max(n_box_v) * 1.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_01_dataset_overview.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_01  dataset overview")

    # ── Fig 02: Per-Class Annotation Count (Training) ────────────────────────
    tc  = dfs["train"]
    cc  = tc.groupby("class_name").size().reindex(CLASS_NAMES_DISPLAY, fill_value=0)
    ypos = np.arange(len(CLASS_NAMES_DISPLAY))

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(ypos, cc.values, color=CLS_COLORS, height=0.68, zorder=3)
    for bar, val in zip(bars, cc.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Bounding Boxes")
    ax.set_title("Training Set — Per-Class Annotation Count", pad=12)
    ax.set_xlim(0, cc.max() * 1.18)
    for b in [4.5, 14.5]:
        ax.axhline(y=b, color="#7F8C8D", lw=0.9, linestyle="--", alpha=0.7)
    for i, (nm, cnt) in enumerate(zip(CLASS_NAMES_DISPLAY, cc.values)):
        if cnt < 30:
            ax.text(cnt + 50, i, "  ⚠ < 30", va="center",
                    color="#E74C3C", fontsize=8.5, fontstyle="italic")
    patches = [
        mpatches.Patch(color=CROP_PAL["Corn"],   label="Corn (classes 1–5)"),
        mpatches.Patch(color=CROP_PAL["Pepper"], label="Pepper (classes 6–15)"),
        mpatches.Patch(color=CROP_PAL["Tomato"], label="Tomato (classes 16–23)"),
        mpatches.Patch(color=HEALTHY,            label="Healthy variants"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_02_class_distribution_train.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_02  class distribution (train)")

    # ── Fig 03: Cross-Split Distribution ─────────────────────────────────────
    cnt_abs = {
        s: dfs[s].groupby("class_name").size()
                  .reindex(CLASS_NAMES_DISPLAY, fill_value=0)
        for s in ["train", "valid", "test"]
    }
    cdf   = pd.DataFrame(cnt_abs)
    cnorm = cdf.div(cdf.sum(axis=0), axis=1) * 100
    yp    = np.arange(len(CLASS_NAMES_DISPLAY))
    w     = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle("Class Distribution Across Dataset Splits",
                 fontsize=15, fontweight="bold", y=1.01)
    for ax, data, xlabel, title in [
        (axes[0], cdf,   "Annotation Box Count",            "Absolute Box Counts"),
        (axes[1], cnorm, "Relative Frequency (% of split)", "Normalised Distribution"),
    ]:
        for i, (sp, col) in enumerate(SPLIT_PAL.items()):
            ax.barh(yp + (i - 1) * w, data[sp], height=w,
                    color=col, alpha=0.85, label=sp.capitalize(), zorder=3)
        ax.set_yticks(yp)
        ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_title(title)
        ax.legend(loc="lower right")
        for b in [4.5, 14.5]:
            ax.axhline(y=b, color="#7F8C8D", lw=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_03_cross_split_distribution.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_03  cross-split distribution")

    # ── Fig 04: Annotation Density (boxes per image) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Annotation Density — Boxes per Image", fontsize=15, fontweight="bold")
    for ax, split in zip(axes, ["train", "valid"]):
        per_img = dfs[split].groupby("img_id").size()
        ax.hist(per_img.values, bins=range(0, int(per_img.max()) + 2),
                color=SPLIT_PAL[split], alpha=0.75,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(per_img.mean(),   color="navy",    linestyle="--", lw=2,
                   label=f"Mean  = {per_img.mean():.2f}")
        ax.axvline(per_img.median(), color="darkred", linestyle=":",  lw=2,
                   label=f"Median = {per_img.median():.0f}")
        ax.set_xlabel("Boxes per Image"); ax.set_ylabel("Number of Images")
        ax.set_title(f"{split.capitalize()} Split  (σ = {per_img.std():.2f})")
        ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_04_annotation_density.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_04  annotation density")

    # ── Fig 05: Bounding Box Spatial Heatmap ──────────────────────────────────
    tb = dfs["train"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        "Bounding Box Centre Spatial Distribution (Training Set)\n"
        "(0,0) = top-left  ·  (1,1) = bottom-right  ·  white + = image centre",
        fontsize=12, fontweight="bold")
    panels = [("All Classes", tb)] + [
        (c, tb[tb.crop == c]) for c in ["Corn", "Pepper", "Tomato"]
    ]
    for ax, (label, subset) in zip(axes, panels):
        if len(subset) == 0:
            ax.set_visible(False); continue
        h2d = ax.hist2d(subset.cx.values, subset.cy.values,
                        bins=40, cmap="YlOrRd", density=True, cmin=1e-6)
        plt.colorbar(h2d[3], ax=ax, shrink=0.8, label="Density")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()
        ax.set_xlabel("cx"); ax.set_ylabel("cy"); ax.set_title(label)
        ax.set_aspect("equal"); ax.grid(False)
        ax.plot(0.5, 0.5, "w+", markersize=10, markeredgewidth=2)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_05_bbox_spatial_heatmap.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_05  bounding-box spatial heatmap")

    # ── Fig 06: Bounding Box Geometry ────────────────────────────────────────
    tb2 = dfs["train"].copy()
    tb2["aspect"]   = tb2.bw / tb2.bh.clip(1e-6)
    tb2["area_pct"] = tb2.area * 100

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
    ax_sc = fig.add_subplot(gs[0, :2])
    ax_bp = fig.add_subplot(gs[0, 2])
    ax_wh = fig.add_subplot(gs[1, 0])
    ax_hh = fig.add_subplot(gs[1, 1])
    ax_ar = fig.add_subplot(gs[1, 2])
    fig.suptitle("Bounding Box Geometry Analysis — Training Set",
                 fontsize=14, fontweight="bold")

    samp = tb2.sample(min(6000, len(tb2)), random_state=42)
    for crop, col in CROP_PAL.items():
        sub = samp[samp.crop == crop]
        ax_sc.scatter(sub.bw, sub.bh, c=col, alpha=0.20, s=7, label=crop)
    ax_sc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Square")
    ax_sc.set_xlabel("Normalised Width"); ax_sc.set_ylabel("Normalised Height")
    ax_sc.set_title("Box Width vs Height (6k sample)")
    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1)
    ax_sc.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in CROP_PAL.items()])

    crop_order = ["Corn", "Pepper", "Tomato"]
    bp = ax_bp.boxplot(
        [tb2[tb2.crop == c].area_pct.values for c in crop_order],
        labels=crop_order, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, crop in zip(bp["boxes"], crop_order):
        patch.set_facecolor(CROP_PAL[crop]); patch.set_alpha(0.75)
    ax_bp.set_ylabel("Box Area (% of image)"); ax_bp.set_title("Box Area by Crop")

    for ax, col, lbl, title in [
        (ax_wh, "bw",     "Normalised Width",  "Width Distribution"),
        (ax_hh, "bh",     "Normalised Height", "Height Distribution"),
        (ax_ar, "aspect", "Width / Height",    "Aspect Ratio Distribution"),
    ]:
        for crop, color in CROP_PAL.items():
            vals = tb2[tb2.crop == crop][col].dropna()
            vals = vals[vals < vals.quantile(0.99)]
            ax.hist(vals, bins=40, color=color, alpha=0.50,
                    density=True, label=crop, edgecolor="none")
        ax.set_xlabel(lbl); ax.set_ylabel("Density")
        ax.set_title(title); ax.legend(fontsize=9)

    out = OUTPUT_DIR / "fig_06_bbox_geometry.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_06  bounding-box geometry")

    # ── Fig 07: Class Imbalance ───────────────────────────────────────────────
    cnt_r = {}
    for s in ["train", "valid", "test"]:
        n = (dfs[s].groupby("class_name").size()
                   .reindex(CLASS_NAMES_DISPLAY, fill_value=0))
        cnt_r[s] = n / n.sum()
    cdf2 = pd.DataFrame(cnt_r)
    rv = (cdf2["valid"] / cdf2["train"].clip(1e-9)).clip(0, 5)
    rt = (cdf2["test"]  / cdf2["train"].clip(1e-9)).clip(0, 5)

    fig, ax = plt.subplots(figsize=(11, 8))
    yp2 = np.arange(len(CLASS_NAMES_DISPLAY)); w2 = 0.36
    ax.barh(yp2 - w2 / 2, rv.values, height=w2, color=SPLIT_PAL["valid"],
            alpha=0.82, label="Val / Train ratio", zorder=3)
    ax.barh(yp2 + w2 / 2, rt.values, height=w2, color=SPLIT_PAL["test"],
            alpha=0.82, label="Test / Train ratio", zorder=3)
    ax.axvline(1.0, color="black", lw=1.2, linestyle="--", alpha=0.7,
               label="Balanced (ratio = 1.0)")
    ax.axvspan(0.7, 1.3, alpha=0.06, color="green", label="±30% balance zone")
    ax.set_yticks(yp2)
    ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Ratio of relative frequencies")
    ax.set_title("Class Imbalance — Split Frequency Ratios", pad=10)
    ax.legend(loc="lower right"); ax.set_xlim(0, 5.2)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_07_class_imbalance.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_07  class imbalance")

    # ── Fig 08: Training Configuration Table ─────────────────────────────────
    ni_tr = n_imgs["train"] + NUM_NEGATIVES
    ni_va = n_imgs["valid"]
    cfg_rows = [
        ("Architecture",        "FasterRCNN-ResNet50-FPN-v2",  "V2: improved COCO pretraining"),
        ("Pretrained weights",  "FasterRCNN_ResNet50_FPN_V2",  "COCO box AP 46.7 (vs 37.0 v1)"),
        ("num_classes",         "24 (23 disease + background)", "0 = bg; 1-23 = disease labels"),
        ("Image size",          f"{IMG_SIZE}×{IMG_SIZE}",      "Matches original dataset images"),
        ("Batch size",          str(BATCH_SIZE),               "Safe on 24 GB unified memory"),
        ("Epochs",              str(EPOCHS_DEFAULT),           "Pretrained backbone → fast convergence"),
        ("Early-stop patience", str(PATIENCE),                 "Stop after N non-improving epochs"),
        ("Optimizer",           "SGD (momentum=0.9)",          "Standard for Faster RCNN fine-tuning"),
        ("LR (lr0)",            f"{LR0}",                      "Cosine decay from warmup"),
        ("Warmup epochs",       str(WARMUP_EPOCHS),            "Linear ramp for stability"),
        ("Freeze backbone",     f"{FREEZE_BACKBONE_EPOCHS} ep","Prevents early overwriting of pretrained features"),
        ("Grad clip",           str(GRAD_CLIP),                "Prevents exploding gradients"),
        ("Augmentation",        "HFlip + ColorJitter + Blur",  "torchvision.transforms.v2"),
        ("Hard negatives",      str(NUM_NEGATIVES),            "Diverse non-crop images; OOD guard"),
        ("Device",              "MPS (M4 Pro)",                "Apple Silicon Metal Performance Shaders"),
        ("workers",             "0",                           "MPS requires 0; macOS fork conflicts"),
        ("Train images",        f"{ni_tr:,}",                  f"Incl. {NUM_NEGATIVES} hard-negatives"),
        ("Val images",          f"{ni_va:,}",                  "Original validate split"),
        ("Classes",             "23",                          "Corn ×5, Pepper ×10, Tomato ×8"),
        ("Dataset source",      "Ghana Crop Disease",          "CSV annotations; XYXY absolute px"),
    ]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    tbl = ax.table(cellText=cfg_rows,
                   colLabels=["Parameter", "Value", "Rationale / Notes"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    tbl.auto_set_column_width([0, 1, 2])
    for col in range(3):
        cell = tbl[0, col]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.065)
    for row in range(1, len(cfg_rows) + 1):
        bg = "#F4F6F7" if row % 2 == 0 else "white"
        for col in range(3):
            tbl[row, col].set_facecolor(bg)
            tbl[row, col].set_height(0.046)
    ax.set_title("Training Configuration Summary — FasterRCNN v2",
                 fontsize=14, fontweight="bold", pad=20, y=0.98)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_08_training_config.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_08  training config table")

    # ── Fig 09: LR Schedule + Augmentation Profile ───────────────────────────
    ep_arr = np.arange(1, EPOCHS_DEFAULT + 1)

    def _cosine_lr(ep):
        if ep <= WARMUP_EPOCHS:
            return LR0 * ep / WARMUP_EPOCHS
        prog = (ep - WARMUP_EPOCHS) / (EPOCHS_DEFAULT - WARMUP_EPOCHS)
        return LR0 * 0.5 * (1 + np.cos(np.pi * prog))

    lr_vals = np.array([_cosine_lr(e) for e in ep_arr])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Training Schedule & Augmentation Profile — FasterRCNN v2",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(ep_arr, lr_vals, color="#2980B9", lw=2.2, zorder=4, label="Learning Rate")
    ax.fill_between(ep_arr, 0, lr_vals, alpha=0.12, color="#2980B9")
    ax.axvspan(1, WARMUP_EPOCHS,           alpha=0.12, color="#F39C12",
               label=f"Warmup ({WARMUP_EPOCHS} ep)")
    ax.axvspan(WARMUP_EPOCHS + 1, EPOCHS_DEFAULT, alpha=0.06, color="#27AE60",
               label="Cosine Decay")
    ax.axvline(FREEZE_BACKBONE_EPOCHS + 1, color="#8E44AD", linestyle=":", lw=1.8,
               label=f"Unfreeze backbone @ ep {FREEZE_BACKBONE_EPOCHS + 1}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title(f"Cosine LR Schedule  (lr0={LR0}, {EPOCHS_DEFAULT} epochs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(1, EPOCHS_DEFAULT); ax.set_ylim(bottom=0)

    aug_params = {
        "HorizontalFlip":    0.500,
        "ColorJitter (bri)": 0.300,
        "ColorJitter (con)": 0.300,
        "ColorJitter (sat)": 0.300,
        "ColorJitter (hue)": 0.050,
        "RandomGrayscale":   0.050,
        "GaussianBlur":      1.000,   # always applied with sigma range
    }
    ay  = np.arange(len(aug_params))
    av  = list(aug_params.values())
    al  = list(aug_params.keys())
    ac  = ["#C0392B" if v >= 0.5 else "#2980B9" if v >= 0.1 else "#95A5A6" for v in av]
    ax2 = axes[1]
    ax2.barh(ay, av, color=ac, height=0.65, zorder=3)
    ax2.set_yticks(ay); ax2.set_yticklabels(al, fontsize=9.5); ax2.invert_yaxis()
    ax2.set_xlabel("Probability / Strength"); ax2.set_title("Augmentation Parameters")
    ax2.set_xlim(0, 1.18)
    for i, v in enumerate(av):
        ax2.text(v + 0.015, i,
                 f"{v:.3f}".rstrip("0").rstrip("."), va="center", fontsize=9)
    leg2 = [
        mpatches.Patch(color="#C0392B", label="Strong (≥0.5)"),
        mpatches.Patch(color="#2980B9", label="Moderate (0.1–<0.5)"),
        mpatches.Patch(color="#95A5A6", label="Mild (<0.1)"),
    ]
    ax2.legend(handles=leg2, loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_09_lr_schedule_augmentation.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_09  LR schedule + augmentation")

    # ── Fig 10: Training Metrics (post-training only) ─────────────────────────
    if METRICS_FILE.exists():
        history = _load_metrics()
        if history.get("epoch"):
            ep = history["epoch"]
            metric_series = [
                ("train_total",   "Total Train Loss",    "#E74C3C"),
                ("train_cls",     "Classifier Loss",     "#3498DB"),
                ("train_box_reg", "Box Reg Loss",        "#E67E22"),
                ("train_obj",     "Objectness Loss",     "#9B59B6"),
                ("train_rpn",     "RPN Box Reg Loss",    "#1ABC9C"),
                ("val_map50",     "Val mAP@0.50",        "#27AE60"),
            ]
            available = [(k, l, c) for k, l, c in metric_series if history.get(k)]
            ncols = 3
            nrows = (len(available) + ncols - 1) // ncols
            fig, axes_g = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
            flat_axes   = np.array(axes_g).flatten()
            fig.suptitle("Training Metrics — FasterRCNN v2", fontsize=15, fontweight="bold")
            for ax, (key, lbl, color) in zip(flat_axes, available):
                ax.plot(ep, history[key], color=color, lw=2)
                ax.set_title(lbl); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
            for ax in flat_axes[len(available):]:
                ax.set_visible(False)
            plt.tight_layout()
            out = OUTPUT_DIR / "fig_10_training_metrics.png"
            plt.savefig(out); plt.close(); saved.append(out)
            print("  ✓  fig_10  training metrics")
        else:
            print("  –  fig_10 skipped (no training history found)")
    else:
        print("  –  fig_10 skipped (run after training)")

    # ── Fig 11: Per-Class AP (post-training evaluation) ───────────────────────
    if per_class_ap is not None:
        aps = [per_class_ap.get(c, float("nan")) for c in range(1, 24)]
        colors_ap = [
            "#27AE60" if v >= 0.7 else "#E67E22" if v >= 0.4 else "#E74C3C"
            for v in aps
        ]
        ypos_ap = np.arange(23)

        fig, ax = plt.subplots(figsize=(11, 9))
        bars = ax.barh(ypos_ap, aps, color=colors_ap, height=0.68, zorder=3)
        for bar, val in zip(bars, aps):
            if not math.isnan(val):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=9)
        ax.set_yticks(ypos_ap)
        ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("Average Precision @ IoU=0.50")
        ax.set_title("Per-Class AP@0.5 — FasterRCNN v2", pad=12)
        ax.set_xlim(0, 1.15)
        ax.axvline(np.nanmean(aps), color="navy", linestyle="--", lw=1.5,
                   label=f"mAP@0.5 = {np.nanmean(aps):.3f}")
        for b in [4.5, 14.5]:
            ax.axhline(y=b, color="#7F8C8D", lw=0.9, linestyle="--", alpha=0.7)
        legend_handles = [
            mpatches.Patch(color="#27AE60", label="AP ≥ 0.70  (strong)"),
            mpatches.Patch(color="#E67E22", label="AP 0.40–0.70  (moderate)"),
            mpatches.Patch(color="#E74C3C", label="AP < 0.40  (weak)"),
        ]
        ax.legend(handles=legend_handles, loc="lower right")
        plt.tight_layout()
        out = OUTPUT_DIR / "fig_11_per_class_ap.png"
        plt.savefig(out); plt.close(); saved.append(out)
        print("  ✓  fig_11  per-class AP")
    else:
        print("  –  fig_11 skipped (run after training)")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  Publication figures  →  {OUTPUT_DIR}")
    print(sep)
    for f in saved:
        print(f"  {f.name:<54}  {f.stat().st_size / 1024:>7.1f} KB")
    print(sep)
    print(f"  Total: {len(saved)} figures  |  300 DPI  |  PNG")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Export
# ══════════════════════════════════════════════════════════════════════════════

def export_model(model: nn.Module) -> None:
    """
    Export the trained model in three formats.

    Priority:
      1. TorchScript mobile  (.ptl)  — primary mobile target (Android / iOS LibTorch)
      2. ONNX                (.onnx) — universal fallback
      3. ExecuTorch          (.pte)  — attempted; Faster RCNN's dynamic NMS output
                                       means this may partially succeed (backbone only)
      4. Metadata YAML               — class names, thresholds, input spec
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    print(f"\n  Export directory: {MODELS_DIR}")

    # ── 1. TorchScript mobile (.ptl) — most reliable ─────────────────────────
    print("  [1/4]  TorchScript mobile …")
    try:
        scripted = torch.jit.script(model)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(scripted)
        ptl_path  = MODELS_DIR / "crop_disease_fasterrcnn.ptl"
        optimized._save_for_lite_interpreter(str(ptl_path))
        print(f"         ✅ {ptl_path.name}  ({ptl_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"         ⚠  TorchScript mobile failed: {e}")
        try:
            # Fall back to plain .pt if mobile_optimizer fails
            scripted = torch.jit.script(model)
            pt_path = MODELS_DIR / "crop_disease_fasterrcnn_jit.pt"
            scripted.save(str(pt_path))
            print(f"         ✅ Saved plain TorchScript: {pt_path.name}")
        except Exception as e2:
            print(f"         ✗  TorchScript also failed: {e2}")

    # ── 2. ONNX export ────────────────────────────────────────────────────────
    print("  [2/4]  ONNX …")
    try:
        class _ONNXWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x: torch.Tensor):
                out = self.m([x[0]])
                return out[0]["boxes"], out[0]["scores"], out[0]["labels"].float()

        wrapper  = _ONNXWrapper(model)
        example  = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        onnx_path = MODELS_DIR / "crop_disease_fasterrcnn.onnx"
        torch.onnx.export(
            wrapper, example, str(onnx_path),
            opset_version=17,
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes={
                "images": {0: "batch"},
                "boxes":  {0: "n_det"},
                "scores": {0: "n_det"},
                "labels": {0: "n_det"},
            },
        )
        print(f"         ✅ {onnx_path.name}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"         ⚠  ONNX export failed: {e}")

    # ── 3. ExecuTorch export ──────────────────────────────────────────────────
    # Faster RCNN's NMS produces variable-length output which cannot be statically
    # exported.  Strategy: export backbone + FPN only (always works) as a feature
    # extractor .pte, then try the full model with strict=False.
    print("  [3/4]  ExecuTorch (.pte) …")
    try:
        from executorch.exir import to_edge
        from torch.export import export as torch_export
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        class _BackboneExportWrapper(nn.Module):
            """
            Exports backbone + FPN only.
            Returns a dict of feature maps as a flat tuple.
            Mobile app uses these features for downstream tasks or runs
            the detection head locally via TorchScript (.ptl).
            """
            def __init__(self, m):
                super().__init__()
                self.backbone = m.backbone

            def forward(self, x: torch.Tensor) -> tuple:
                feats = self.backbone(x)
                # FPN outputs keys: "0","1","2","3","pool"
                return (
                    feats["0"], feats["1"], feats["2"],
                    feats["3"], feats["pool"],
                )

        backbone_wrapper = _BackboneExportWrapper(model)
        backbone_wrapper.eval()
        example_input = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)

        exported = torch_export(backbone_wrapper, example_input, strict=False)
        edge_prog = to_edge(exported)
        try:
            edge_prog = edge_prog.to_backend(XnnpackPartitioner())
            print("         XNNPACK backend applied to backbone")
        except Exception as xe:
            print(f"         XNNPACK skipped ({xe}); using default backend")
        et_prog  = edge_prog.to_executorch()
        pte_path = MODELS_DIR / "crop_disease_fasterrcnn_backbone.pte"
        with open(pte_path, "wb") as f:
            f.write(et_prog.buffer)
        print(f"         ✅ Backbone ExecuTorch: {pte_path.name}  "
              f"({pte_path.stat().st_size / 1e6:.1f} MB)")
        print("            Note: backbone-only export.  Pair with .ptl for full")
        print("            detection (RPN + ROI head) at runtime.")

        # Attempt full model export (may fail due to variable NMS output)
        class _FullModelWrapper(nn.Module):
            MAX_DETS = 100

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor) -> tuple:
                out    = self.m([x[0]])
                boxes  = out[0]["boxes"]
                scores = out[0]["scores"]
                labels = out[0]["labels"]
                # topk with fixed k avoids data-dependent Python branches
                k = torch.tensor(self.MAX_DETS).clamp(max=scores.shape[0])
                topk_s, idx = torch.topk(scores, k.item(), sorted=True)
                topk_b = boxes[idx]
                topk_l = labels[idx]
                # Pad to MAX_DETS so output shape is always [MAX_DETS, ...]
                pad = self.MAX_DETS - topk_s.shape[0]
                if pad > 0:
                    topk_b = torch.cat([topk_b, topk_b.new_zeros(pad, 4)])
                    topk_s = torch.cat([topk_s, topk_s.new_zeros(pad)])
                    topk_l = torch.cat([topk_l, topk_l.new_zeros(pad)])
                return topk_b, topk_s, topk_l

        full_wrapper = _FullModelWrapper(model)
        full_wrapper.eval()
        try:
            exp_full  = torch_export(full_wrapper, example_input, strict=False)
            edge_full = to_edge(exp_full)
            try:
                edge_full = edge_full.to_backend(XnnpackPartitioner())
            except Exception:
                pass
            et_full  = edge_full.to_executorch()
            pte_full = MODELS_DIR / "crop_disease_fasterrcnn.pte"
            with open(pte_full, "wb") as f:
                f.write(et_full.buffer)
            print(f"         ✅ Full model ExecuTorch: {pte_full.name}  "
                  f"({pte_full.stat().st_size / 1e6:.1f} MB)")
        except Exception as e_full:
            print(f"         ℹ  Full ExecuTorch export skipped ({e_full})")
            print("            Use .ptl (TorchScript mobile) for full detection on device.")

    except Exception as e:
        print(f"         ⚠  ExecuTorch failed: {e}")
        print("            Use crop_disease_fasterrcnn.ptl for mobile deployment.")

    # ── 4. Metadata YAML ──────────────────────────────────────────────────────
    print("  [4/4]  Metadata YAML …")
    from datetime import datetime, timezone
    metadata = {
        "model_name":       "crop_disease_fasterrcnn",
        "architecture":     "FasterRCNN-ResNet50-FPN-v2",
        "task":             "object_detection",
        "exported_at":      datetime.now(timezone.utc).isoformat(),
        "input_size":       IMG_SIZE,
        "input_channels":   3,
        "num_classes":      NUM_CLASSES - 1,   # 23 disease classes (excl. background)
        "class_names":      CLASS_NAMES_DISPLAY,
        "conf_threshold":   0.50,
        "iou_nms_threshold":0.45,
        "crops_covered":    ["Corn", "Pepper", "Tomato"],
        "label_offset":     1,   # model output labels are 1-indexed; subtract 1 to index class_names
        "notes": (
            "Trained on Ghana Crop Disease dataset. "
            "Hard-negative mining applied for OOD robustness. "
            "Labels 1-23 correspond to class_names[0]-class_names[22]."
        ),
        "mobile_integration": {
            "primary_format":   ".ptl  (TorchScript mobile / LibTorch)",
            "fallback_format":  ".onnx (ONNX Runtime)",
            "executorch_format":".pte  (ExecuTorch backbone or full model)",
            "input_format":     "NCHW_RGB_float32_0to1",
            "model_transform":  "GeneralizedRCNNTransform handles normalisation internally",
        },
    }
    meta_path = MODELS_DIR / "model_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"         ✅ {meta_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ── Exported artefacts in {MODELS_DIR} ───")
    for fp in sorted(MODELS_DIR.iterdir()):
        print(f"  {fp.name:<52}  {fp.stat().st_size / 1e6:>7.2f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faster RCNN v2 crop-disease training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_fasterrcnn.py                     # full pipeline
  python train_fasterrcnn.py --dry-run           # 2-epoch timing validation
  python train_fasterrcnn.py --skip-negatives    # negatives already staged
  python train_fasterrcnn.py --figures-only      # regenerate figures only
  python train_fasterrcnn.py --export-only       # re-export best checkpoint
  python train_fasterrcnn.py --no-figures        # train without figures
  DRY_RUN=1 python train_fasterrcnn.py           # dry-run via env var
        """.strip(),
    )
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run 2 epochs; print epoch-time estimate")
    parser.add_argument("--epochs",         type=int, default=None,
                        help=f"Override epoch count (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--skip-negatives", action="store_true",
                        help="Skip hard-negative download/staging")
    parser.add_argument("--figures-only",   action="store_true",
                        help="Regenerate figures only; skip training")
    parser.add_argument("--export-only",    action="store_true",
                        help="Export best checkpoint only; skip training")
    parser.add_argument("--no-figures",     action="store_true",
                        help="Train without generating figures afterwards")
    args = parser.parse_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN", "0") == "1"
    epochs  = 2 if dry_run else (args.epochs or EPOCHS_DEFAULT)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Figures-only shortcut ─────────────────────────────────────────────────
    if args.figures_only:
        print("─── Figures-only mode ───────────────────────────────────────────")
        generate_figures()
        return

    # ── Export-only shortcut ──────────────────────────────────────────────────
    if args.export_only:
        print("─── Export-only mode ────────────────────────────────────────────")
        best_pth = CKPT_DIR / "best.pth"
        if not best_pth.exists():
            print(f"  ✗  No best checkpoint found at {best_pth}")
            return
        model  = build_model()
        ckpt   = torch.load(best_pth, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        export_model(model)
        return

    # ── Step 1: Hard negatives ────────────────────────────────────────────────
    print("\n─── Step 1/3: Hard-negative images (OOD guard) ──────────────────")
    neg_paths = prepare_hard_negatives(skip=args.skip_negatives)

    # ── Step 2: Training ──────────────────────────────────────────────────────
    print("\n─── Step 2/3: Training ───────────────────────────────────────────")
    device  = resolve_device()
    is_mps  = device.type == "mps"
    workers = 0 if is_mps else min(8, os.cpu_count() or 1)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # AMP: CUDA only (fp16 autocast + loss scaling for 30-50% throughput gain)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Build datasets
    train_df = _load_csv(TRAIN_CSV)
    val_df   = _load_csv(VAL_CSV)

    train_ds = CropDiseaseDataset(
        train_df, TRAIN_IMG_DIR,
        transform=get_train_transform(),
        neg_paths=neg_paths,
    )
    val_ds = CropDiseaseDataset(
        val_df, VAL_IMG_DIR,
        transform=get_val_transform(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),   # CUDA only; MPS uses unified memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    n_train = len(train_ds)
    n_val   = len(val_ds)
    log_startup(device, n_train, n_val, epochs, dry_run)

    # Build model + optimizer
    model     = build_model()
    model.to(device)

    # Separate param groups: backbone (lower LR) + rest
    backbone_params  = [p for p in model.backbone.parameters() if p.requires_grad]
    rest_params      = [p for p in model.parameters()
                        if p.requires_grad and not any(
                            p is bp for bp in backbone_params)]
    optimizer = torch.optim.SGD(
        [{"params": rest_params,     "lr": LR0},
         {"params": backbone_params, "lr": LR0 * 0.1}],
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, epochs)

    # Resume from checkpoint if available
    start_epoch = 0
    best_map    = 0.0
    history     = _load_metrics()

    last_pth = CKPT_DIR / "last.pth"
    if last_pth.exists() and _is_resumable(last_pth):
        print(f"  Resuming from: {last_pth}")
        start_epoch, best_map, history = load_checkpoint(
            last_pth, model, optimizer, scheduler)
        model.to(device)
        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"  Resumed at epoch {start_epoch}, best mAP@0.5 = {best_map:.4f}")
    elif last_pth.exists():
        print(f"  ⚠  last.pth found but not resumable (weights-only?). Fresh run.")

    # Freeze backbone for the first FREEZE_BACKBONE_EPOCHS epochs
    initial_freeze = max(0, FREEZE_BACKBONE_EPOCHS - start_epoch)
    if initial_freeze > 0:
        set_backbone_grad(model, False)
        print(f"  Backbone frozen for first {FREEZE_BACKBONE_EPOCHS} epochs")

    no_improve = 0
    t0         = time.perf_counter()

    for epoch in range(start_epoch + 1, epochs + 1):
        # Unfreeze backbone after FREEZE_BACKBONE_EPOCHS
        # PyTorch optimizer naturally skips requires_grad=False params in step(),
        # so simply setting requires_grad=True is sufficient — no param-group surgery.
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            set_backbone_grad(model, True)
            print(f"\n  Backbone unfrozen at epoch {epoch}")

        t_ep = time.perf_counter()
        losses = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        ep_time = time.perf_counter() - t_ep
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        # Periodic validation (expensive: ~10min per run on CPU/MPS)
        val_map = float("nan")
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            print(f"  Evaluating val mAP@0.5 (epoch {epoch}) …")
            eval_result = evaluate(model, val_loader, device)
            val_map = eval_result["map50"]
            print(f"  val mAP@0.5 = {val_map:.4f}")

        # Log to history
        history["epoch"].append(epoch)
        history["train_total"].append(losses["total"])
        history["train_cls"].append(losses["classifier"])
        history["train_box_reg"].append(losses["box_reg"])
        history["train_obj"].append(losses["objectness"])
        history["train_rpn"].append(losses["rpn_box_reg"])
        history["val_map50"].append(val_map if not math.isnan(val_map) else None)
        history["lr"].append(lr_now)
        _save_metrics(history)

        is_best = (not math.isnan(val_map)) and (val_map > best_map)
        if is_best:
            best_map  = val_map
            no_improve = 0
            print(f"  ✨  New best mAP@0.5: {best_map:.4f}")
        elif not math.isnan(val_map):
            no_improve += 1

        save_checkpoint(epoch, model, optimizer, scheduler, best_map, history, is_best)

        sep_out = "─" * 66
        print(f"  {sep_out}")
        print(f"  Epoch {epoch:3d}/{epochs}  |  "
              f"loss={losses['total']:.4f}  |  "
              f"mAP@0.5={val_map:.4f}  |  "
              f"lr={lr_now:.2e}  |  "
              f"time={ep_time:.0f}s")
        print(f"  {sep_out}")

        if dry_run and epoch >= 2:
            elapsed = time.perf_counter() - t0
            est_total = elapsed / 2 * EPOCHS_DEFAULT
            print(f"\n  Dry-run epoch time  : {elapsed / 2:.1f}s")
            print(f"  Estimated full run  : ~{est_total/60:.0f} min  "
                  f"({est_total/3600:.1f} h)  @ {EPOCHS_DEFAULT} epochs")
            print(f"  Best checkpoint     : {CKPT_DIR}/best.pth")
            print("  (Figures & export skipped in dry-run mode)")
            return

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    total_time = time.perf_counter() - t0
    print(f"\n✅  Training complete!  ({total_time / 3600:.1f} h total)")
    print(f"   Best mAP@0.5 : {best_map:.4f}")
    print(f"   Checkpoint   : {CKPT_DIR}/best.pth")

    # Load best weights for final evaluation and export
    best_pth = CKPT_DIR / "best.pth"
    if best_pth.exists():
        ckpt = torch.load(best_pth, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

    # Final full evaluation for per-class AP (Fig 11)
    print("\n  Running final evaluation for per-class AP …")
    final_result = evaluate(model, val_loader, device)
    print(f"  Final val mAP@0.5 = {final_result['map50']:.4f}")

    # ── Step 3: Publication figures ───────────────────────────────────────────
    if not args.no_figures:
        print("\n─── Step 3/3: Generating publication figures ─────────────────")
        generate_figures(per_class_ap=final_result.get("per_class_ap"))

    # ── Step 4: Export ────────────────────────────────────────────────────────
    print("\n─── Step 4: Export (ExecuTorch / ONNX / TorchScript mobile) ─────")
    export_model(model)


if __name__ == "__main__":
    main()
