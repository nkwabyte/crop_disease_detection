#!/usr/bin/env python3
"""
train_alt_faster_rcnn.py — Faster R-CNN baseline + ablation study, one file.

Merged from train_fasterrcnn.py and train_alt_fasterrcnn.py. Two pipelines share
one dataset/eval/checkpoint stack but keep separate hyperparameters and output
directories, selected with --mode:

  --mode baseline   Train the single Faster R-CNN v2 production baseline
                    (ResNet50-FPN-v2), generate figures, export to
                    ExecuTorch/ONNX/TorchScript.  -> outputs/fasterrcnn_output/

  --mode ablation   Train and compare 7 configurations across backbone depth,
                    proposal count, NMS policy and anchor scale, mirroring
                    Ren et al. (2015).                -> outputs/alt_fasterrcnn_output/

    1. mobilenet_300           MobileNetV3-FPN,  300 proposals (lightweight)
    2. resnet50_100            ResNet50-FPN-v2,  100 proposals
    3. resnet50_300        (*) ResNet50-FPN-v2,  300 proposals (selected baseline)
    4. resnet50_1000           ResNet50-FPN-v2, 1000 proposals
    5. resnet50_no_nms         ResNet50-FPN-v2,  300 proposals, NMS disabled
    6. resnet50_small_anchors  ResNet50-FPN-v2,  300 proposals, anchors 16-256 px
    7. resnet101_300           ResNet101-FPN,    300 proposals (heavier backbone)

Why the ablation's constants carry an ABL_ prefix
-------------------------------------------------
The two pipelines genuinely disagree on hyperparameters — the ablation trains 15
epochs with patience 5 and 100 hard negatives, the baseline uses the config
defaults (30 epochs, config patience, config negatives). Sharing those names
would silently retrain one pipeline with the other's settings, so every ablation
value is ABL_-prefixed and its own code refers to the prefixed names.

Shared helpers use the baseline's implementations. They are a superset: the same
logic with docstrings and type hints, and train_one_epoch takes an optional
GradScaler that defaults to None — which is exactly the ablation's previous
non-AMP behaviour. save_checkpoint is the one exception and stays separate as
save_ablation_checkpoint, because the ablation's signature leads with config_id.

Usage
-----
  python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline
  python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --dry-run
  python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation
  python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs resnet50_300
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
from src.fasterrcnn.config import (
    PROJECT_ROOT,
    DATASET_DIR,
    NEG_DIR,
    OUTPUT_DIR_BASELINE as OUTPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    TEST_IMG_DIR,
    NUM_CLASSES,
    IMG_SIZE,
    EPOCHS_DEFAULT,
    PATIENCE_DEFAULT as PATIENCE,
    BATCH_SIZE,
    CUDA_BATCH_SIZE,
    LR0,
    WEIGHT_DECAY,
    MOMENTUM,
    WARMUP_EPOCHS,
    FREEZE_BACKBONE_EPOCHS,
    GRAD_CLIP,
    EVAL_EVERY,
    NUM_NEGATIVES,
    CLASS_NAMES,
)
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from src.fasterrcnn.config import (
    PROJECT_ROOT,
    DATASET_DIR,
    NEG_DIR,
    OUTPUT_DIR_ALT as OUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    NUM_CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    LR0,
    WEIGHT_DECAY,
    MOMENTUM,
    GRAD_CLIP,
    CLASS_NAMES,
)

# ============================================================================
# Baseline constants (config-derived)
# ============================================================================
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
MODELS_DIR   = OUTPUT_DIR / "models"
METRICS_FILE = OUTPUT_DIR / "metrics_history.json"
CLASS_NAMES_DISPLAY = CLASS_NAMES[1:]

# ============================================================================
# Shared: dataset, transforms, eval, scheduling, checkpoints
# ============================================================================

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
        # Annotations pre-grouped by image id. The obvious alternative — filtering
        # with df[df["img_id"] == img_id] inside __getitem__ — rescans the whole
        # frame for every sample: O(rows) per item, O(rows²) per epoch. On the
        # 40,852-row train split that was ~7.7 ms of pure worker CPU per image.
        self._boxes  = {}
        self._labels = {}
        # Images live under a per-crop subdirectory (data/detector/<split>/<Crop>/),
        # so the crop is part of every image's path.
        self._crop   = {}
        for _img_id, _grp in df.groupby("img_id", sort=False):
            self._boxes[_img_id]  = _grp[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
            self._labels[_img_id] = _grp["integer_label"].to_numpy(dtype=np.int64)
            self._crop[_img_id]   = _grp["crop"].iloc[0]
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
        img_path = self.image_dir / self._crop[img_id] / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_t = v2.functional.to_image(img)           # uint8 [C,H,W]
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes  = self._boxes[img_id].copy()
        labels = self._labels[img_id].copy()                 # 1-indexed

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

def collate_fn(batch):
    return tuple(zip(*batch))

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / (union + 1e-6)

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)

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

        # A non-finite loss poisons every subsequent step: the weights go NaN and
        # never recover, so the remaining epochs are guaranteed waste. Fail loudly
        # and immediately instead — the last saved checkpoint predates the
        # divergence and stays valid. Learned the expensive way: a Faster R-CNN run
        # spent 24 of 30 epochs on NaN weights while the process looked healthy,
        # because the training log was block-buffered.
        if not torch.isfinite(losses):
            parts = ", ".join(f"{k}={v.item():.4g}" for k, v in loss_dict.items())
            raise RuntimeError(
                f"non-finite loss at epoch {epoch}, batch {batch_idx}: total={losses.item()} "
                f"({parts}). Training aborted before the weights were destroyed. "
                f"Most likely the backbone unfreezing into peak LR — compare "
                f"FREEZE_BACKBONE_EPOCHS with WARMUP_EPOCHS in the model's config.py."
            )

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

def load_checkpoint(ckpt_path: Path, model, optimizer, scheduler):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_map", 0.0), ckpt.get("metrics_history", {})

def _is_resumable(ckpt_path: Path) -> bool:
    """Return True only when checkpoint contains training state (not a weights-only export)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return isinstance(ckpt, dict) and all(
            k in ckpt for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict")
        )
    except Exception:
        return False

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
                    print(f"    [WARN]  seed {seed}: {exc}")
                    err += 1
                else:
                    ok += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} fetched  (ok={ok}, errors={err})")
        print(f"  Download complete: {ok} new, {err} failed")

    all_negs = sorted(neg_img_dir.glob("*.jpg"))[:num]
    print(f"  Hard negatives ready: {len(all_negs)} images  ({neg_img_dir})")
    print("  [OK]  Hard-negative setup complete")
    return all_negs

# ============================================================================
# Baseline-only
# ============================================================================

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

def write_final_eval(result: dict, model: nn.Module, split: str = "valid") -> None:
    """Persist final per-class AP + mAP as JSON for the cross-model benchmark aggregator.

    Writes outputs/fasterrcnn_output/final_eval.json and a copy into the shared
    outputs/benchmarks/ folder. Fully guarded by the caller — never breaks training.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_class = {CLASS_NAMES[c]: (None if math.isnan(v) else round(float(v), 5))
                 for c, v in result["per_class_ap"].items()}
    payload = {
        "model_name": "fasterrcnn_resnet50_fpn_v2",
        "architecture": "Faster RCNN ResNet-50-FPN-v2",
        "split": split,
        "map50": round(float(result["map50"]), 5),
        "num_params": sum(p.numel() for p in model.parameters()),
        "per_class_ap": per_class,
        "evaluated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    with open(OUTPUT_DIR / "final_eval.json", "w") as f:
        json.dump(payload, f, indent=2)
    bench_dir = PROJECT_ROOT / "outputs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_dir / "fasterrcnn_resnet50_fpn_v2.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Final eval saved → {OUTPUT_DIR / 'final_eval.json'}")

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
        print(f"  [WARN]  Figure generation skipped (missing: {exc})")
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
    print("  [OK]  fig_01  dataset overview")

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
            ax.text(cnt + 50, i, "  [WARN] < 30", va="center",
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
    print("  [OK]  fig_02  class distribution (train)")

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
    print("  [OK]  fig_03  cross-split distribution")

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
    print("  [OK]  fig_04  annotation density")

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
    print("  [OK]  fig_05  bounding-box spatial heatmap")

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
    # Tick labels are set afterwards rather than passed in: boxplot's `labels`
    # kwarg was renamed `tick_labels` in matplotlib 3.9 and removed in 3.11, and
    # requirements.txt allows >=3.8. Setting them on the axis works on every version.
    bp = ax_bp.boxplot(
        [tb2[tb2.crop == c].area_pct.values for c in crop_order],
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    ax_bp.set_xticks(range(1, len(crop_order) + 1))
    ax_bp.set_xticklabels(crop_order)
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
    print("  [OK]  fig_06  bounding-box geometry")

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
    print("  [OK]  fig_07  class imbalance")

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
    print("  [OK]  fig_08  training config table")

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
    print("  [OK]  fig_09  LR schedule + augmentation")

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
            print("  [OK]  fig_10  training metrics")
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
        print("  [OK]  fig_11  per-class AP")
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
        print(f"         [OK] {ptl_path.name}  ({ptl_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"         [WARN]  TorchScript mobile failed: {e}")
        try:
            # Fall back to plain .pt if mobile_optimizer fails
            scripted = torch.jit.script(model)
            pt_path = MODELS_DIR / "crop_disease_fasterrcnn_jit.pt"
            scripted.save(str(pt_path))
            print(f"         [OK] Saved plain TorchScript: {pt_path.name}")
        except Exception as e2:
            print(f"         [FAIL]  TorchScript also failed: {e2}")

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
        print(f"         [OK] {onnx_path.name}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"         [WARN]  ONNX export failed: {e}")

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
        print(f"         [OK] Backbone ExecuTorch: {pte_path.name}  "
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
            print(f"         [OK] Full model ExecuTorch: {pte_full.name}  "
                  f"({pte_full.stat().st_size / 1e6:.1f} MB)")
        except Exception as e_full:
            print(f"         [INFO]  Full ExecuTorch export skipped ({e_full})")
            print("            Use .ptl (TorchScript mobile) for full detection on device.")

    except Exception as e:
        print(f"         [WARN]  ExecuTorch failed: {e}")
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
    print(f"         [OK] {meta_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ── Exported artefacts in {MODELS_DIR} ───")
    for fp in sorted(MODELS_DIR.iterdir()):
        print(f"  {fp.name:<52}  {fp.stat().st_size / 1e6:>7.2f} MB")

def main_baseline() -> None:
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
    parser.add_argument("--batch-size",     type=int, default=None,
                        help=f"Override batch size (default: {BATCH_SIZE} MPS/CPU, {CUDA_BATCH_SIZE} CUDA)")
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
            print(f"  [FAIL]  No best checkpoint found at {best_pth}")
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
    is_cuda = device.type == "cuda"
    workers = 0 if is_mps else min(16 if is_cuda else 8, os.cpu_count() or 1)
    batch   = args.batch_size if args.batch_size is not None else (CUDA_BATCH_SIZE if is_cuda else BATCH_SIZE)

    if is_cuda:
        torch.backends.cudnn.benchmark = True

    # AMP: CUDA only (fp16 autocast + loss scaling for 30-50% throughput gain)
    # FRCNN_NO_AMP=1 disables mixed precision — arm B of the divergence bisect.
    _no_amp = os.environ.get("FRCNN_NO_AMP") == "1"
    scaler = (torch.amp.GradScaler("cuda")
              if device.type == "cuda" and not _no_amp else None)
    if _no_amp:
        print("  AMP disabled (FRCNN_NO_AMP=1)")

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
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=is_cuda,   # CUDA only; MPS uses unified memory
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=is_cuda,
        persistent_workers=(workers > 0),
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
        print(f"  [WARN]  last.pth found but not resumable (weights-only?). Fresh run.")

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
            print(f"  [*]  New best mAP@0.5: {best_map:.4f}")
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
    print(f"\n[OK]  Training complete!  ({total_time / 3600:.1f} h total)")
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

    # Persist per-class AP for the cross-model benchmark (guarded — never breaks training)
    try:
        write_final_eval(final_result, model, split="valid")
    except Exception as _exc:
        print(f"  [WARN]  benchmark summary skipped: {_exc}")

    # ── Step 3: Publication figures ───────────────────────────────────────────
    if not args.no_figures:
        print("\n─── Step 3/3: Generating publication figures ─────────────────")
        generate_figures(per_class_ap=final_result.get("per_class_ap"))

    # ── Step 4: Export ────────────────────────────────────────────────────────
    print("\n─── Step 4: Export (ExecuTorch / ONNX / TorchScript mobile) ─────")
    export_model(model)

# ============================================================================
# Ablation-only
# ============================================================================

CKPT_ROOT    = OUT_DIR / "checkpoints"

ABL_MODELS_DIR   = OUT_DIR / "models"

FIGS_DIR     = OUT_DIR / "figures"

RESULTS_PATH = OUT_DIR / "results.json"

ABL_EPOCHS_DEFAULT = 15       # shorter than the baseline pipeline; ablation is comparative

ABL_PATIENCE       = 5

ABL_WARMUP_EPOCHS  = 2

ABL_EVAL_EVERY     = 3        # evaluate mAP every N epochs

ABL_NUM_NEGATIVES  = 100      # hard-negative images

BENCH_RUNS     = 100      # iterations per speed benchmark

BENCH_WARMUP   = 20       # warm-up iterations before timing

SEED           = 42

ABL_CLASS_NAMES = [
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

ABL_CLASS_NAMES_DISPLAY = ABL_CLASS_NAMES[1:]   # 23 disease names, 0-indexed

@dataclass
class AblationConfig:
    config_id:    str
    backbone:     str              # 'mobilenet' | 'resnet50' | 'resnet101'
    num_proposals: int             # rpn_post_nms_top_n (train & test)
    nms_thresh:   float            # RPN NMS threshold (1.0 = disabled)
    anchor_sizes: Optional[Tuple]  # None → keep model defaults
    label:        str              # human-readable for plots
    color:        str
    marker:       str
    is_baseline:  bool = False
    # Filled in after building the model
    n_params:     int  = 0

ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        config_id="mobilenet_300",
        backbone="mobilenet",
        num_proposals=300,
        nms_thresh=0.7,
        anchor_sizes=None,
        label="MobileNetV3-FPN",
        color="#1f77b4",
        marker="o",
    ),
    AblationConfig(
        config_id="resnet50_100",
        backbone="resnet50",
        num_proposals=100,
        nms_thresh=0.7,
        anchor_sizes=None,
        label="ResNet50v2 (100 props)",
        color="#ff7f0e",
        marker="s",
    ),
    AblationConfig(
        config_id="resnet50_300",
        backbone="resnet50",
        num_proposals=300,
        nms_thresh=0.7,
        anchor_sizes=None,
        label="ResNet50v2 (baseline (*))",
        color="#2ca02c",
        marker="*",
        is_baseline=True,
    ),
    AblationConfig(
        config_id="resnet50_1000",
        backbone="resnet50",
        num_proposals=1000,
        nms_thresh=0.7,
        anchor_sizes=None,
        label="ResNet50v2 (1000 props)",
        color="#d62728",
        marker="D",
    ),
    AblationConfig(
        config_id="resnet50_no_nms",
        backbone="resnet50",
        num_proposals=300,
        nms_thresh=1.0,       # disabled
        anchor_sizes=None,
        label="ResNet50v2 (no NMS)",
        color="#9467bd",
        marker="^",
    ),
    AblationConfig(
        config_id="resnet50_small_anchors",
        backbone="resnet50",
        num_proposals=300,
        nms_thresh=0.7,
        anchor_sizes=(16, 32, 64, 128, 256),    # smaller: suits lesion detection
        label="ResNet50v2 (small anchors)",
        color="#8c564b",
        marker="v",
    ),
    AblationConfig(
        config_id="resnet101_300",
        backbone="resnet101",
        num_proposals=300,
        nms_thresh=0.7,
        anchor_sizes=None,
        label="ResNet101-FPN",
        color="#e377c2",
        marker="P",
    ),
]

CONFIG_MAP: Dict[str, AblationConfig] = {c.config_id: c for c in ABLATION_CONFIGS}

def build_ablation_model(cfg: AblationConfig, num_classes: int = NUM_CLASSES) -> FasterRCNN:
    """Build FasterRCNN variant described by cfg, with pretrained backbone."""
    if cfg.backbone == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    elif cfg.backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    elif cfg.backbone == "resnet101":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=ResNet101_Weights.DEFAULT,
            trainable_layers=3,
        )
        model = FasterRCNN(backbone, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    # Replace classification head for our num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Patch RPN proposal count
    model.rpn._post_nms_top_n["training"] = cfg.num_proposals
    model.rpn._post_nms_top_n["testing"]  = cfg.num_proposals

    # Patch RPN NMS threshold
    model.rpn.nms_thresh = cfg.nms_thresh

    # Custom anchor generator (ResNet50 only — 5 FPN levels)
    if cfg.anchor_sizes is not None and cfg.backbone == "resnet50":
        anchor_gen = AnchorGenerator(
            sizes=tuple((s,) for s in cfg.anchor_sizes),
            aspect_ratios=((0.5, 1.0, 2.0),) * len(cfg.anchor_sizes),
        )
        model.rpn.anchor_generator = anchor_gen

    cfg.n_params = sum(p.numel() for p in model.parameters())
    return model

def make_loaders(neg_paths: list, device: torch.device):
    """Build train and val DataLoaders."""
    df_train = _load_csv(TRAIN_CSV)
    df_val   = _load_csv(VAL_CSV)

    train_ds = CropDiseaseDataset(
        df_train, TRAIN_IMG_DIR,
        transform=get_train_transform(),
        neg_paths=neg_paths,
    )
    val_ds = CropDiseaseDataset(
        df_val, VAL_IMG_DIR,
        transform=get_val_transform(),
    )

    workers = 0 if device.type == "mps" else 4
    pin     = device.type not in ("mps", "cpu")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=workers, pin_memory=pin, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=workers, pin_memory=pin, collate_fn=collate_fn,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)

def ckpt_dir(config_id: str) -> Path:
    return CKPT_ROOT / config_id

def save_ablation_checkpoint(config_id: str, epoch: int, model, optimizer, scheduler,
                    best_map: float, history: dict, is_best: bool) -> None:
    d = ckpt_dir(config_id)
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map":   best_map,
        "history":    history,
    }
    last = d / "last.pth"
    torch.save(state, last)
    if is_best:
        shutil.copy2(last, d / "best.pth")
    if epoch % 5 == 0:
        shutil.copy2(last, d / f"epoch_{epoch:04d}.pth")

def _empty_history():
    return {"epoch": [], "train_total": [], "train_cls": [],
            "train_box_reg": [], "train_obj": [], "val_map50": [], "lr": []}

@torch.no_grad()
def benchmark_fps(model, device, img_size: int = IMG_SIZE,
                  warmup: int = BENCH_WARMUP, runs: int = BENCH_RUNS) -> dict:
    """Measure inference FPS (batch=1) and backbone-only latency."""
    model.eval()
    dummy = torch.rand(1, 3, img_size, img_size, device=device)

    # Warm up
    for _ in range(warmup):
        _ = model([dummy[0]])

    # Full model
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model([dummy[0]])
    if device.type == "cuda":
        torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) / runs * 1000.0

    # Backbone only
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model.backbone(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    backbone_ms = (time.perf_counter() - t0) / runs * 1000.0

    return {
        "full_ms":     round(full_ms, 2),
        "fps":         round(1000.0 / full_ms, 2),
        "backbone_ms": round(backbone_ms, 2),
    }

def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}

def save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

def train_config(cfg: AblationConfig, train_loader, val_loader,
                 device: torch.device, epochs: int, dry_run: bool) -> dict:
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  Config      : {cfg.config_id}  ({cfg.label})")
    print(f"  Backbone    : {cfg.backbone}   proposals={cfg.num_proposals}"
          f"  nms_thresh={cfg.nms_thresh}")
    anchors = str(cfg.anchor_sizes) if cfg.anchor_sizes else "default"
    print(f"  Anchors     : {anchors}")
    print(f"  Device      : {device}   epochs={epochs}")
    print(f"{sep}\n")

    model = build_ablation_model(cfg).to(device)
    print(f"  Parameters  : {cfg.n_params:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer, ABL_WARMUP_EPOCHS, epochs)

    # Resume
    last_ckpt = ckpt_dir(cfg.config_id) / "last.pth"
    start_epoch = 1
    best_map    = 0.0
    history     = _empty_history()
    if _is_resumable(last_ckpt):
        start_epoch, best_map, history = load_checkpoint(
            last_ckpt, model, optimizer, scheduler)
        start_epoch += 1
        print(f"  Resumed from epoch {start_epoch - 1}  (best mAP@0.5={best_map:.4f})")

    patience_count = 0
    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_total"].append(train_metrics["total"])
        history["train_cls"].append(train_metrics["classifier"])
        history["train_box_reg"].append(train_metrics["box_reg"])
        history["train_obj"].append(train_metrics["objectness"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        val_map = 0.0
        if epoch % ABL_EVAL_EVERY == 0 or epoch == epochs:
            val_res = evaluate(model, val_loader, device, num_classes=NUM_CLASSES - 1)
            val_map = val_res["map50"]
            print(f"  [Eval] epoch {epoch:3d}  mAP@0.5={val_map:.4f}")
        history["val_map50"].append(val_map)

        is_best = val_map > best_map and epoch % ABL_EVAL_EVERY == 0
        if is_best:
            best_map = val_map
            patience_count = 0
        elif epoch % ABL_EVAL_EVERY == 0:
            patience_count += 1

        save_ablation_checkpoint(cfg.config_id, epoch, model, optimizer, scheduler,
                        best_map, history, is_best)

        if dry_run and epoch >= 2:
            elapsed = time.time() - t_start
            est = elapsed / 2 * epochs
            print(f"\n  [DRY-RUN] 2 epochs in {elapsed:.1f}s"
                  f" → estimated {est/60:.0f} min for {epochs} epochs")
            break

        if patience_count >= ABL_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}  (no val improvement for {ABL_PATIENCE} evals)")
            break

    # Copy best weights to models/
    best_ckpt = ckpt_dir(cfg.config_id) / "best.pth"
    if best_ckpt.exists():
        ABL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, ABL_MODELS_DIR / f"{cfg.config_id}_best.pth")

    # Speed benchmark
    print(f"\n  Benchmarking {cfg.config_id} ...")
    bench = benchmark_fps(model, device)
    print(f"  FPS={bench['fps']:.1f}  full={bench['full_ms']:.1f}ms"
          f"  backbone={bench['backbone_ms']:.1f}ms")

    return {
        "config_id":   cfg.config_id,
        "label":       cfg.label,
        "backbone":    cfg.backbone,
        "n_params":    cfg.n_params,
        "num_proposals": cfg.num_proposals,
        "nms_thresh":  cfg.nms_thresh,
        "anchor_sizes": list(cfg.anchor_sizes) if cfg.anchor_sizes else None,
        "is_baseline": cfg.is_baseline,
        "best_map50":  best_map,
        "final_epoch": history["epoch"][-1] if history["epoch"] else 0,
        "fps":         bench["fps"],
        "full_ms":     bench["full_ms"],
        "backbone_ms": bench["backbone_ms"],
        "history":     history,
    }

def _set_rcparams():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "axes.axisbelow":     True,
    })

def _save_fig(fig, name: str, saved: list):
    path = FIGS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    saved.append(name)
    print(f"  Saved {name}")

def _draw_box(ax, x, y, w, h, text, fc, ec="none", tc="white", fs=10, alpha=1.0,
              radius=0.1):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=fc, edgecolor=ec,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, wrap=True)

def _arrow(ax, x1, y1, x2, y2, color="#444444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=16))

def generate_arch_figures() -> list:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _set_rcparams()
    saved = []

    # ── fig_arch_01 : End-to-end pipeline ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("Faster RCNN — End-to-End Detection Pipeline",
                 fontsize=14, fontweight="bold", y=0.98)

    blocks = [
        (0.3,  1.5, 1.8, 2.0, "Input\nImage\n640×640",           "#2C3E50"),
        (2.5,  1.5, 2.2, 2.0, "Backbone\n+ FPN\n(multi-scale)",  "#1A5276"),
        (5.2,  0.8, 2.0, 1.3, "P2  256-ch",                      "#154360"),
        (5.2,  2.2, 2.0, 1.3, "P3  256-ch",                      "#1B4F72"),
        (5.2,  3.6, 2.0, 1.3, "P4  256-ch",                      "#21618C"),
        (7.7,  1.5, 2.0, 2.0, "Region\nProposal\nNetwork (RPN)", "#6C3483"),
        (10.2, 1.5, 2.0, 2.0, "RoI\nAlign\n7×7",                 "#117A65"),
        (12.7, 1.5, 2.0, 2.0, "Box\nClassifier\n+ Regressor",    "#784212"),
        (14.9, 1.5, 0.8, 2.0, "Final\nDets",                     "#1E8449"),
    ]
    for (x, y, w, h, txt, fc) in blocks:
        _draw_box(ax, x, y, w, h, txt, fc, tc="white", fs=9)

    arrows = [(2.1, 2.5, 2.5, 2.5), (4.7, 1.45, 5.2, 1.45),
              (4.7, 2.85, 5.2, 2.85), (4.7, 4.25, 5.2, 4.25),
              (7.2, 2.5, 7.7, 2.5), (9.7, 2.5, 10.2, 2.5),
              (12.2, 2.5, 12.7, 2.5), (14.7, 2.5, 14.9, 2.5)]
    for (x1, y1, x2, y2) in arrows:
        _arrow(ax, x1, y1, x2, y2)

    ax.text(8.7, 0.35, "Region Proposals (top-N after NMS)",
            ha="center", va="center", fontsize=8.5, color="#6C3483", style="italic")
    _save_fig(fig, "fig_arch_01_pipeline.png", saved)

    # ── fig_arch_02 : Backbone comparison ─────────────────────────────────────
    backbones = ["MobileNetV3\n-Large FPN", "ResNet50\nFPN-v2 (*)", "ResNet101\nFPN"]
    params_m  = [19.04, 43.37, 60.35]    # millions
    depths    = [48, 50, 101]
    fpn_ch    = [256, 256, 256]
    pretrain  = ["IN-1k", "COCO (v2)", "IN-1k"]
    ap_coco   = [26.6, 46.7, 51.2]       # approximate COCO mAP from literature

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Backbone Architecture Comparison", fontsize=14, fontweight="bold")

    colors = ["#1f77b4", "#2ca02c", "#e377c2"]

    ax = axes[0]
    bars = ax.bar(backbones, params_m, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, params_m):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{v:.1f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Parameter Count (M)")
    ax.set_ylabel("Parameters (millions)")
    ax.set_ylim(0, 75)

    ax = axes[1]
    bars2 = ax.bar(backbones, ap_coco, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars2, ap_coco):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("COCO mAP (literature)")
    ax.set_ylabel("mAP@0.5:0.95 (%)")
    ax.set_ylim(0, 60)

    ax = axes[2]
    for i, (bk, p, d, pt) in enumerate(zip(backbones, params_m, depths, pretrain)):
        ax.scatter([p], [d], s=200, color=colors[i], marker="o", zorder=5, label=bk.replace("\n", " "))
        ax.annotate(f"  {pt}", (p, d), fontsize=8.5, va="center", color=colors[i])
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Network Depth (layers)")
    ax.set_title("Depth vs Parameters")
    ax.legend(fontsize=8.5, frameon=True, loc="lower right")

    plt.tight_layout()
    _save_fig(fig, "fig_arch_02_backbone_comparison.png", saved)

    # ── fig_arch_03 : RPN detail ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.suptitle("Region Proposal Network (RPN) Architecture",
                 fontsize=14, fontweight="bold")

    _draw_box(ax, 0.2, 2.5, 1.5, 2.0, "Feature\nMap\nHxWx256", "#1A5276", tc="white", fs=9)
    _draw_box(ax, 2.2, 2.5, 1.5, 2.0, "3×3\nConv\n256-d",      "#6C3483", tc="white", fs=9)
    _draw_box(ax, 4.5, 0.7, 1.6, 1.5, "1×1 Conv\nCls Head\n2k scores",  "#117A65", tc="white", fs=9)
    _draw_box(ax, 4.5, 3.3, 1.6, 1.5, "1×1 Conv\nReg Head\n4k deltas",  "#784212", tc="white", fs=9)
    _draw_box(ax, 7.2, 0.7, 2.0, 1.5, "Anchor\nScores\n(obj / bg)",     "#1E8449", tc="white", fs=9)
    _draw_box(ax, 7.2, 3.3, 2.0, 1.5, "Box\nDeltas\n(Δx,Δy,Δw,Δh)",    "#922B21", tc="white", fs=9)
    _draw_box(ax, 10.0, 1.7, 2.0, 2.5, "NMS\n(IoU<0.7)\n→ Top-N\nproposals", "#2C3E50", tc="white", fs=9)
    _draw_box(ax, 12.5, 2.0, 1.2, 2.0, "Region\nProposals", "#1A5276", tc="white", fs=9)

    for (x1, y1, x2, y2) in [
        (1.7, 3.5, 2.2, 3.5), (3.7, 3.5, 4.5, 1.45), (3.7, 3.5, 4.5, 4.05),
        (6.1, 1.45, 7.2, 1.45), (6.1, 4.05, 7.2, 4.05),
        (9.2, 1.45, 10.0, 2.3), (9.2, 4.05, 10.0, 3.7),
        (12.0, 2.95, 12.5, 2.95),
    ]:
        _arrow(ax, x1, y1, x2, y2)

    # Anchor visualization
    _draw_box(ax, 0.2, 0.1, 1.5, 1.8, "Anchors\nk=9/15\nper location", "#E67E22", tc="white", fs=9)
    _arrow(ax, 1.7, 1.0, 4.5, 1.0)
    ax.text(2.9, 1.15, "k anchors\nper cell", ha="center", va="bottom", fontsize=8.5, color="#E67E22")

    ax.text(7.0, 6.2,
            "k = anchors per FPN level per spatial location = |scales| × |ratios| = 3 × 3 = 9 (default) or 1 × 3 = 3 per size",
            ha="center", va="center", fontsize=8.5, color="#555555", style="italic")
    _save_fig(fig, "fig_arch_03_rpn_detail.png", saved)

    # ── fig_arch_04 : Anchor visualization ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Anchor Box Configuration: Default vs Small Anchors",
                 fontsize=14, fontweight="bold")

    def _draw_anchors(ax, sizes, title, img_size=640):
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_aspect("equal")
        ax.set_facecolor("#FDFEFE")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        cx, cy = img_size // 2, img_size // 2
        ratios = [0.5, 1.0, 2.0]
        colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]
        for si, (sz, col) in enumerate(zip(sizes, colors)):
            for ratio in ratios:
                w = sz * math.sqrt(ratio)
                h = sz / math.sqrt(ratio)
                rect = plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                     fill=False, edgecolor=col,
                                     linewidth=2.0, linestyle="-", alpha=0.85)
                ax.add_patch(rect)
            ax.plot([], [], color=col, linewidth=2, label=f"Scale {sz}px")

        ax.plot(cx, cy, "+", color="black", markersize=12, markeredgewidth=2)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        grid_step = img_size // 8
        for g in range(0, img_size + 1, grid_step):
            ax.axhline(g, color="#CCCCCC", linewidth=0.5)
            ax.axvline(g, color="#CCCCCC", linewidth=0.5)
        ax.invert_yaxis()

    _draw_anchors(axes[0], [32, 64, 128, 256, 512],
                  "Default Anchors (32–512 px)\n5 scales × 3 ratios = 15 per location")
    _draw_anchors(axes[1], [16, 32, 64, 128, 256],
                  "Small Anchors (16–256 px)\nOptimised for lesion detection")

    plt.tight_layout()
    _save_fig(fig, "fig_arch_04_anchor_visualization.png", saved)

    # ── fig_arch_05 : FPN structure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle("Feature Pyramid Network (FPN) — Multi-Scale Feature Extraction",
                 fontsize=14, fontweight="bold")

    # Bottom-up backbone columns (left side)
    bu_levels = [("C2", "160×160", "#D5E8D4"), ("C3", "80×80",  "#DAE8FC"),
                 ("C4", "40×40",   "#FFE6CC"), ("C5", "20×20",  "#F8CECC")]
    for i, (name, res, fc) in enumerate(bu_levels):
        y = 1.0 + i * 1.7
        _draw_box(ax, 0.3, y, 2.0, 1.2, f"{name}\n{res}\n(backbone)",
                  fc, ec="#999999", tc="#222222", fs=9)

    # Top-down FPN columns (right side)
    td_levels = [("P2", "160×160\n256-ch", "#27AE60"), ("P3", "80×80\n256-ch",  "#2980B9"),
                 ("P4", "40×40\n256-ch",   "#E67E22"), ("P5", "20×20\n256-ch",  "#C0392B"),
                 ("P6", "10×10\n256-ch",   "#8E44AD")]
    for i, (name, label, fc) in enumerate(td_levels):
        y = 0.6 + i * 1.6
        _draw_box(ax, 7.5, y, 2.2, 1.2, f"{name}\n{label}", fc, tc="white", fs=9)

    # Up-sample arrows (top-down path)
    for i in range(3, 0, -1):
        y_top = 0.6 + i * 1.6 + 0.6
        y_bot = 0.6 + (i - 1) * 1.6 + 0.6
        _arrow(ax, 8.6, y_top, 8.6, y_bot + 1.2)

    # Lateral connections
    for i in range(4):
        y = 1.0 + i * 1.7 + 0.6
        y_p = 0.6 + i * 1.6 + 0.6
        _arrow(ax, 2.3, y, 7.5, y_p, color="#7F8C8D")

    # P6 (from C5 via max-pool)
    _arrow(ax, 8.6, 0.6 + 3 * 1.6 + 1.2, 8.6, 0.6 + 4 * 1.6 + 0.6, color="#8E44AD")

    # Labels
    ax.text(1.3, 8.3, "Bottom-up\n(Backbone)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2C3E50")
    ax.text(8.6, 8.3, "Top-down FPN\n(Lateral + Upsample)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2C3E50")
    ax.text(5.0, 8.3, "1×1 Conv\nLateral\nConnections", ha="center", va="center",
            fontsize=9, color="#7F8C8D", style="italic")

    # RPN arrow from each P level
    for i in range(5):
        y = 0.6 + i * 1.6 + 0.6
        _draw_box(ax, 10.5, y, 1.8, 1.2, "RPN\n(per level)", "#6C3483", tc="white", fs=8)
        _arrow(ax, 9.7, y, 10.5, y)

    _save_fig(fig, "fig_arch_05_fpn_structure.png", saved)

    print(f"\n  Architecture figures: {len(saved)} saved to {FIGS_DIR}")
    return saved

def generate_perf_figures(results: dict) -> list:
    if not results:
        print("  No results to plot. Run training first.")
        return []

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _set_rcparams()
    saved = []

    cfg_order = [c.config_id for c in ABLATION_CONFIGS if c.config_id in results]
    if not cfg_order:
        return []

    labels   = [results[cid]["label"]    for cid in cfg_order]
    maps     = [results[cid]["best_map50"] for cid in cfg_order]
    fps_vals = [results[cid]["fps"]      for cid in cfg_order]
    params   = [results[cid]["n_params"] / 1e6 for cid in cfg_order]
    colors   = [CONFIG_MAP[cid].color    for cid in cfg_order]
    markers  = [CONFIG_MAP[cid].marker   for cid in cfg_order]
    is_base  = [results[cid]["is_baseline"] for cid in cfg_order]

    # ── fig_cmp_01 : mAP bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("mAP@0.5 Comparison Across Configurations", fontsize=13, fontweight="bold")

    x = np.arange(len(cfg_order))
    bars = ax.bar(x, maps, color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.8)
    for bar, v, baseline in zip(bars, maps, is_base):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}" + (" (*)" if baseline else ""),
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if baseline else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], rotation=0, fontsize=8.5)
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, max(maps) * 1.18)
    ax.set_title("Best Validation mAP@0.5 per Configuration")

    _save_fig(fig, "fig_cmp_01_map_bar.png", saved)

    # ── fig_cmp_02 : Speed-accuracy scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Speed–Accuracy Trade-off", fontsize=13, fontweight="bold")

    for cid, fp, mp, lb, col, mk, bl in zip(cfg_order, fps_vals, maps,
                                              labels, colors, markers, is_base):
        ms = 240 if bl else 150
        ax.scatter(fp, mp, s=ms, c=col, marker=mk, zorder=5,
                   edgecolors="black" if bl else "none", linewidths=1.5)
        ax.annotate(f"  {lb}", (fp, mp), fontsize=7.5,
                    va="center", color=col, fontweight="bold" if bl else "normal")

    ax.set_xlabel("Inference Speed (FPS, batch=1)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Speed vs Accuracy ((*) = selected baseline)")
    _save_fig(fig, "fig_cmp_02_speed_accuracy.png", saved)

    # ── fig_cmp_03 : Loss convergence curves ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Training Convergence Curves", fontsize=13, fontweight="bold")

    for cid, col, lb in zip(cfg_order, colors, labels):
        h = results[cid]["history"]
        if h.get("epoch"):
            axes[0].plot(h["epoch"], h["train_total"], color=col, label=lb, linewidth=1.8)
            map_ep = [(e, m) for e, m in zip(h["epoch"], h["val_map50"]) if m > 0]
            if map_ep:
                ep_m, map_m = zip(*map_ep)
                axes[1].plot(ep_m, map_m, color=col, label=lb, marker="o",
                             markersize=5, linewidth=1.8)

    for ax, title, ylabel in zip(
        axes,
        ["Total Training Loss", "Validation mAP@0.5"],
        ["Loss", "mAP@0.5"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7.5, loc="upper right" if "Loss" in title else "lower right")

    plt.tight_layout()
    _save_fig(fig, "fig_cmp_03_convergence.png", saved)

    # ── fig_cmp_04 : Proposal count ablation ──────────────────────────────────
    prop_configs = ["resnet50_100", "resnet50_300", "resnet50_1000"]
    p_vals = [results[c]["num_proposals"] for c in prop_configs if c in results]
    m_vals = [results[c]["best_map50"]    for c in prop_configs if c in results]
    f_vals = [results[c]["fps"]           for c in prop_configs if c in results]

    if len(p_vals) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Proposal Count Ablation (ResNet50-FPN-v2)",
                     fontsize=13, fontweight="bold")

        col_line = "#2ca02c"
        axes[0].plot(p_vals, m_vals, "o-", color=col_line, linewidth=2.2, markersize=8)
        for pv, mv in zip(p_vals, m_vals):
            axes[0].annotate(f"  {mv:.3f}", (pv, mv), fontsize=9)
        axes[0].set_xlabel("Number of RPN Proposals")
        axes[0].set_ylabel("mAP@0.5")
        axes[0].set_title("mAP vs Proposals")
        axes[0].set_xscale("log")

        axes[1].plot(p_vals, f_vals, "s--", color="#d62728", linewidth=2.2, markersize=8)
        for pv, fv in zip(p_vals, f_vals):
            axes[1].annotate(f"  {fv:.1f}", (pv, fv), fontsize=9)
        axes[1].set_xlabel("Number of RPN Proposals")
        axes[1].set_ylabel("FPS")
        axes[1].set_title("Speed vs Proposals")
        axes[1].set_xscale("log")

        plt.tight_layout()
        _save_fig(fig, "fig_cmp_04_proposal_ablation.png", saved)

    # ── fig_cmp_05 : Radar chart ───────────────────────────────────────────────
    metrics_keys = ["mAP@0.5 (norm)", "FPS (norm)", "Efficiency (norm)",
                    "mAP×10"]
    raw = {
        "mAP@0.5 (norm)":   {c: results[c]["best_map50"] for c in cfg_order},
        "FPS (norm)":        {c: results[c]["fps"]        for c in cfg_order},
        "Efficiency (norm)": {c: results[c]["fps"] * results[c]["best_map50"]
                               for c in cfg_order},
        "mAP×10":            {c: results[c]["best_map50"] * 10 for c in cfg_order},
    }
    # Normalise each axis to [0,1]
    norm_data = {}
    for mk in metrics_keys:
        vals = list(raw[mk].values())
        mn, mx = min(vals), max(vals)
        if mx > mn:
            norm_data[mk] = {c: (raw[mk][c] - mn) / (mx - mn) for c in cfg_order}
        else:
            norm_data[mk] = {c: 0.5 for c in cfg_order}

    N = len(metrics_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
    fig.suptitle("Multi-Metric Comparison (Normalised)", fontsize=13, fontweight="bold")

    for cid, col, lb in zip(cfg_order, colors, labels):
        vals = [norm_data[mk][cid] for mk in metrics_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=col, linewidth=2, label=lb)
        ax.fill(angles, vals, color=col, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_keys, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.5)
    _save_fig(fig, "fig_cmp_05_radar.png", saved)

    # ── fig_cmp_06 : Params vs mAP ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Model Complexity vs Detection Performance",
                 fontsize=13, fontweight="bold")

    for cid, pm, mp, lb, col, mk, bl in zip(
            cfg_order, params, maps, labels, colors, markers, is_base):
        ms = 250 if bl else 150
        ax.scatter(pm, mp, s=ms, c=col, marker=mk, zorder=5,
                   edgecolors="black" if bl else "none", linewidths=1.5)
        ax.annotate(f"  {lb}", (pm, mp), fontsize=7.5, va="center", color=col)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Parameters vs mAP@0.5 ((*) = selected configuration)")
    _save_fig(fig, "fig_cmp_06_params_vs_map.png", saved)

    # ── fig_cmp_07 : Inference time breakdown ──────────────────────────────────
    backbone_ms = [results[c]["backbone_ms"] for c in cfg_order]
    full_ms     = [results[c]["full_ms"]     for c in cfg_order]
    head_ms     = [max(0.0, f - b) for f, b in zip(full_ms, backbone_ms)]

    x = np.arange(len(cfg_order))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Inference Time Breakdown (ms, batch=1)",
                 fontsize=13, fontweight="bold")

    bars_b = ax.bar(x, backbone_ms, width=0.5, label="Backbone", color="#2980B9", zorder=3)
    bars_h = ax.bar(x, head_ms, width=0.5, bottom=backbone_ms,
                    label="RPN+RoI Head", color="#E67E22", zorder=3)

    for xi, (bms, fms) in enumerate(zip(backbone_ms, full_ms)):
        ax.text(xi, fms + 1.0, f"{fms:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8.5)
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.set_title("Backbone vs Head Inference Latency")
    _save_fig(fig, "fig_cmp_07_inference_breakdown.png", saved)

    # ── fig_cmp_08 : NMS ablation comparison ──────────────────────────────────
    nms_group = ["resnet50_300", "resnet50_no_nms"]
    nms_avail = [c for c in nms_group if c in results]
    anchor_group = ["resnet50_300", "resnet50_small_anchors"]
    anchor_avail = [c for c in anchor_group if c in results]

    if len(nms_avail) + len(anchor_avail) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("NMS & Anchor Ablation (ResNet50-FPN-v2)",
                     fontsize=13, fontweight="bold")

        if len(nms_avail) >= 2:
            lbls = [results[c]["label"] for c in nms_avail]
            ms   = [results[c]["best_map50"] for c in nms_avail]
            cols = [CONFIG_MAP[c].color for c in nms_avail]
            axes[0].bar(lbls, ms, color=cols, width=0.4, zorder=3)
            for i, mv in enumerate(ms):
                axes[0].text(i, mv + 0.002, f"{mv:.3f}", ha="center",
                             va="bottom", fontsize=11, fontweight="bold")
            axes[0].set_ylabel("mAP@0.5")
            axes[0].set_title("Effect of NMS Threshold")
            axes[0].set_ylim(0, max(ms) * 1.2)

        if len(anchor_avail) >= 2:
            lbls = [results[c]["label"] for c in anchor_avail]
            ms   = [results[c]["best_map50"] for c in anchor_avail]
            cols = [CONFIG_MAP[c].color for c in anchor_avail]
            axes[1].bar(lbls, ms, color=cols, width=0.4, zorder=3)
            for i, mv in enumerate(ms):
                axes[1].text(i, mv + 0.002, f"{mv:.3f}", ha="center",
                             va="bottom", fontsize=11, fontweight="bold")
            axes[1].set_ylabel("mAP@0.5")
            axes[1].set_title("Effect of Anchor Scale")
            axes[1].set_ylim(0, max(ms) * 1.2)

        plt.tight_layout()
        _save_fig(fig, "fig_cmp_08_nms_anchor_ablation.png", saved)

    print(f"\n  Performance figures: {len(saved)} saved to {FIGS_DIR}")
    return saved

def generate_table_figures(results: dict) -> list:
    if not results:
        return []

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _set_rcparams()
    saved = []

    cfg_order = [c.config_id for c in ABLATION_CONFIGS if c.config_id in results]

    # ── fig_tbl_01 : Main results table ───────────────────────────────────────
    col_headers = ["Configuration", "Backbone", "Proposals",
                   "NMS Thresh", "Anchor Sizes", "Params (M)",
                   "mAP@0.5", "FPS", "Baseline"]
    rows = []
    best_map = max(results[c]["best_map50"] for c in cfg_order)

    for cid in cfg_order:
        r = results[cid]
        anc = str(r["anchor_sizes"]) if r["anchor_sizes"] else "default"
        rows.append([
            r["label"],
            r["backbone"],
            str(r["num_proposals"]),
            f"{r['nms_thresh']:.1f}",
            anc,
            f"{r['n_params'] / 1e6:.1f}",
            f"{r['best_map50']:.4f}",
            f"{r['fps']:.1f}",
            "(*)" if r["is_baseline"] else "",
        ])

    fig, ax = plt.subplots(figsize=(20, len(rows) * 0.75 + 2))
    ax.axis("off")
    fig.suptitle("Table 1: Faster RCNN Configuration Comparison",
                 fontsize=14, fontweight="bold", y=0.98)

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best mAP and baseline
    for i, (cid, row) in enumerate(zip(cfg_order, rows)):
        r = results[cid]
        fc = "#D5F5E3" if r["is_baseline"] else (
             "#EBF5FB" if float(r["best_map50"]) == best_map else "#FDFEFE")
        for j in range(len(col_headers)):
            tbl[(i + 1, j)].set_facecolor(fc)
        if float(r["best_map50"]) == best_map:
            tbl[(i + 1, 6)].set_text_props(fontweight="bold", color="#1E8449")

    plt.tight_layout()
    _save_fig(fig, "fig_tbl_01_main_results.png", saved)

    # ── fig_tbl_02 : Speed comparison ─────────────────────────────────────────
    speed_headers = ["Configuration", "Backbone", "Params (M)",
                     "Full Model (ms)", "Backbone (ms)", "Head (ms)", "FPS"]
    speed_rows = []
    for cid in cfg_order:
        r = results[cid]
        head = max(0.0, r["full_ms"] - r["backbone_ms"])
        speed_rows.append([
            r["label"],
            r["backbone"],
            f"{r['n_params'] / 1e6:.1f}",
            f"{r['full_ms']:.1f}",
            f"{r['backbone_ms']:.1f}",
            f"{head:.1f}",
            f"{r['fps']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(18, len(speed_rows) * 0.75 + 2))
    ax.axis("off")
    fig.suptitle("Table 2: Inference Speed Comparison (batch=1, single image)",
                 fontsize=14, fontweight="bold", y=0.98)

    tbl2 = ax.table(cellText=speed_rows, colLabels=speed_headers,
                    cellLoc="center", loc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9.5)
    tbl2.scale(1.0, 1.6)

    for j in range(len(speed_headers)):
        tbl2[(0, j)].set_facecolor("#6C3483")
        tbl2[(0, j)].set_text_props(color="white", fontweight="bold")

    fastest_fps = max(float(r[6]) for r in speed_rows)
    for i, row in enumerate(speed_rows):
        fc = "#FDEDEC" if float(row[6]) == fastest_fps else "#FDFEFE"
        for j in range(len(speed_headers)):
            tbl2[(i + 1, j)].set_facecolor(fc)

    plt.tight_layout()
    _save_fig(fig, "fig_tbl_02_speed_comparison.png", saved)

    return saved

def print_latex_tables(results: dict) -> None:
    if not results:
        return

    cfg_order = [c.config_id for c in ABLATION_CONFIGS if c.config_id in results]
    print("\n" + "═" * 80)
    print("  LaTeX Table 1: Main Results")
    print("═" * 80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Faster RCNN Configuration Ablation on Crop Disease Detection Dataset}")
    print(r"\label{tab:fasterrcnn_ablation}")
    print(r"\begin{tabular}{lllrrrr}")
    print(r"\hline")
    print(r"\textbf{Config} & \textbf{Backbone} & \textbf{Props} & "
          r"\textbf{Params (M)} & \textbf{mAP@0.5} & \textbf{FPS} \\")
    print(r"\hline")

    for cid in cfg_order:
        r = results[cid]
        star = r" $\star$" if r["is_baseline"] else ""
        bold_s = r"\textbf{" if r["is_baseline"] else ""
        bold_e = r"}" if r["is_baseline"] else ""
        print(
            f"{bold_s}{r['label']}{star}{bold_e} & "
            f"{r['backbone']} & "
            f"{r['num_proposals']} & "
            f"{r['n_params']/1e6:.1f} & "
            f"{r['best_map50']:.4f} & "
            f"{r['fps']:.1f} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save to file
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    latex_path = FIGS_DIR / "table_ablation.tex"
    with open(latex_path, "w") as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Faster RCNN Ablation Study}" + "\n")
        f.write(r"\label{tab:fasterrcnn_ablation}" + "\n")
        f.write(r"\begin{tabular}{lllrrrr}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Config} & \textbf{Backbone} & \textbf{Proposals} & "
                r"\textbf{Params (M)} & \textbf{mAP@0.5} & \textbf{FPS} \\" + "\n")
        f.write(r"\hline" + "\n")
        for cid in cfg_order:
            r = results[cid]
            star = r" $\star$" if r["is_baseline"] else ""
            f.write(
                f"{r['label']}{star} & "
                f"{r['backbone']} & "
                f"{r['num_proposals']} & "
                f"{r['n_params']/1e6:.1f} & "
                f"{r['best_map50']:.4f} & "
                f"{r['fps']:.1f} \\\\\n"
            )
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\n  LaTeX table saved → {latex_path}")

def main_ablation():
    parser = argparse.ArgumentParser(
        description="Faster RCNN Ablation Study — crop disease detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIG_MAP.keys()), default=None,
        metavar="CONFIG_ID",
        help="Subset of configs to train (default: all 7). "
             f"Choices: {list(CONFIG_MAP.keys())}",
    )
    parser.add_argument("--epochs", type=int, default=ABL_EPOCHS_DEFAULT,
                        help="Training epochs per config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 epochs per config for timing estimate")
    parser.add_argument("--skip-negatives", action="store_true",
                        help="Skip hard-negative download (use cached)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate all figures from existing results.json")
    parser.add_argument("--arch-figures", action="store_true",
                        help="Generate architecture figures only (no training)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ── Create output directories ──────────────────────────────────────────────
    for d in [OUT_DIR, CKPT_ROOT, ABL_MODELS_DIR, FIGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Architecture figures (always generated unless --figures-only without arch) ─
    if not args.figures_only:
        print("\n" + "═" * 66)
        print("  Generating architecture figures …")
        print("═" * 66)
        generate_arch_figures()

    if args.arch_figures:
        print("\n  Architecture figures complete. Exiting (--arch-figures).")
        return

    # ── Figures-only mode ─────────────────────────────────────────────────────
    if args.figures_only:
        results = load_results()
        if not results:
            print("  No results.json found. Run training first, or use --arch-figures.")
            return
        print("\n" + "═" * 66)
        print("  Regenerating figures from results.json …")
        print("═" * 66)
        generate_arch_figures()
        generate_perf_figures(results)
        generate_table_figures(results)
        print_latex_tables(results)
        return

    # ── Determine which configs to run ────────────────────────────────────────
    run_ids = args.configs if args.configs else [c.config_id for c in ABLATION_CONFIGS]
    cfgs_to_run = [CONFIG_MAP[cid] for cid in run_ids]

    # ── Validate dataset paths ─────────────────────────────────────────────────
    for p in [TRAIN_CSV, VAL_CSV, TRAIN_IMG_DIR, VAL_IMG_DIR]:
        if not p.exists():
            print(f"  ERROR: required path missing: {p}")
            print("  Please verify the dataset/ directory structure.")
            raise SystemExit(1)

    # ── Hard negatives ─────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("  Preparing hard-negative images …")
    print("═" * 66)
    neg_paths = prepare_hard_negatives(ABL_NUM_NEGATIVES, skip=args.skip_negatives)

    # ── Device ────────────────────────────────────────────────────────────────
    device = resolve_device()
    print(f"\n  Device: {device}  |  torchvision: {torchvision.__version__}")

    # ── DataLoaders (shared across all configs) ────────────────────────────────
    print("\n  Building DataLoaders …")
    train_loader, val_loader, n_train, n_val = make_loaders(neg_paths, device)
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    # ── Train each configuration ───────────────────────────────────────────────
    results = load_results()
    total   = len(cfgs_to_run)

    for idx, cfg in enumerate(cfgs_to_run, 1):
        print(f"\n{'═'*66}")
        print(f"  Config {idx}/{total}: {cfg.config_id}")
        print(f"{'═'*66}")

        result = train_config(cfg, train_loader, val_loader, device,
                              epochs=args.epochs, dry_run=args.dry_run)
        results[cfg.config_id] = result
        save_results(results)
        print(f"  Saved results ({cfg.config_id}  mAP@0.5={result['best_map50']:.4f})")

    # ── Final figures ──────────────────────────────────────────────────────────
    if not args.no_figures and results:
        print("\n" + "═" * 66)
        print("  Generating performance figures …")
        print("═" * 66)
        generate_perf_figures(results)
        generate_table_figures(results)
        print_latex_tables(results)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print(f"  {'Config ID':<30}  {'mAP@0.5':>8}  {'FPS':>7}  {'Params':>10}")
    print("  " + "─" * 60)
    for cid in [c.config_id for c in ABLATION_CONFIGS if c.config_id in results]:
        r = results[cid]
        star = " (*)" if r["is_baseline"] else ""
        print(f"  {cid:<30}  {r['best_map50']:>8.4f}  "
              f"{r['fps']:>7.1f}  {r['n_params']/1e6:>9.1f}M{star}")
    print("═" * 66)
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Results JSON    : {RESULTS_PATH}")
    print(f"  Figures         : {FIGS_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Dispatch to the baseline or the ablation pipeline.

    --mode is parsed here and stripped from sys.argv, so each pipeline's own
    argument parser sees exactly the arguments it did before the merge.
    """
    import sys
    mode = "baseline"
    argv = sys.argv[1:]
    if "--mode" in argv:
        i = argv.index("--mode")
        if i + 1 >= len(argv):
            sys.exit("--mode requires a value: baseline | ablation")
        mode = argv[i + 1]
        del argv[i:i + 2]
        sys.argv = [sys.argv[0]] + argv
    if mode == "baseline":
        main_baseline()
    elif mode == "ablation":
        main_ablation()
    else:
        sys.exit(f"unknown --mode {mode!r}: expected 'baseline' or 'ablation'")


if __name__ == "__main__":
    main()