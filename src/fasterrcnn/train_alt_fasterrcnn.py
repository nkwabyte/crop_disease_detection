#!/usr/bin/env python3
"""
train_alt_fasterrcnn.py — Faster RCNN Ablation Study for Crop Disease Detection

Compares 7 Faster RCNN configurations across backbone depth, proposal count,
NMS policy, and anchor scale — mirroring the comparative analysis of the original
Faster RCNN paper (Ren et al., 2015) for publication justification.

Configurations
--------------
  1.  mobilenet_300          MobileNetV3-FPN,  300 proposals  (lightweight)
  2.  resnet50_100           ResNet50-FPN-v2,  100 proposals
  3.  resnet50_300       ★   ResNet50-FPN-v2,  300 proposals  (selected baseline)
  4.  resnet50_1000          ResNet50-FPN-v2, 1000 proposals
  5.  resnet50_no_nms        ResNet50-FPN-v2,  300 proposals, NMS disabled
  6.  resnet50_small_anchors ResNet50-FPN-v2,  300 proposals, anchors 16-256 px
  7.  resnet101_300          ResNet101-FPN,    300 proposals  (heavier backbone)

Outputs
-------
  outputs/alt_fasterrcnn_output/
  ├── checkpoints/{config_id}/  last.pth  best.pth  epoch_NNNN.pth
  ├── models/{config_id}_best.pth
  ├── figures/
  │   ├── fig_arch_01_pipeline.png
  │   ├── fig_arch_02_backbone_comparison.png
  │   ├── fig_arch_03_rpn_detail.png
  │   ├── fig_arch_04_anchor_visualization.png
  │   ├── fig_arch_05_fpn_structure.png
  │   ├── fig_cmp_01_map_bar.png
  │   ├── fig_cmp_02_speed_accuracy.png
  │   ├── fig_cmp_03_convergence.png
  │   ├── fig_cmp_04_proposal_ablation.png
  │   ├── fig_cmp_05_radar.png
  │   ├── fig_cmp_06_params_vs_map.png
  │   ├── fig_tbl_01_main_results.png
  │   └── fig_tbl_02_speed_comparison.png
  └── results.json

Usage
-----
  python train_alt_fasterrcnn.py                              # all 7 configs
  python train_alt_fasterrcnn.py --configs resnet50_300       # single config
  python train_alt_fasterrcnn.py --configs mobilenet_300 resnet101_300
  python train_alt_fasterrcnn.py --epochs 5                   # quick test
  python train_alt_fasterrcnn.py --figures-only               # regen figures
  python train_alt_fasterrcnn.py --arch-figures               # arch figs only
  python train_alt_fasterrcnn.py --dry-run                    # 2-ep timing
"""

import argparse
import json
import math
import os
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torchvision
from torchvision import tv_tensors
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import v2

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Paths & directory structure
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"   # shared with train_fasterrcnn.py
OUT_DIR      = PROJECT_ROOT / "outputs" / "alt_fasterrcnn_output"
CKPT_ROOT    = OUT_DIR / "checkpoints"
MODELS_DIR   = OUT_DIR / "models"
FIGS_DIR     = OUT_DIR / "figures"
RESULTS_PATH = OUT_DIR / "results.json"

TRAIN_CSV     = DATASET_DIR / "final_train_labels.csv"
VAL_CSV       = DATASET_DIR / "final_validate_labels.csv"
TRAIN_IMG_DIR = DATASET_DIR / "train"
VAL_IMG_DIR   = DATASET_DIR / "validate"


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

NUM_CLASSES    = 24       # 23 disease classes + background (label 0)
IMG_SIZE       = 640
EPOCHS_DEFAULT = 15       # shorter than main pipeline; ablation is comparative
PATIENCE       = 5
BATCH_SIZE     = 4
LR0            = 5e-3
WEIGHT_DECAY   = 5e-4
MOMENTUM       = 0.9
WARMUP_EPOCHS  = 2
GRAD_CLIP      = 10.0
EVAL_EVERY     = 3        # evaluate mAP every N epochs
NUM_NEGATIVES  = 100      # hard-negative images
BENCH_RUNS     = 100      # iterations per speed benchmark
BENCH_WARMUP   = 20       # warm-up iterations before timing
SEED           = 42

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
CLASS_NAMES_DISPLAY = CLASS_NAMES[1:]   # 23 disease names, 0-indexed


# ══════════════════════════════════════════════════════════════════════════════
# Ablation configurations
# ══════════════════════════════════════════════════════════════════════════════

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
        label="ResNet50v2 (baseline ★)",
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


# ══════════════════════════════════════════════════════════════════════════════
# Hard negatives  (shared cache with train_fasterrcnn.py)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_hard_negatives(num: int = NUM_NEGATIVES, skip: bool = False) -> list:
    if skip:
        neg_img_dir = NEG_DIR / "images"
        return sorted(neg_img_dir.glob("*.jpg"))[:num] if neg_img_dir.exists() else []

    neg_img_dir = NEG_DIR / "images"
    neg_img_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    seeds = random.sample(range(1, 2000), num)

    pending = [
        (s, neg_img_dir / f"negative_{s:04d}.jpg")
        for s in seeds
        if not (neg_img_dir / f"negative_{s:04d}.jpg").exists()
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
            for done, fut in enumerate(as_completed(futures), 1):
                _, exc = fut.result()
                if exc:
                    err += 1
                else:
                    ok += 1
                if done % 25 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} done  (ok={ok}, err={err})")

    all_negs = sorted(neg_img_dir.glob("*.jpg"))[:num]
    print(f"  Hard negatives ready: {len(all_negs)}")
    return all_negs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df["x1"] < df["x2"]) & (df["y1"] < df["y2"])].copy()
    df["img_id"] = df["fname"].apply(lambda x: x.rsplit(".", 1)[0])
    return df


def _sanitise_boxes(boxes: torch.Tensor, labels: torch.Tensor, h: int, w: int):
    if boxes.numel() == 0:
        return boxes, labels
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], labels[keep]


def _make_target(boxes: torch.Tensor, labels: torch.Tensor, idx: int) -> dict:
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() else torch.zeros(0)
    return {
        "boxes":    boxes,
        "labels":   labels,
        "image_id": torch.tensor([idx]),
        "area":     area,
        "iscrowd":  torch.zeros(labels.shape[0], dtype=torch.int64),
    }


class CropDiseaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path,
                 transform=None, neg_paths: Optional[list] = None):
        self.image_ids = df["img_id"].unique()
        self.df        = df
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.neg_paths = neg_paths or []
        self._n_pos    = len(self.image_ids)

    def __len__(self):
        return self._n_pos + len(self.neg_paths)

    def __getitem__(self, idx: int):
        if idx < self._n_pos:
            return self._get_positive(idx)
        return self._get_negative(idx - self._n_pos)

    def _get_positive(self, idx: int):
        img_id  = self.image_ids[idx]
        records = self.df[self.df["img_id"] == img_id]
        img_path = self.image_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_t = v2.functional.to_image(img)
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes  = records[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        labels = records["integer_label"].values.astype(np.int64)

        boxes_tv = tv_tensors.BoundingBoxes(
            torch.as_tensor(boxes), format="XYXY", canvas_size=(h, w))
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            img_t, boxes_tv = self.transform(img_t, boxes_tv)
        else:
            img_t = v2.functional.to_dtype(img_t, torch.float32, scale=True)

        boxes_out = torch.as_tensor(boxes_tv, dtype=torch.float32)
        boxes_out, labels_t = _sanitise_boxes(boxes_out, labels_t, h, w)
        return img_t, _make_target(boxes_out, labels_t, idx)

    def _get_negative(self, neg_idx: int):
        img_path = self.neg_paths[neg_idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        img_t = v2.functional.to_image(img)
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes_tv = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32), format="XYXY", canvas_size=(h, w))
        if self.transform:
            img_t, boxes_tv = self.transform(img_t, boxes_tv)
        else:
            img_t = v2.functional.to_dtype(img_t, torch.float32, scale=True)

        empty = torch.zeros((0, 4), dtype=torch.float32)
        return img_t, _make_target(empty, torch.zeros((0,), dtype=torch.int64),
                                   self._n_pos + neg_idx)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transform():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_val_transform():
    return v2.Compose([v2.ToDtype(torch.float32, scale=True)])


# ══════════════════════════════════════════════════════════════════════════════
# Model building
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Device & dataloader helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
# Checkpoint utilities  (per-config subdirectory)
# ══════════════════════════════════════════════════════════════════════════════

def ckpt_dir(config_id: str) -> Path:
    return CKPT_ROOT / config_id


def save_checkpoint(config_id: str, epoch: int, model, optimizer, scheduler,
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


def _is_resumable(path: Path) -> bool:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(ck, dict) and all(
            k in ck for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict"))
    except Exception:
        return False


def load_checkpoint(path: Path, model, optimizer, scheduler):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    return ck["epoch"], ck.get("best_map", 0.0), ck.get("history", _empty_history())


def _empty_history():
    return {"epoch": [], "train_total": [], "train_cls": [],
            "train_box_reg": [], "train_obj": [], "val_map50": [], "lr": []}


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, optimizer, loader, device, epoch: int) -> dict:
    model.train()
    totals = {"total": 0.0, "classifier": 0.0, "box_reg": 0.0,
              "objectness": 0.0, "rpn_box_reg": 0.0}
    n = len(loader)

    for bi, (images, targets) in enumerate(loader):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in targets]
        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        totals["total"] += losses.item()
        for key, val in loss_dict.items():
            short = key.replace("loss_", "")
            if short in totals:
                totals[short] += val.item()

        if (bi + 1) % max(1, n // 4) == 0 or bi == n - 1:
            lr  = optimizer.param_groups[0]["lr"]
            pct = (bi + 1) / n * 100
            print(f"    ep {epoch:3d} [{pct:5.1f}%] "
                  f"loss={totals['total']/(bi+1):.4f}  lr={lr:.2e}")

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 23) -> dict:
    """VOC-style mAP@0.5, pure NumPy (no torchmetrics)."""
    model.eval()
    class_dets: dict = defaultdict(list)
    class_ngt:  dict = defaultdict(int)

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, gt in zip(preds, targets):
            gt_boxes  = gt["boxes"].cpu().numpy()
            gt_labels = gt["labels"].cpu().numpy()
            p_boxes   = pred["boxes"].cpu().numpy()
            p_scores  = pred["scores"].cpu().numpy()
            p_labels  = pred["labels"].cpu().numpy()

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
                matched  = set()
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
        tp   = np.array([d[1] for d in dets], dtype=np.float32)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(1 - tp)
        rec  = tp_c / ngt
        prec = tp_c / (tp_c + fp_c)
        ap   = sum(float(np.max(prec[rec >= t])) if len(prec[rec >= t]) else 0.0
                   for t in np.linspace(0, 1, 11)) / 11.0
        aps[c] = ap

    valid = [v for v in aps.values() if not math.isnan(v)]
    return {"map50": float(np.mean(valid)) if valid else 0.0, "per_class_ap": aps}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Speed benchmark
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Results I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop for one configuration
# ══════════════════════════════════════════════════════════════════════════════

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
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, epochs)

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
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            val_res = evaluate(model, val_loader, device, num_classes=NUM_CLASSES - 1)
            val_map = val_res["map50"]
            print(f"  [Eval] epoch {epoch:3d}  mAP@0.5={val_map:.4f}")
        history["val_map50"].append(val_map)

        is_best = val_map > best_map and epoch % EVAL_EVERY == 0
        if is_best:
            best_map = val_map
            patience_count = 0
        elif epoch % EVAL_EVERY == 0:
            patience_count += 1

        save_checkpoint(cfg.config_id, epoch, model, optimizer, scheduler,
                        best_map, history, is_best)

        if dry_run and epoch >= 2:
            elapsed = time.time() - t_start
            est = elapsed / 2 * epochs
            print(f"\n  [DRY-RUN] 2 epochs in {elapsed:.1f}s"
                  f" → estimated {est/60:.0f} min for {epochs} epochs")
            break

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}  (no val improvement for {PATIENCE} evals)")
            break

    # Copy best weights to models/
    best_ckpt = ckpt_dir(cfg.config_id) / "best.pth"
    if best_ckpt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, MODELS_DIR / f"{cfg.config_id}_best.pth")

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


# ══════════════════════════════════════════════════════════════════════════════
# rcParams helper
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Architecture figures  (generated without training data)
# ══════════════════════════════════════════════════════════════════════════════

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
    backbones = ["MobileNetV3\n-Large FPN", "ResNet50\nFPN-v2 ★", "ResNet101\nFPN"]
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


# ══════════════════════════════════════════════════════════════════════════════
# Performance comparison figures  (require results.json)
# ══════════════════════════════════════════════════════════════════════════════

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
                f"{v:.3f}" + (" ★" if baseline else ""),
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
    ax.set_title("Speed vs Accuracy (★ = selected baseline)")
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
    ax.set_title("Parameters vs mAP@0.5 (★ = selected configuration)")
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


# ══════════════════════════════════════════════════════════════════════════════
# Paper-style table figures
# ══════════════════════════════════════════════════════════════════════════════

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
            "★" if r["is_baseline"] else "",
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


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table output
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Main driver
# ══════════════════════════════════════════════════════════════════════════════

def main():
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
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
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
    for d in [OUT_DIR, CKPT_ROOT, MODELS_DIR, FIGS_DIR]:
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
    neg_paths = prepare_hard_negatives(NUM_NEGATIVES, skip=args.skip_negatives)

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
        star = " ★" if r["is_baseline"] else ""
        print(f"  {cid:<30}  {r['best_map50']:>8.4f}  "
              f"{r['fps']:>7.1f}  {r['n_params']/1e6:>9.1f}M{star}")
    print("═" * 66)
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Results JSON    : {RESULTS_PATH}")
    print(f"  Figures         : {FIGS_DIR}")


if __name__ == "__main__":
    main()
