#!/usr/bin/env python3
"""
train_swin.py — Swin crop-disease detector (Swin-V2-T backbone + FPN + Faster R-CNN head).

A hierarchical Vision Transformer detector for the two-stage pipeline and the mobile
app. The backbone is a torchvision Swin-V2-T (ImageNet-pretrained) that produces a
genuine multi-scale feature pyramid (/4, /8, /16, /32), wrapped in an FPN and fed to
the *same* Faster R-CNN detection head (RPN + RoI) used by the ResNet and ViT pipelines.
Compared with the single-scale ViTDet, the hierarchical pyramid usually helps on small
disease lesions — a natural "plain vs hierarchical transformer" comparison for the paper.

Design notes
------------
  • Swin uses windowed attention with relative position bias, so — unlike ViT — it
    handles variable input sizes and needs no fixed-square resize.
  • Four stage outputs (channels 96/192/384/768, channels-last) are permuted to
    NCHW and fed to a FeaturePyramidNetwork (out=256) with a LastLevelMaxPool, giving
    the 5-level pyramid the default Faster R-CNN anchor generator expects.
  • AdamW + low LR (transformer fine-tuning), linear warmup → cosine decay, gradient
    accumulation for a larger effective batch, backbone frozen for the first epochs.

This file is deliberately SELF-CONTAINED: the dataset, hard-negative download,
evaluation, scheduler, checkpointing, figures and export are duplicated here rather
than imported from another model package, so the packages stay fully decoupled.

Usage
-----
  python -m src.swin.train_swin                    # full pipeline (steps 1–4)
  python -m src.swin.train_swin --dry-run          # 2-epoch timing estimate
  python -m src.swin.train_swin --skip-negatives   # skip hard-negative download
  python -m src.swin.train_swin --figures-only      # regenerate figures only
  python -m src.swin.train_swin --export-only       # re-export best checkpoint
  python -m src.swin.train_swin --no-figures        # train without figures
  python -m src.swin.train_swin --epochs 20         # override epoch count
  python -m src.swin.train_swin --batch-size 4      # override batch size (e.g. on CUDA)
  DRY_RUN=1 python -m src.swin.train_swin           # dry-run via env var
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models import Swin_V2_T_Weights, swin_v2_t
from torchvision.models.detection import FasterRCNN
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.transforms import v2

from src.swin.config import (
    PROJECT_ROOT, DATASET_DIR, NEG_DIR,
    OUTPUT_DIR, CKPT_DIR, MODELS_DIR, METRICS_FILE, FINAL_EVAL_FILE,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
    NUM_CLASSES, IMG_SIZE,
    SWIN_IN_CHANNELS, FPN_OUT_CHANNELS,
    EPOCHS_DEFAULT, PATIENCE_DEFAULT as PATIENCE, BATCH_SIZE, ACCUM_STEPS,
    LR0, WEIGHT_DECAY, WARMUP_EPOCHS, FREEZE_BACKBONE_EPOCHS, GRAD_CLIP,
    EVAL_EVERY, NUM_NEGATIVES, CONF_THRESHOLD, IOU_THRESHOLD,
    CLASS_NAMES, CLASS_NAMES_DISPLAY,
)

BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Hard negatives (OOD guard)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_hard_negatives(num: int = NUM_NEGATIVES, skip: bool = False) -> list:
    """Download diverse non-crop images (empty annotations) to suppress OOD detections.

    Fully resumable: already-downloaded files are reused, and a fixed seed picks the
    same image ids each run. Shares the data/negatives/ cache with the other pipelines.
    """
    neg_img_dir = NEG_DIR / "images"
    if skip:
        print("  Hard-negative preparation skipped (--skip-negatives).")
        return sorted(neg_img_dir.glob("*.jpg"))[:num] if neg_img_dir.exists() else []

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
                    err += 1
                else:
                    ok += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} fetched  (ok={ok}, errors={err})")
        print(f"  Download complete: {ok} new, {err} failed")

    all_negs = sorted(neg_img_dir.glob("*.jpg"))[:num]
    print(f"  Hard negatives ready: {len(all_negs)} images  ({neg_img_dir})")
    return all_negs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df["x1"] < df["x2"]) & (df["y1"] < df["y2"])].copy()
    df["img_id"] = df["fname"].apply(lambda x: x.rsplit(".", 1)[0])
    return df


class CropDiseaseDataset(Dataset):
    """CSV-based detection dataset. Positives grouped by img_id; negatives have empty targets.

    integer_label values are kept 1-indexed (label 0 = background), matching label_map.json.
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None,
                 neg_paths: Optional[list] = None):
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

    def _get_positive(self, idx: int):
        img_id  = self.image_ids[idx]
        records = self.df[self.df["img_id"] == img_id]

        img_path = self.image_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_t = v2.functional.to_image(img)
        h, w  = img_t.shape[-2], img_t.shape[-1]

        boxes  = records[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        labels = records["integer_label"].values.astype(np.int64)

        boxes_tv = tv_tensors.BoundingBoxes(torch.as_tensor(boxes), format="XYXY",
                                            canvas_size=(h, w))
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

        boxes_tv = tv_tensors.BoundingBoxes(torch.zeros((0, 4), dtype=torch.float32),
                                            format="XYXY", canvas_size=(h, w))
        if self.transform:
            img_t, boxes_tv = self.transform(img_t, boxes_tv)
        else:
            img_t = v2.functional.to_dtype(img_t, torch.float32, scale=True)

        empty_boxes  = torch.zeros((0, 4), dtype=torch.float32)
        empty_labels = torch.zeros((0,),   dtype=torch.int64)
        return img_t, _make_target(empty_boxes, empty_labels, self._n_pos + neg_idx)


def _sanitise_boxes(boxes: torch.Tensor, labels: torch.Tensor, h: int, w: int) -> tuple:
    if boxes.numel() == 0:
        return boxes, labels
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], labels[keep]


def _make_target(boxes: torch.Tensor, labels: torch.Tensor, idx: int) -> dict:
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]),
            "area": area, "iscrowd": torch.zeros(labels.shape[0], dtype=torch.int64)}


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
# Model — Swin-V2-T backbone + FPN + Faster R-CNN head
# ══════════════════════════════════════════════════════════════════════════════

class SwinFPNBackbone(nn.Module):
    """Swin-V2-T → 4 multi-scale feature maps → FeaturePyramidNetwork (out_channels)."""

    # capture the output after these indices of swin.features (the 4 stage outputs at
    # strides /4, /8, /16, /32 with channels 96, 192, 384, 768)
    _STAGE_IDX = (1, 3, 5, 7)

    def __init__(self, out_channels: int = FPN_OUT_CHANNELS, pretrained: bool = True):
        super().__init__()
        weights = Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            swin = swin_v2_t(weights=weights)
        except Exception as exc:
            print(f"  [WARN]  Could not load pretrained Swin weights ({exc}); using random init.")
            swin = swin_v2_t(weights=None)
        self.features = swin.features
        self.out_channels = out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(SWIN_IN_CHANNELS),
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        key = 0
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self._STAGE_IDX:
                feats[str(key)] = h.permute(0, 3, 1, 2).contiguous()  # BHWC → BCHW
                key += 1
        return self.fpn(feats)


def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE,
                pretrained: bool = True) -> FasterRCNN:
    backbone = SwinFPNBackbone(FPN_OUT_CHANNELS, pretrained)
    # FPN yields a 5-level pyramid (keys 0-3 + LastLevelMaxPool "pool"); the default
    # Faster R-CNN anchor generator is built automatically for those 5 levels.
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone, num_classes=num_classes,
        box_roi_pool=roi_pooler,
        min_size=img_size, max_size=img_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model


def set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
    """Toggle grads on the Swin backbone (the FPN stays trainable)."""
    for p in model.backbone.features.parameters():
        p.requires_grad = requires_grad


# ══════════════════════════════════════════════════════════════════════════════
# Device / schedule
# ══════════════════════════════════════════════════════════════════════════════

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


def log_startup(device, n_train, n_val, epochs, batch, dry_run):
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  Model        : Swin (Swin-V2-T backbone + FPN + Faster R-CNN head)")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {batch}  (accum {ACCUM_STEPS} → effective {batch * ACCUM_STEPS})")
    print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}  (multi-scale FPN: /4, /8, /16, /32)")
    print(f"  Train images : {n_train:,}  (incl. {NUM_NEGATIVES} hard-negatives)")
    print(f"  Val images   : {n_val:,}")
    print(f"  Epochs       : {epochs}" + ("  ← DRY RUN" if dry_run else ""))
    print(f"  Optimizer    : AdamW  lr={LR0}  (cosine, warmup {WARMUP_EPOCHS} ep)")
    print(f"  Backbone     : frozen for first {FREEZE_BACKBONE_EPOCHS} epochs")
    print(f"  num_classes  : {NUM_CLASSES}  (23 diseases + background)")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(epoch, model, optimizer, scheduler, best_map, metrics_history, is_best):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map": best_map,
        "metrics_history": metrics_history,
    }
    last_path = CKPT_DIR / "last.pth"
    torch.save(state, last_path)
    if is_best:
        shutil.copy2(last_path, CKPT_DIR / "best.pth")
    if epoch % 10 == 0:
        shutil.copy2(last_path, CKPT_DIR / f"epoch_{epoch:04d}.pth")


def _is_resumable(ckpt_path: Path) -> bool:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return isinstance(ckpt, dict) and all(
            k in ckpt for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict"))
    except Exception:
        return False


def load_checkpoint(ckpt_path: Path, model, optimizer, scheduler):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_map", 0.0), ckpt.get("metrics_history", {})


# ══════════════════════════════════════════════════════════════════════════════
# Train / evaluate
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, optimizer, loader, device, epoch, scaler=None,
                    accum_steps: int = ACCUM_STEPS) -> dict:
    model.train()
    totals = {"total": 0.0, "classifier": 0.0, "box_reg": 0.0,
              "objectness": 0.0, "rpn_box_reg": 0.0}
    n = len(loader)
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if scaler is not None else contextlib.nullcontext())

    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(loader):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with autocast_ctx:
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())
        scaled = losses / accum_steps

        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (batch_idx + 1) % accum_steps == 0 or batch_idx == n - 1:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad()

        totals["total"] += losses.item()
        for key, val in loss_dict.items():
            short = key.replace("loss_", "")
            if short in totals:
                totals[short] += val.item()

        if (batch_idx + 1) % max(1, n // 5) == 0 or batch_idx == n - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch:3d}  [{(batch_idx + 1) / n * 100:5.1f}%]  "
                  f"loss={totals['total'] / (batch_idx + 1):.4f}  lr={lr:.2e}")

    return {k: v / n for k, v in totals.items()}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 23) -> dict:
    """VOC-style mAP@0.5 + per-class AP (11-point interpolation). Pure NumPy."""
    model.eval()
    class_dets: dict = defaultdict(list)
    class_ngt:  dict = defaultdict(int)

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

    aps = {}
    for c in range(1, num_classes + 1):
        ngt = class_ngt[c]
        if ngt == 0:
            aps[c] = float("nan"); continue
        dets = sorted(class_dets.get(c, []), key=lambda x: -x[0])
        if not dets:
            aps[c] = 0.0; continue
        tp = np.array([d[1] for d in dets], dtype=np.float32)
        fp = 1 - tp
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec  = tp_c / ngt
        prec = tp_c / (tp_c + fp_c)
        ap = 0.0
        for thresh in np.linspace(0, 1, 11):
            p = prec[rec >= thresh]
            ap += float(np.max(p)) if len(p) > 0 else 0.0
        aps[c] = ap / 11.0

    valid = [v for v in aps.values() if not math.isnan(v)]
    return {"map50": float(np.mean(valid)) if valid else 0.0, "per_class_ap": aps}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics persistence
# ══════════════════════════════════════════════════════════════════════════════

def _empty_history() -> dict:
    return {"epoch": [], "train_total": [], "train_cls": [], "train_box_reg": [],
            "train_obj": [], "train_rpn": [], "val_map50": [], "lr": []}


def _load_metrics() -> dict:
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return _empty_history()


def _save_metrics(history: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_final_eval(result: dict, model: nn.Module, split: str = "test") -> None:
    """Persist final per-class AP + mAP as JSON so the paper's cross-model aggregator can read it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_class = {CLASS_NAMES[c]: (None if math.isnan(v) else round(float(v), 5))
                 for c, v in result["per_class_ap"].items()}
    payload = {
        "model_name": "swin_v2_t_fpn",
        "architecture": "Swin-V2-T backbone + FPN + Faster R-CNN head",
        "split": split,
        "map50": round(float(result["map50"]), 5),
        "num_params": _count_params(model),
        "per_class_ap": per_class,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(FINAL_EVAL_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Final eval saved → {FINAL_EVAL_FILE}")

    # Also drop a copy into the shared benchmark folder for cross-model comparison.
    try:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_DIR / "swin_v2_t_fpn.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"  [WARN]  Could not write shared benchmark summary: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Publication figures
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_df(split: str) -> pd.DataFrame:
    path_map = {"train": TRAIN_CSV, "valid": VAL_CSV, "test": TEST_CSV}
    df = pd.read_csv(path_map[split])
    df = df[(df["x1"] < df["x2"]) & (df["y1"] < df["y2"])].copy()
    df["split"]      = split
    df["img_id"]     = df["fname"].apply(lambda x: x.rsplit(".", 1)[0])
    df["class_name"] = df["class"]
    df["crop"]       = df["class"].apply(lambda x: x.split()[0])
    return df


def generate_figures(pre_only: bool = False) -> None:
    """Write publication figures (300 DPI) to OUTPUT_DIR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})

    # ── fig_01 — dataset overview ────────────────────────────────────────────
    try:
        splits = {s: _load_split_df(s) for s in ("train", "valid", "test")}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        counts_imgs = [splits[s]["img_id"].nunique() for s in ("train", "valid", "test")]
        counts_box  = [len(splits[s]) for s in ("train", "valid", "test")]
        x = np.arange(3); labels = ["train", "valid", "test"]
        axes[0].bar(x, counts_imgs, color="#5B8FF9")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
        axes[0].set_title("Images per split"); axes[0].set_ylabel("images")
        for i, v in enumerate(counts_imgs):
            axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        axes[1].bar(x, counts_box, color="#61DDAA")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_title("Annotations per split"); axes[1].set_ylabel("boxes")
        for i, v in enumerate(counts_box):
            axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("Swin — Dataset Overview", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig_01_dataset_overview.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [WARN]  fig_01 skipped: {exc}")

    # ── fig_02 — Swin architecture schematic ─────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
        stages = [
            ("Input\n640×640×3", "#AAB7C4"),
            ("Patch embed\n/4 · 96ch", "#5B8FF9"),
            ("Stage 1–4\nwindow attention\n/4·96 /8·192\n/16·384 /32·768", "#3E6FE0"),
            ("FPN\n(out 256, 5 lvls)", "#61DDAA"),
            ("RPN + RoI\nhead", "#F6BD16"),
            ("Boxes + 23\nclasses", "#E8684A"),
        ]
        w = 1.7; gap = 0.3; x = 0.15
        for i, (txt, col) in enumerate(stages):
            box = FancyBboxPatch((x, 0.8), w, 1.4, boxstyle="round,pad=0.04",
                                 fc=col, ec="black", lw=0.8, alpha=0.9)
            ax.add_patch(box)
            ax.text(x + w / 2, 1.5, txt, ha="center", va="center", fontsize=8, wrap=True)
            if i < len(stages) - 1:
                ax.add_patch(FancyArrowPatch((x + w, 1.5), (x + w + gap, 1.5),
                             arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))
            x += w + gap
        ax.set_title("Swin — Swin-V2-T backbone + FPN feeding a Faster R-CNN detection head",
                     fontweight="bold", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig_02_swin_architecture.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [WARN]  fig_02 skipped: {exc}")

    # ── fig_03 — LR schedule ─────────────────────────────────────────────────
    try:
        total = EPOCHS_DEFAULT
        ep = np.arange(total)
        lr = []
        for e in ep:
            if e < WARMUP_EPOCHS:
                lr.append(LR0 * (e + 1) / max(1, WARMUP_EPOCHS))
            else:
                prog = (e - WARMUP_EPOCHS) / max(1, total - WARMUP_EPOCHS)
                lr.append(LR0 * 0.5 * (1 + math.cos(math.pi * prog)))
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(ep, lr, color="#3E6FE0", lw=2)
        ax.axvspan(0, WARMUP_EPOCHS, color="#F6BD16", alpha=0.15, label="warmup")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
        ax.set_title("Swin — AdamW LR schedule (linear warmup → cosine decay)", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig_03_lr_schedule.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [WARN]  fig_03 skipped: {exc}")

    if pre_only:
        print("  Pre-training figures written (no checkpoint required).")

    # ── fig_08 — training metrics (needs metrics_history.json) ────────────────
    if METRICS_FILE.exists():
        try:
            h = _load_metrics()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            axes[0].plot(h["epoch"], h["train_total"], color="#E8684A", lw=2, label="total")
            for key, col, lbl in [("train_cls", "#5B8FF9", "cls"),
                                  ("train_box_reg", "#61DDAA", "box"),
                                  ("train_obj", "#F6BD16", "obj"),
                                  ("train_rpn", "#9270CA", "rpn")]:
                if h.get(key):
                    axes[0].plot(h["epoch"], h[key], lw=1.2, alpha=0.8, label=lbl)
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].set_title("Training loss"); axes[0].legend(fontsize=8)
            ep = [e for e, m in zip(h["epoch"], h["val_map50"]) if m is not None]
            mp = [m for m in h["val_map50"] if m is not None]
            axes[1].plot(ep, mp, color="#27AE60", lw=2, marker="o", ms=3)
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP@0.5")
            axes[1].set_title("Validation mAP@0.5")
            fig.suptitle("Swin — Training Metrics", fontweight="bold")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "fig_08_training_metrics.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"  [WARN]  fig_08 skipped: {exc}")

    # ── fig_09 — per-class AP (needs final_eval.json) ────────────────────────
    if FINAL_EVAL_FILE.exists():
        try:
            with open(FINAL_EVAL_FILE) as f:
                fe = json.load(f)
            names = list(fe["per_class_ap"].keys())
            vals  = [0.0 if v is None else v for v in fe["per_class_ap"].values()]
            crop_col = {"Corn": "#F6BD16", "Pepper": "#27AE60", "Tomato": "#E8684A"}
            colors = [crop_col.get(n.split()[0], "#5B8FF9") for n in names]
            fig, ax = plt.subplots(figsize=(9, 8))
            yp = np.arange(len(names))
            ax.barh(yp, vals, color=colors)
            ax.set_yticks(yp); ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("AP@0.5")
            ax.set_title(f"Swin — Per-class AP@0.5  (mAP={fe['map50']:.3f})", fontweight="bold")
            for i, v in enumerate(vals):
                ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "fig_09_per_class_ap.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"  [WARN]  fig_09 skipped: {exc}")

    print(f"  Figures written → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Export — TorchScript (.ptl) + ONNX + ExecuTorch (.pte)
# ══════════════════════════════════════════════════════════════════════════════

class _BackboneWrapper(nn.Module):
    """Feature-extractor wrapper: normalized image tensor → the FPN pyramid feature maps.

    Static shapes throughout (Swin windowed-attention + FPN convs), so this exports
    cleanly to ExecuTorch as a multi-output backbone.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        return tuple(feats.values())


def export_model(model: nn.Module) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    # ── 1. TorchScript mobile (.ptl) — most reliable full-model target ────────
    # The two-stage model uses ImageList internally, which the lite interpreter
    # rejects; fall back to a plain TorchScript archive (loadable via LibTorch),
    # exactly as the ResNet Faster R-CNN pipeline does.
    print("  [1/4]  TorchScript mobile …")
    try:
        scripted = torch.jit.script(model)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(scripted)
        ptl_path  = MODELS_DIR / "crop_disease_swin.ptl"
        optimized._save_for_lite_interpreter(str(ptl_path))
        print(f"         [OK] {ptl_path.name}  ({ptl_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"         [WARN]  TorchScript mobile failed: {exc}")
        # Remove any partial lite-interpreter file, then save a plain TorchScript archive.
        (MODELS_DIR / "crop_disease_swin.ptl").unlink(missing_ok=True)
        try:
            scripted = torch.jit.script(model)
            pt_path  = MODELS_DIR / "crop_disease_swin_jit.pt"
            scripted.save(str(pt_path))
            print(f"         [OK] Saved plain TorchScript: {pt_path.name}  "
                  f"({pt_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc2:
            print(f"         [FAIL]  TorchScript also failed: {exc2}")

    # ── 2. ONNX (universal fallback) ─────────────────────────────────────────
    # Wrap so the graph returns plain (boxes, scores, labels) tensors rather than a
    # list-of-dicts. dynamo=False uses the legacy exporter (no onnxscript dependency).
    print("  [2/4]  ONNX …")
    try:
        class _ONNXWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor):
                out = self.m([x[0]])
                return out[0]["boxes"], out[0]["scores"], out[0]["labels"].float()

        onnx_path = MODELS_DIR / "crop_disease_swin.onnx"
        torch.onnx.export(
            _ONNXWrapper(model), example, str(onnx_path),
            opset_version=17, dynamo=False,
            input_names=["images"], output_names=["boxes", "scores", "labels"],
            dynamic_axes={"images": {0: "batch"}, "boxes": {0: "n_det"},
                          "scores": {0: "n_det"}, "labels": {0: "n_det"}},
        )
        print(f"         [OK] {onnx_path.name}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"         [WARN]  ONNX export failed: {exc}")

    # ── 3. ExecuTorch (.pte) — backbone (clean) + full model (attempt) ───────
    print("  [3/4]  ExecuTorch (.pte) …")
    try:
        from executorch.exir import to_edge
        from torch.export import export as torch_export
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        # 3a. Backbone-only — static-shape Swin+FPN, exports reliably.
        backbone_wrapper = _BackboneWrapper(model.backbone).eval()
        exported = torch_export(backbone_wrapper, (example,), strict=False)
        edge = to_edge(exported)
        try:
            edge = edge.to_backend(XnnpackPartitioner())
        except Exception:
            pass
        et = edge.to_executorch()
        pte_bb = MODELS_DIR / "crop_disease_swin_backbone.pte"
        with open(pte_bb, "wb") as f:
            f.write(et.buffer)
        print(f"         [OK] {pte_bb.name}  ({pte_bb.stat().st_size / 1e6:.1f} MB)  [backbone]")

        # 3b. Full detection model — dynamic RPN/RoI control flow; best-effort.
        try:
            exp_full = torch_export(model, ([example[0]],), strict=False)
            edge_full = to_edge(exp_full)
            try:
                edge_full = edge_full.to_backend(XnnpackPartitioner())
            except Exception:
                pass
            et_full = edge_full.to_executorch()
            pte_full = MODELS_DIR / "crop_disease_swin.pte"
            with open(pte_full, "wb") as f:
                f.write(et_full.buffer)
            print(f"         [OK] {pte_full.name}  ({pte_full.stat().st_size / 1e6:.1f} MB)  [full]")
        except Exception as exc:
            print(f"         [INFO]  Full-model .pte not exported ({type(exc).__name__}); "
                  f"use the backbone .pte + .ptl for on-device detection.")
    except Exception as exc:
        print(f"         [WARN]  ExecuTorch export failed: {exc}")

    # ── 4. Metadata YAML ─────────────────────────────────────────────────────
    print("  [4/4]  Metadata …")
    import yaml
    metadata = {
        "model_name": "crop_disease_swin",
        "architecture": "Swin-V2-T backbone + FPN + Faster R-CNN head",
        "task": "object_detection",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "input_size": IMG_SIZE,
        "input_channels": 3,
        "num_classes": len(CLASS_NAMES_DISPLAY),
        "class_names": CLASS_NAMES_DISPLAY,
        "label_offset": 1,   # model labels are 1-indexed; subtract 1 to index class_names
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "crops_covered": ["Corn", "Pepper", "Tomato"],
        "primary_format": ".ptl (TorchScript mobile / LibTorch)",
        "executorch_format": ".pte (ExecuTorch backbone; full model if export succeeds)",
        "notes": ("Labels 1-23 correspond to class_names[0]-class_names[22]. "
                  "Apply conf_threshold at inference before displaying results."),
        "android_integration": {
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "pte_file": "crop_disease_swin.pte",
            "backbone_pte_file": "crop_disease_swin_backbone.pte",
            "input_normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "input_format": "NCHW_RGB",
        },
    }
    meta_path = MODELS_DIR / "model_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"         [OK] {meta_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _make_loaders(neg_paths, batch, workers):
    train_df = _load_csv(TRAIN_CSV)
    val_df   = _load_csv(VAL_CSV)
    train_ds = CropDiseaseDataset(train_df, TRAIN_IMG_DIR, get_train_transform(), neg_paths)
    val_ds   = CropDiseaseDataset(val_df, VAL_IMG_DIR, get_val_transform())
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                          collate_fn=collate_fn, pin_memory=False)
    val_ld   = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers,
                          collate_fn=collate_fn, pin_memory=False)
    return train_ld, val_ld, len(train_ds), len(val_ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Swin crop-disease detector")
    parser.add_argument("--dry-run", action="store_true", help="2-epoch timing estimate")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--skip-negatives", action="store_true", help="skip hard-negative download")
    parser.add_argument("--figures-only", action="store_true", help="regenerate figures only")
    parser.add_argument("--export-only", action="store_true", help="re-export best checkpoint only")
    parser.add_argument("--no-figures", action="store_true", help="train without figures")
    parser.add_argument("--no-pretrained", action="store_true", help="random-init Swin (no ImageNet)")
    args = parser.parse_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    epochs  = args.epochs if args.epochs is not None else (2 if dry_run else EPOCHS_DEFAULT)
    batch   = args.batch_size if args.batch_size is not None else BATCH_SIZE

    device  = resolve_device()
    workers = 0 if device.type == "mps" else min(8, os.cpu_count() or 1)

    # ── figures-only ─────────────────────────────────────────────────────────
    if args.figures_only:
        print("Regenerating Swin figures …")
        generate_figures(pre_only=not FINAL_EVAL_FILE.exists())
        return

    # ── export-only ──────────────────────────────────────────────────────────
    if args.export_only:
        best = CKPT_DIR / "best.pth"
        if not best.exists():
            print(f"[ERROR] No checkpoint at {best}. Train first."); return
        model = build_model(pretrained=False)
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Exporting best checkpoint …")
        export_model(model)
        return

    # ── Step 1 — hard negatives ──────────────────────────────────────────────
    print("Step 1 — hard negatives")
    neg_paths = prepare_hard_negatives(NUM_NEGATIVES, skip=args.skip_negatives)

    # ── Step 2 — train ───────────────────────────────────────────────────────
    train_ld, val_ld, n_train, n_val = _make_loaders(neg_paths, batch, workers)
    log_startup(device, n_train, n_val, epochs, batch, dry_run)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch, best_map, history = 1, 0.0, _load_metrics()
    last_ckpt = CKPT_DIR / "last.pth"
    if last_ckpt.exists() and _is_resumable(last_ckpt):
        start_epoch, best_map, history = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        start_epoch += 1
        print(f"  Resumed from epoch {start_epoch - 1}  (best mAP={best_map:.4f})")
    if not history:
        history = _load_metrics()

    backbone_frozen = None
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        # freeze/unfreeze the Swin backbone
        want_frozen = epoch <= FREEZE_BACKBONE_EPOCHS
        if backbone_frozen != want_frozen:
            set_backbone_grad(model, requires_grad=not want_frozen)
            backbone_frozen = want_frozen
            print(f"  Backbone {'frozen' if want_frozen else 'unfrozen'} (epoch {epoch})")

        stats = train_one_epoch(model, optimizer, train_ld, device, epoch, scaler)
        scheduler.step()

        val_map = None
        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            print(f"  Evaluating val mAP@0.5 (epoch {epoch}) …")
            val_map = evaluate(model, val_ld, device, num_classes=23)["map50"]
            print(f"  val mAP@0.5 = {val_map:.4f}")

        history["epoch"].append(epoch)
        history["train_total"].append(round(stats["total"], 5))
        history["train_cls"].append(round(stats["classifier"], 5))
        history["train_box_reg"].append(round(stats["box_reg"], 5))
        history["train_obj"].append(round(stats["objectness"], 5))
        history["train_rpn"].append(round(stats["rpn_box_reg"], 5))
        history["val_map50"].append(None if val_map is None else round(val_map, 5))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        _save_metrics(history)

        is_best = val_map is not None and val_map > best_map
        if is_best:
            best_map = val_map
            epochs_no_improve = 0
            print(f"  [*]  New best mAP@0.5: {best_map:.4f}")
        elif val_map is not None:
            epochs_no_improve += 1

        save_checkpoint(epoch, model, optimizer, scheduler, best_map, history, is_best)
        print(f"  epoch {epoch}/{epochs}  loss={stats['total']:.4f}  best_mAP={best_map:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping (no improvement for {PATIENCE} evals).")
            break

    print(f"\n  Training complete in {(time.time() - t0) / 60:.1f} min  |  best mAP@0.5 = {best_map:.4f}")
    if dry_run:
        print("  DRY RUN complete — skipping final eval / figures / export.")
        return

    # ── final test-set eval ──────────────────────────────────────────────────
    best = CKPT_DIR / "best.pth"
    if best.exists():
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_df = _load_csv(TEST_CSV)
    test_ds = CropDiseaseDataset(test_df, TEST_IMG_DIR, get_val_transform())
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers,
                         collate_fn=collate_fn)
    print("\n  Final evaluation on test split …")
    final_result = evaluate(model, test_ld, device, num_classes=23)
    print(f"  Final mAP@0.5 = {final_result['map50']:.4f}")
    save_final_eval(final_result, model, split="test")

    # ── Step 3 — figures ─────────────────────────────────────────────────────
    if not args.no_figures:
        print("\nStep 3 — figures")
        generate_figures()

    # ── Step 4 — export ──────────────────────────────────────────────────────
    print("\nStep 4 — export")
    export_model(model)
    print("\n[OK]  Swin pipeline complete.")


if __name__ == "__main__":
    main()
