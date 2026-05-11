#!/usr/bin/env python3
"""
train_classifier.py — Crop-Type Leaf Classifier (Stage 1 of 2-stage pipeline)

PURPOSE
-------
Trains a lightweight EfficientNet-B2 classifier that identifies whether a
leaf image belongs to Corn, Pepper, or Tomato — before the YOLO disease
detector runs.  This prevents cross-crop false positives (e.g., a mango
leaf being labelled as Tomato).

PIPELINE
--------
  Input Image
       │
       ▼
  ┌─────────────────────────────────┐
  │  Stage 1: Crop Classifier       │  ← this model
  │  EfficientNet-B2 (3 classes)    │
  └─────────────────────────────────┘
       │  Corn / Pepper / Tomato / Unknown
       ▼
  ┌─────────────────────────────────┐
  │  Stage 2: Disease Detector      │
  │  YOLO (filtered to crop classes)│
  └─────────────────────────────────┘
       │
       ▼
  Disease bounding boxes

PREREQUISITES
-------------
  Run generate_classifier_csv.py first to produce:
    dataset/classifier_train.csv
    dataset/classifier_valid.csv
    dataset/classifier_test.csv

OUTPUTS  (→ outputs/classifier_output/)
--------
  best.pth               — best validation-accuracy checkpoint
  last.pth               — final-epoch checkpoint
  metrics_history.json   — per-epoch loss / accuracy
  figures/
    fig_01_training_curves.png
    fig_02_confusion_matrix.png
    fig_03_per_class_accuracy.png
    fig_04_sample_predictions.png

USAGE
-----
  python train_classifier.py                    # full training run
  python train_classifier.py --dry-run          # 2 epochs to check setup
  python train_classifier.py --epochs 30        # override epoch count
  python train_classifier.py --batch-size 64
  python train_classifier.py --figures-only     # regenerate figures from best.pth
  python train_classifier.py --no-figures
  python train_classifier.py --confidence 0.60  # unknown-crop threshold
"""

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_DIR   = PROJECT_ROOT / "dataset"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "classifier_output"
FIGURES_DIR   = OUTPUT_DIR / "figures"

CROP_CLASSES  = ["Corn", "Pepper", "Tomato"]
NUM_CLASSES   = len(CROP_CLASSES)       # 3

IMG_SIZE      = 260                     # EfficientNet-B2 native resolution
EPOCHS_DEFAULT = 40
BATCH_DEFAULT  = 64                     # M4 Pro 24 GB handles batch-64 with headroom
LR_DEFAULT     = 1e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 8                      # early-stopping patience (epochs)
CONF_DEFAULT   = 0.55                   # min softmax confidence to accept a crop
GRAD_CLIP      = 1.0                    # max gradient norm for training stability

# Corn classes 0–4, Pepper 5–14, Tomato 15–22 (for pipeline integration)
CROP_TO_YOLO_CLASSES: dict[str, list[int]] = {
    "Corn":   list(range(0, 5)),
    "Pepper": list(range(5, 15)),
    "Tomato": list(range(15, 23)),
}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _build_transforms(split: str) -> transforms.Compose:
    """Return augmentation pipeline for train or eval splits."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


class CropDataset(Dataset):
    """Image classification dataset backed by a CSV produced by
    generate_classifier_csv.py."""

    def __init__(self, csv_path: Path, split: str) -> None:
        df = pd.read_csv(csv_path)
        self.records  = df[["image_path", "crop_id"]].values.tolist()
        self.transform = _build_transforms(split)
        self.root      = PROJECT_ROOT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rel_path, label = self.records[idx]
        img_path = self.root / rel_path
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), int(label)


def _make_loaders(
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CropDataset(DATASET_DIR / "classifier_train.csv", "train")
    valid_ds = CropDataset(DATASET_DIR / "classifier_valid.csv", "valid")
    test_ds  = CropDataset(DATASET_DIR / "classifier_test.csv",  "test")

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Keep workers alive between epochs — eliminates per-epoch spawn cost on macOS
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"Dataset  train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}")
    return train_loader, valid_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """EfficientNet-B2 with ImageNet weights; classifier head replaced."""
    weights = EfficientNet_B2_Weights.DEFAULT
    model   = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    scaler,                      # GradScaler for CUDA; None for MPS / CPU
    use_autocast: bool = False,  # float16 autocast for CUDA and MPS
) -> tuple[float, float]:
    """One forward (+ optional backward) pass.  Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0

    # nullcontext is used when autocast is disabled (CPU), so no branching in loop
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_autocast
        else contextlib.nullcontext()
    )

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(images)
                loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

            preds        = logits.argmax(dim=1)
            total_loss  += loss.item() * labels.size(0)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Return (true_labels, pred_labels, softmax_probs) for the whole split."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().tolist()
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_true.extend(labels.tolist())
        all_pred.extend(preds)
        all_probs.extend(probs)
    return all_true, all_pred, all_probs


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(PROJECT_ROOT)}")


def fig_training_curves(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["valid_loss"], label="Valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["valid_acc"]], label="Valid")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Crop Classifier — Training Curves", fontsize=13)
    _save(fig, "fig_01_training_curves.png")


def fig_confusion_matrix(true_labels: list[int], pred_labels: list[int]) -> None:
    cm   = confusion_matrix(true_labels, pred_labels)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_n],
        ["Counts", "Normalised"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CROP_CLASSES)
        ax.set_yticklabels(CROP_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = data[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color="white" if val > data.max() * 0.6 else "black",
                        fontsize=10)
        fig.colorbar(im, ax=ax)

    fig.suptitle("Crop Classifier — Confusion Matrix (Test Set)", fontsize=13)
    _save(fig, "fig_02_confusion_matrix.png")


def fig_per_class_accuracy(true_labels: list[int], pred_labels: list[int]) -> None:
    cm  = confusion_matrix(true_labels, pred_labels)
    acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(CROP_CLASSES, acc * 100, color=colors, edgecolor="white", width=0.5)
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{a*100:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy on Test Set")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_03_per_class_accuracy.png")


def fig_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n: int = 12,
) -> None:
    """Show a grid of sample images with true/predicted labels."""
    model.eval()
    inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inv_std  = [1 / 0.229,       1 / 0.224,       1 / 0.225]
    inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)

    collected = []
    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            probs = torch.softmax(model(images_dev), dim=1).cpu()
            preds = probs.argmax(dim=1)
            for img, lbl, pred, prob in zip(images, labels, preds, probs):
                collected.append((img, lbl.item(), pred.item(), prob.max().item()))
            if len(collected) >= n:
                break

    random.shuffle(collected)
    samples = collected[:n]
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, (img, true, pred, conf) in zip(axes, samples):
        img_np = inv_norm(img).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img_np)
        color = "green" if true == pred else "red"
        ax.set_title(
            f"True: {CROP_CLASSES[true]}\nPred: {CROP_CLASSES[pred]} ({conf:.0%})",
            fontsize=8, color=color,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_04_sample_predictions.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main training routine
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        n_gpus   = torch.cuda.device_count()
        print(f"Device: CUDA  ({gpu_name}, {vram_gb:.1f} GB VRAM"
              + (f", {n_gpus}× GPUs)" if n_gpus > 1 else ")"))
        # Let cuDNN auto-tune for the fastest conv algorithm (fixed input size)
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS  (Apple Silicon — unified memory)")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU  ({os.cpu_count()} logical cores)")

    # AMP strategy:
    #   CUDA → float16 autocast + GradScaler (loss scaling prevents underflow)
    #   MPS  → float16 autocast only  (MPS ops are numerically stable; no scaler)
    #   CPU  → plain float32  (autocast has no benefit on CPU)
    use_autocast = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Data ──────────────────────────────────────────────────────────────────
    # pin_memory only helps CUDA (DMA transfer to device memory).
    # MPS uses unified memory — the CPU and GPU share the same physical RAM,
    # so pinning adds overhead rather than removing it.
    pin_memory  = device.type == "cuda"
    # On macOS, multiprocessing uses 'spawn' (not 'fork'), so workers are slower
    # to start.  persistent_workers=True (set inside _make_loaders) means workers
    # are initialised once and reused across all epochs.
    # GPU servers typically have 16-64 CPU cores — use more of them for data loading.
    num_workers = min(8, os.cpu_count() or 1) if device.type == "cuda" else min(4, os.cpu_count() or 1)
    print(f"DataLoader workers: {num_workers}  pin_memory: {pin_memory}")
    train_loader, valid_loader, test_loader = _make_loaders(
        args.batch_size, num_workers, pin_memory
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    # Wrap with DataParallel when multiple GPUs are available.
    # Checkpoints always save the unwrapped weights (model.module.state_dict)
    # so they load cleanly on any single-GPU or MPS machine.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Model: EfficientNet-B2  →  {NUM_CLASSES} classes  [{torch.cuda.device_count()}× DataParallel]")
    else:
        print(f"Model: EfficientNet-B2  →  {NUM_CLASSES} classes")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=WEIGHT_DECAY)

    # Cosine annealing with linear warm-up (5 epochs)
    warmup_epochs  = 5
    total_epochs   = args.epochs
    warmup_sched   = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "valid_loss": [], "valid_acc": [],
    }
    best_valid_acc = 0.0
    epochs_no_improve = 0

    print(f"\nTraining for up to {total_epochs} epochs …\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("─" * 60)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_autocast
        )
        valid_loss, valid_acc = run_epoch(
            model, valid_loader, criterion, None, device, None, use_autocast
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{valid_loss:>8.4f}  {valid_acc*100:>6.2f}%  {lr_now:>8.2e}"
              f"  [{elapsed:.0f}s]")

        # Unwrap DataParallel before saving so the state dict has no "module." prefix
        raw = model.module if isinstance(model, nn.DataParallel) else model
        cpu_state = {k: v.cpu() for k, v in raw.state_dict().items()}

        # Save last checkpoint every epoch
        torch.save({"epoch": epoch, "model": cpu_state,
                    "optimizer": optimizer.state_dict(),
                    "valid_acc": valid_acc},
                   OUTPUT_DIR / "last.pth")

        # Save best checkpoint
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model": cpu_state,
                        "valid_acc": valid_acc},
                       OUTPUT_DIR / "best.pth")
            print(f"        ✓ new best valid acc: {best_valid_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"({PATIENCE} epochs without improvement).")
                break

    # Save metrics history
    with (OUTPUT_DIR / "metrics_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved → {OUTPUT_DIR / 'metrics_history.json'}")

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    ckpt = torch.load(OUTPUT_DIR / "best.pth", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model"])

    true_labels, pred_labels, probs = collect_predictions(
        model, test_loader, device
    )
    test_acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%  (best epoch: {ckpt['epoch']})\n")
    print(classification_report(true_labels, pred_labels, target_names=CROP_CLASSES))

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.no_figures:
        print("Generating figures …")
        fig_training_curves(history)
        fig_confusion_matrix(true_labels, pred_labels)
        fig_per_class_accuracy(true_labels, pred_labels)
        fig_sample_predictions(model, test_loader, device)
        print("Done.\n")


def figures_only(args: argparse.Namespace) -> None:
    """Regenerate all figures from an existing best.pth checkpoint."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    ckpt   = torch.load(OUTPUT_DIR / "best.pth", map_location=device,
                        weights_only=True)
    model  = build_model().to(device)
    model.load_state_dict(ckpt["model"])

    with (OUTPUT_DIR / "metrics_history.json").open() as f:
        history = json.load(f)

    num_workers  = min(4, os.cpu_count() or 1)
    _, _, test_loader = _make_loaders(args.batch_size, num_workers, pin_memory=False)

    true_labels, pred_labels, _ = collect_predictions(model, test_loader, device)
    fig_training_curves(history)
    fig_confusion_matrix(true_labels, pred_labels)
    fig_per_class_accuracy(true_labels, pred_labels)
    fig_sample_predictions(model, test_loader, device)
    print("Figures regenerated.")


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper  (importable for use in app_gradio.py / app_streamlit.py)
# ══════════════════════════════════════════════════════════════════════════════

class CropClassifier:
    """
    Thin wrapper around the trained classifier for use in the inference pipeline.

    Usage
    -----
    >>> clf = CropClassifier()                     # loads best.pth automatically
    >>> crop, conf, yolo_classes = clf.predict("path/to/leaf.jpg")
    >>> print(crop, conf, yolo_classes)
    'Tomato'  0.97  [15, 16, 17, 18, 19, 20, 21, 22]
    >>> crop, conf, _ = clf.predict("path/to/mango.jpg")
    >>> print(crop)
    'unknown'  (conf < threshold → reject before YOLO)
    """

    _eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        confidence_threshold: float = CONF_DEFAULT,
        device: torch.device | None = None,
    ) -> None:
        if checkpoint is None:
            checkpoint = OUTPUT_DIR / "best.pth"
        checkpoint = Path(checkpoint)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device    = device
        self.threshold = confidence_threshold
        self.model     = build_model().to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, image: Path | str | Image.Image
    ) -> tuple[str, float, list[int]]:
        """
        Classify a leaf image.

        Returns
        -------
        crop_label : str
            "Corn", "Pepper", "Tomato", or "unknown"
        confidence : float
            Softmax probability of the top class.
        yolo_class_ids : list[int]
            YOLO class IDs to use for this crop (empty list if unknown).
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        tensor = self._eval_transform(image).unsqueeze(0).to(self.device)
        probs  = torch.softmax(self.model(tensor), dim=1)[0]
        conf, idx = probs.max(0)
        conf  = conf.item()
        label = CROP_CLASSES[idx.item()]

        if conf < self.threshold:
            return "unknown", conf, []

        return label, conf, CROP_TO_YOLO_CLASSES[label]


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B2 crop-type leaf classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_DEFAULT)
    parser.add_argument("--lr",          type=float, default=LR_DEFAULT)
    parser.add_argument("--confidence",  type=float, default=CONF_DEFAULT,
                        help="Min softmax confidence to accept a crop prediction")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run 2 epochs to verify the setup")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip training; regenerate figures from best.pth")
    parser.add_argument("--no-figures",  action="store_true",
                        help="Train without generating figures")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 2
        print("[DRY-RUN] Running 2 epochs only.")

    # Validate CSVs exist
    for split in ("train", "valid", "test"):
        csv = DATASET_DIR / f"classifier_{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} not found. Run  python generate_classifier_csv.py  first."
            )

    if args.figures_only:
        figures_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
