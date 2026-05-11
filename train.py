#!/usr/bin/env python3
"""
train.py — Complete YOLO26 crop-disease training pipeline.

Steps performed in order:
  1.  Write data_fixed.yaml     — absolute paths, Kaggle-safe, idempotent
  2.  Stage hard negatives      — downloads 300 diverse non-crop images,
                                  copies them into the training split with
                                  empty label files so YOLO learns to detect
                                  nothing on unrelated imagery (OOD guard)
  3.  Train YOLO26              — resume-aware, multi-GPU, dry-run capable
  4.  Generate figures          — 9 publication-quality PNGs → outputs/yolo_output/
                                  (Fig 10 training metrics added if results.csv exists)

Usage
-----
  python train.py                     # full pipeline (steps 1–4)
  python train.py --dry-run           # 1-epoch validation + epoch-time estimate
  python train.py --skip-negatives    # skip download  (negatives already staged)
  python train.py --figures-only      # regenerate figures only, no training
  python train.py --no-figures        # train without generating figures
  DRY_RUN=1 python train.py           # dry-run via environment variable
"""

import argparse
import os
import random
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Must be set before torch is imported so MPS ops without native kernels fall
# back to CPU silently instead of raising NotImplementedError at runtime.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "main"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"
FIXED_YAML   = PROJECT_ROOT / "data_fixed.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "yolo_output"
EXP_NAME     = "crop_disease_yolo26"

MODEL_SIZE     = "yolo26n"   # switch to yolo26s/m/l/x if val mAP plateaus
IMG_SIZE       = 640
BASE_BATCH     = 32          # per-GPU; 32 × yolo26n fits comfortably in 24 GB
EPOCHS_DEFAULT = 200
PATIENCE       = 25
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45
NUM_NEGATIVES  = 300         # hard-negative images to download and stage

CLASS_NAMES = [
    "Corn_Cercospora_Leaf_Spot", "Corn_Common_Rust", "Corn_Healthy",
    "Corn_Northern_Leaf_Blight", "Corn_Streak",
    "Pepper_Bacterial_Spot", "Pepper_Cercospora", "Pepper_Early_Blight",
    "Pepper_Fusarium", "Pepper_Healthy", "Pepper_Late_Blight",
    "Pepper_Leaf_Blight", "Pepper_Leaf_Curl", "Pepper_Leaf_Mosaic",
    "Pepper_Septoria",
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Fusarium",
    "Tomato_Healthy", "Tomato_Late_Blight", "Tomato_Leaf_Curl",
    "Tomato_Mosaic", "Tomato_Septoria",
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — data_fixed.yaml
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data_yaml() -> None:
    """Write data_fixed.yaml with absolute paths. Safe to call repeatedly."""
    cfg = {
        "path" : str(DATA_DIR),
        "train": str(DATA_DIR / "train" / "images"),
        "val"  : str(DATA_DIR / "valid" / "images"),
        "test" : str(DATA_DIR / "test"  / "images"),
        "nc"   : len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(FIXED_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  data_fixed.yaml → {FIXED_YAML}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Hard negatives
# ══════════════════════════════════════════════════════════════════════════════

def prepare_hard_negatives(num: int = NUM_NEGATIVES, skip: bool = False) -> None:
    """
    Download diverse non-crop images and stage them in the training split.

    Each image gets an *empty* label file.  YOLO interprets empty labels as
    background-only images and learns to produce zero detections on them —
    the primary mechanism that prevents false positives on unrelated imagery.

    A fixed random seed ensures the same image IDs are chosen every run;
    already-downloaded files are skipped, making this fully resumable.
    """
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
                seed, exc = future.result()
                if exc:
                    print(f"    ⚠  seed {seed}: {exc}")
                    err += 1
                else:
                    ok += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} fetched  (ok={ok}, errors={err})")
        print(f"  Download complete: {ok} new, {err} failed")

    # Stage into training split ------------------------------------------------
    train_img_dir = DATA_DIR / "train" / "images"
    train_lbl_dir = DATA_DIR / "train" / "labels"
    existing = list(neg_img_dir.glob("*.jpg"))
    copied = 0
    for img in existing:
        dst = train_img_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
            (train_lbl_dir / img.with_suffix(".txt").name).touch()
            copied += 1

    staged_total = len(list(train_img_dir.glob("negative_*.jpg")))
    print(f"  Training split: {staged_total} negatives staged ({copied} newly added)")
    print("  ✅  Hard-negative setup complete")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_resumable(pt_path: Path) -> bool:
    """Return True only if the checkpoint contains epoch + optimizer state.

    A bare model export (ONNX, ExecuTorch, or a weights-only .pt) lacks these
    keys.  Passing resume=True on such a file causes YOLO to silently fall back
    to a fresh run with all *default* hyperparameters (data=coco8.yaml, etc.),
    ignoring everything we configured.
    """
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        return isinstance(ckpt, dict) and "epoch" in ckpt and "optimizer" in ckpt
    except Exception:
        return False


def _patch_tal_for_mps() -> None:
    """Patch the TAL assigner to run on CPU when MPS is the active device.

    Root cause: PyTorch MPS has a bug where boolean-mask indexing on tensors
    produced by .expand() (non-contiguous views) raises AcceleratorError:
        "index N is out of bounds: 0, range 0 to <batch_size>"
    This affects two lines in ultralytics/utils/tal.py regardless of AMP
    setting, so disabling fp16 alone does not prevent the crash.

    The PYTORCH_ENABLE_MPS_FALLBACK env var only routes *unimplemented* MPS
    ops to CPU; it does not catch *buggy* ones, so it cannot help here either.

    Fix: intercept TaskAlignedAssigner._forward, move all six input tensors
    to CPU, run the full assigner there, then move the outputs back to MPS.
    The assigner is a lightweight assignment step — it contributes negligibly
    to per-epoch training time.
    """
    from ultralytics.utils.tal import TaskAlignedAssigner

    _orig = TaskAlignedAssigner._forward

    def _cpu_forward(self, pd_scores, pd_bboxes, anc_points,
                     gt_labels, gt_bboxes, mask_gt):
        dev = pd_scores.device
        result = _orig(
            self,
            pd_scores.cpu(), pd_bboxes.cpu(), anc_points.cpu(),
            gt_labels.cpu(), gt_bboxes.cpu(), mask_gt.cpu(),
        )
        return tuple(t.to(dev) if isinstance(t, torch.Tensor) else t
                     for t in result)

    TaskAlignedAssigner._forward = _cpu_forward
    print("  MPS TAL patch applied (assigner runs on CPU to avoid boolean-index bug)")


def resolve_device() -> tuple:
    """Return (device, total_batch, use_amp) for the current hardware.

    AMP is disabled on MPS: fp16 precision corrupts index tensors inside the
    TAL assigner, causing `torch.AcceleratorError: index N is out of bounds`
    around epoch 5.  CUDA and CPU are unaffected.
    """
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            return list(range(n)), BASE_BATCH * n, True   # multi-GPU DDP
        return 0, BASE_BATCH, True
    if torch.backends.mps.is_available():
        return "mps", BASE_BATCH, False   # AMP off — MPS fp16 index bug
    return "cpu", BASE_BATCH, False


def log_startup(device, batch, n_train, n_val, model_tag, epochs, dry_run, use_amp):
    sep = "─" * 62
    amp_note = "" if use_amp else "  (disabled on MPS — fp16 index bug)"
    print(f"\n{sep}")
    print(f"  Device       : {device}"
          + ("  (multi-GPU DDP)" if isinstance(device, list) else ""))
    print(f"  Model        : {model_tag}.pt")
    print(f"  Batch        : {batch}"
          + ("  (16/GPU)" if isinstance(device, list) else ""))
    print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Train images : {n_train}")
    print(f"  Val images   : {n_val}")
    print(f"  Epochs       : {epochs}" + ("  ← DRY RUN" if dry_run else ""))
    print(f"  Cache        : disk")
    print(f"  AMP          : {use_amp}{amp_note}")
    print(f"{sep}\n")


def log_dry_run_summary(elapsed, full_epochs, save_dir):
    sep = "─" * 62
    mins  = elapsed * full_epochs / 60
    hours = mins / 60
    print(f"\n{sep}")
    print(f"  Dry-run epoch time   : {elapsed:.1f}s")
    print(f"  Estimated full run   : ~{mins:.0f} min  ({hours:.1f} h)  @ {full_epochs} epochs")
    print(f"  Best weights         : {save_dir}/weights/best.pt")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Publication figures
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_df(split: str) -> "pd.DataFrame":
    """Load YOLO label files for one split into a tidy DataFrame."""
    import numpy as np
    import pandas as pd

    ldir = DATA_DIR / split / "labels"
    rows = []
    for lf in sorted(ldir.glob("*.txt")):
        lines = [ln.strip() for ln in lf.read_text().split("\n") if ln.strip()]
        if not lines:
            rows.append(dict(split=split, image=lf.stem, class_id=-1,
                             class_name="background", crop="none",
                             cx=np.nan, cy=np.nan,
                             bw=np.nan, bh=np.nan, area=np.nan))
            continue
        for line in lines:
            p = line.split()
            cid = int(p[0])
            cx, cy, bw, bh = map(float, p[1:5])
            cname = CLASS_NAMES[cid]
            rows.append(dict(split=split, image=lf.stem,
                             class_id=cid, class_name=cname,
                             crop=cname.split("_")[0],
                             cx=cx, cy=cy, bw=bw, bh=bh, area=bw * bh))
    return pd.DataFrame(rows)


def generate_figures(save_dir=None) -> None:
    """
    Generate and save all publication figures to OUTPUT_DIR/.

    Parameters
    ----------
    save_dir : Path or None
        Path to the Ultralytics training run directory.  When provided and
        results.csv is present, an additional Fig 10 (training metrics) is saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive; works headlessly
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        print(f"  ⚠  Figure generation skipped (missing dependency: {exc})")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Publication rcParams ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family"        : "DejaVu Sans",
        "font.size"          : 11,
        "axes.titlesize"     : 13,
        "axes.titleweight"   : "bold",
        "axes.labelsize"     : 11,
        "xtick.labelsize"    : 10,
        "ytick.labelsize"    : 10,
        "legend.fontsize"    : 10,
        "legend.framealpha"  : 0.9,
        "figure.dpi"         : 150,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.15,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.3,
        "grid.linestyle"     : "--",
        "axes.axisbelow"     : True,
    })

    # ── Colour palette ────────────────────────────────────────────────────────
    CROP_PAL  = {"Corn": "#E8973A", "Pepper": "#27AE60", "Tomato": "#C0392B"}
    SPLIT_PAL = {"train": "#2980B9", "valid": "#27AE60", "test":  "#E74C3C"}
    HEALTHY   = "#3498DB"

    def cls_color(name):
        if "Healthy" in name:
            return HEALTHY
        for crop, col in CROP_PAL.items():
            if name.startswith(crop):
                return col
        return "#95A5A6"

    CLS_COLORS = [cls_color(c) for c in CLASS_NAMES]

    # ── Load annotation data once (reused by all figures) ────────────────────
    print("  Loading annotation data …")
    dfs    = {s: _load_split_df(s) for s in ["train", "valid", "test"]}
    df_all = pd.concat(dfs.values(), ignore_index=True)
    df_box = df_all[df_all.class_id >= 0].copy()
    n_imgs = {s: len(list((DATA_DIR / s / "images").glob("*.*")))
              for s in ["train", "valid", "test"]}
    for s, df in dfs.items():
        print(f"    {s:6s}: {n_imgs[s]:5d} imgs | "
              f"{(df.class_id>=0).sum():6d} boxes | "
              f"{(df.class_id<0).sum():4d} empty")

    saved = []

    # ── Fig 01: Dataset Split Overview ───────────────────────────────────────
    splits   = ["train", "valid", "test"]
    n_img_v  = [n_imgs[s]                       for s in splits]
    n_box_v  = [(dfs[s].class_id >= 0).sum()    for s in splits]
    n_emp_v  = [(dfs[s].class_id <  0).sum()    for s in splits]
    clrs     = [SPLIT_PAL[s]                    for s in splits]
    total_i  = sum(n_img_v)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset Split Overview", fontsize=15, fontweight="bold", y=1.02)

    ax = axes[0]
    bars = ax.bar(splits, n_img_v, color=clrs, width=0.5, zorder=3)
    for bar, n in zip(bars, n_img_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f"{n:,}\n({n/total_i*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Images per Split"); ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(n_img_v) * 1.3)

    ax = axes[1]
    n_ann = n_img_v[0] - n_emp_v[0]
    _, _, ats = ax.pie(
        [n_ann, n_emp_v[0]],
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
    tc   = dfs["train"][dfs["train"].class_id >= 0]
    cc   = tc.groupby("class_name").size().reindex(CLASS_NAMES, fill_value=0)
    ypos = np.arange(len(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(ypos, cc.values, color=CLS_COLORS, height=0.68, zorder=3)
    for bar, val in zip(bars, cc.values):
        ax.text(bar.get_width() + 35, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Bounding Boxes")
    ax.set_title("Training Set — Per-Class Annotation Count", pad=12)
    ax.set_xlim(0, cc.max() * 1.18)
    for b in [4.5, 14.5]:
        ax.axhline(y=b, color="#7F8C8D", lw=0.9, linestyle="--", alpha=0.7)
    for i, (nm, cnt) in enumerate(zip(CLASS_NAMES, cc.values)):
        if cnt < 30:
            ax.text(cnt + 35, i, "  ⚠ < 30", va="center",
                    color="#E74C3C", fontsize=8.5, fontstyle="italic")
    patches = [
        mpatches.Patch(color=CROP_PAL["Corn"],   label="Corn (classes 0–4)"),
        mpatches.Patch(color=CROP_PAL["Pepper"], label="Pepper (classes 5–14)"),
        mpatches.Patch(color=CROP_PAL["Tomato"], label="Tomato (classes 15–22)"),
        mpatches.Patch(color=HEALTHY,            label="Healthy variants"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_02_class_distribution_train.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_02  class distribution (train)")

    # ── Fig 03: Cross-Split Distribution ─────────────────────────────────────
    cnt_abs = {
        s: dfs[s][dfs[s].class_id >= 0]
           .groupby("class_name").size()
           .reindex(CLASS_NAMES, fill_value=0)
        for s in ["train", "valid", "test"]
    }
    cdf   = pd.DataFrame(cnt_abs)
    cnorm = cdf.div(cdf.sum(axis=0), axis=1) * 100
    yp    = np.arange(len(CLASS_NAMES))
    w     = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle("Class Distribution Across Dataset Splits",
                 fontsize=15, fontweight="bold", y=1.01)
    for ax, data, xlabel, title in [
        (axes[0], cdf,   "Annotation Box Count",             "Absolute Box Counts"),
        (axes[1], cnorm, "Relative Frequency (% of split)",  "Normalised Distribution"),
    ]:
        for i, (sp, col) in enumerate(SPLIT_PAL.items()):
            ax.barh(yp + (i - 1) * w, data[sp], height=w,
                    color=col, alpha=0.85, label=sp.capitalize(), zorder=3)
        ax.set_yticks(yp)
        ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES], fontsize=8.5)
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
    fig.suptitle("Annotation Density — Boxes per Image",
                 fontsize=15, fontweight="bold")
    for ax, split in zip(axes, ["train", "valid"]):
        per_img = (dfs[split][dfs[split].class_id >= 0]
                   .groupby("image").size())
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
    tb = df_box[df_box.split == "train"]
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
        h = ax.hist2d(subset.cx.values, subset.cy.values,
                      bins=40, cmap="YlOrRd", density=True, cmin=1e-6)
        plt.colorbar(h[3], ax=ax, shrink=0.8, label="Density")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()
        ax.set_xlabel("cx"); ax.set_ylabel("cy"); ax.set_title(label)
        ax.set_aspect("equal"); ax.grid(False)
        ax.plot(0.5, 0.5, "w+", markersize=10, markeredgewidth=2)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_05_bbox_spatial_heatmap.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_05  bounding-box spatial heatmap")

    # ── Fig 06: Bounding Box Geometry ────────────────────────────────────────
    tb2 = df_box[df_box.split == "train"].copy()
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
    ax_sc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Square (w=h)")
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
        (ax_wh, "bw",     "Normalised Width",   "Width Distribution"),
        (ax_hh, "bh",     "Normalised Height",  "Height Distribution"),
        (ax_ar, "aspect", "Width / Height",     "Aspect Ratio Distribution"),
    ]:
        for crop, color in CROP_PAL.items():
            vals = tb2[tb2.crop == crop][col].dropna()
            vals = vals[vals < vals.quantile(0.99)]
            ax.hist(vals, bins=40, color=color, alpha=0.50,
                    density=True, label=crop, edgecolor="none")
        ax.set_xlabel(lbl); ax.set_ylabel("Density")
        ax.set_title(title); ax.legend(fontsize=9)

    out = OUTPUT_DIR / "fig_06_bbox_size_analysis.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_06  bounding-box geometry")

    # ── Fig 07: Class Imbalance ───────────────────────────────────────────────
    cnt_r = {}
    for s in ["train", "valid", "test"]:
        n = (dfs[s][dfs[s].class_id >= 0]
             .groupby("class_name").size()
             .reindex(CLASS_NAMES, fill_value=0))
        cnt_r[s] = n / n.sum()
    cdf2 = pd.DataFrame(cnt_r)
    rv = (cdf2["valid"] / cdf2["train"].clip(1e-9)).clip(0, 5)
    rt = (cdf2["test"]  / cdf2["train"].clip(1e-9)).clip(0, 5)

    fig, ax = plt.subplots(figsize=(11, 8))
    yp2 = np.arange(len(CLASS_NAMES)); w2 = 0.36
    ax.barh(yp2 - w2 / 2, rv.values, height=w2, color=SPLIT_PAL["valid"],
            alpha=0.82, label="Val / Train ratio", zorder=3)
    ax.barh(yp2 + w2 / 2, rt.values, height=w2, color=SPLIT_PAL["test"],
            alpha=0.82, label="Test / Train ratio", zorder=3)
    ax.axvline(1.0, color="black", lw=1.2, linestyle="--",
               alpha=0.7, label="Balanced (ratio = 1.0)")
    ax.axvspan(0.7, 1.3, alpha=0.06, color="green", label="±30 % balance zone")
    ax.set_yticks(yp2)
    ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Ratio of relative frequencies (val or test / train)")
    ax.set_title("Class Imbalance — Split Frequency Ratios\n"
                 "Values near 1.0 indicate well-balanced representation", pad=10)
    ax.legend(loc="lower right"); ax.set_xlim(0, 5.2)
    for b in [4.5, 14.5]:
        ax.axhline(y=b, color="#7F8C8D", lw=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_07_class_imbalance.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_07  class imbalance")

    # ── Fig 08: Training Configuration Table ─────────────────────────────────
    ni_tr = n_imgs["train"]; ni_va = n_imgs["valid"]
    cfg_rows = [
        ("Model",               MODEL_SIZE,              "Fastest YOLO26; ~4 MB, 3.2 GFLOPs"),
        ("Image size",          f"{IMG_SIZE}×{IMG_SIZE}", "Matches Roboflow pre-processing"),
        ("Batch size",          str(BASE_BATCH),         "Per-GPU; 32 × yolo26n fits in 24 GB; stabler gradients"),
        ("Epochs",              "200",                   "Cosine LR needs room to decay"),
        ("Early-stop patience", "25",                    "Stop after 25 non-improving epochs"),
        ("Optimizer",           "MuSGD",                 "YOLO26 hybrid Muon + SGD"),
        ("LR (lr0 → lrf)",      "0.01 → 0.01",          "Cosine annealing, flat final ratio"),
        ("Warmup epochs",       "3",                     "Reduced from 5; faster ramp with larger batch"),
        ("Freeze layers",       "5",                     "First 5 backbone layers only; allows texture adaptation"),
        ("Close mosaic",        "20 epochs",             "More clean fine-tune epochs → confident weights"),
        ("cls loss weight",     "0.5",                   "Raised from 0.3; stronger classification signal"),
        ("Label smoothing",     "0.05",                  "Mild; prevents overfit without suppressing confidence"),
        ("Copy-paste aug",      "0.1",                   "Reduced from 0.3; less inter-class patch confusion"),
        ("Cache",               "Disk",                  "Deterministic; avoids YOLO non-determinism warning"),
        ("AMP",                 "True (fp16)",           "~1.4× speedup on M4 Pro MPS"),
        ("Hard negatives",      str(NUM_NEGATIVES),      "Diverse non-crop images; empty labels (OOD guard)"),
        ("Device",              "MPS (M4 Pro)",          "Apple Silicon Metal Performance Shaders"),
        ("Train images",        f"{ni_tr:,}",            f"Incl. {NUM_NEGATIVES} hard-negative images"),
        ("Val images",          f"{ni_va:,}",            "Roboflow auto-split"),
        ("Classes",             "23",                    "Corn ×5, Pepper ×10, Tomato ×8"),
        ("Dataset source",      "Ghana Crop Disease v2", "Roboflow — CC BY 4.0"),
    ]

    fig, ax = plt.subplots(figsize=(15, 8.2))
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
            tbl[row, col].set_height(0.048)
    ax.set_title("Training Configuration Summary",
                 fontsize=14, fontweight="bold", pad=20, y=0.98)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_08_training_config.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_08  training config table")

    # ── Fig 09: LR Schedule + Augmentation Profile ───────────────────────────
    ep_arr = np.arange(1, EPOCHS_DEFAULT + 1)

    def _cosine_lr(ep, warmup=5, lr0=0.01, lrf=0.01, total=EPOCHS_DEFAULT):
        if ep <= warmup:
            return lr0 * ep / warmup
        prog = (ep - warmup) / (total - warmup)
        return lr0 * (lrf + (1 - lrf) * 0.5 * (1 + np.cos(np.pi * prog)))

    lr_vals = np.array([_cosine_lr(e) for e in ep_arr])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Training Schedule & Augmentation Profile",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(ep_arr, lr_vals, color="#2980B9", lw=2.2, zorder=4, label="Learning Rate")
    ax.fill_between(ep_arr, 0, lr_vals, alpha=0.12, color="#2980B9")
    ax.axvspan(1,  5,                       alpha=0.12, color="#F39C12",
               label="Warmup (5 ep)")
    ax.axvspan(6,  EPOCHS_DEFAULT - 10,     alpha=0.06, color="#27AE60",
               label="Cosine Decay")
    ax.axvspan(EPOCHS_DEFAULT - 9, EPOCHS_DEFAULT, alpha=0.12, color="#E74C3C",
               label="Close Mosaic (last 10 ep)")
    ax.axvline(10, color="#8E44AD", linestyle=":", lw=1.8,
               label="Unfreeze backbone @ ep 10")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title(f"Cosine LR Schedule  (lr0=0.01, {EPOCHS_DEFAULT} epochs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(1, EPOCHS_DEFAULT); ax.set_ylim(bottom=0)

    aug_params = {
        "Mosaic":              1.000, "HSV Saturation":    0.700,
        "Scale":               0.500, "Flip LR":           0.500,
        "HSV Value":           0.400, "Erasing":           0.400,
        "MixUp":               0.150, "Translation":       0.100,
        "Copy-Paste":          0.100, "Label Smoothing":   0.050,
        "HSV Hue":             0.015, "Rotation / 90°":   10.0 / 90,
    }
    ax2 = axes[1]
    ay  = np.arange(len(aug_params))
    av  = list(aug_params.values())
    al  = list(aug_params.keys())
    ac  = ["#C0392B" if v >= 0.5 else "#2980B9" if v >= 0.1 else "#95A5A6" for v in av]
    ax2.barh(ay, av, color=ac, height=0.65, zorder=3)
    ax2.set_yticks(ay); ax2.set_yticklabels(al, fontsize=9.5); ax2.invert_yaxis()
    ax2.set_xlabel("Parameter Value"); ax2.set_title("Augmentation Parameters")
    ax2.set_xlim(0, 1.18)
    for i, v in enumerate(av):
        ax2.text(v + 0.015, i,
                 f"{v:.3f}".rstrip("0").rstrip("."), va="center", fontsize=9)
    leg = [
        mpatches.Patch(color="#C0392B", label="Strong (≥0.5)"),
        mpatches.Patch(color="#2980B9", label="Moderate (0.1–<0.5)"),
        mpatches.Patch(color="#95A5A6", label="Mild (<0.1)"),
    ]
    ax2.legend(handles=leg, loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_09_lr_schedule_augmentation.png"
    plt.savefig(out); plt.close(); saved.append(out)
    print("  ✓  fig_09  LR schedule + augmentation")

    # ── Fig 10: Training Metrics (post-training only) ─────────────────────────
    if save_dir is not None:
        csv_path = Path(save_dir) / "results.csv"
        if csv_path.exists():
            df_res = pd.read_csv(csv_path)
            df_res.columns = df_res.columns.str.strip()
            metric_cols = [
                ("train/box_loss",       "Box Loss (train)",   "#E74C3C"),
                ("train/cls_loss",       "Class Loss (train)", "#3498DB"),
                ("val/box_loss",         "Box Loss (val)",     "#E67E22"),
                ("metrics/mAP50(B)",     "mAP@0.50",           "#27AE60"),
                ("metrics/mAP50-95(B)",  "mAP@0.50:0.95",      "#9B59B6"),
                ("metrics/precision(B)", "Precision",          "#F39C12"),
                ("metrics/recall(B)",    "Recall",             "#1ABC9C"),
            ]
            available = [(c, l, k) for c, l, k in metric_cols if c in df_res.columns]
            if available:
                ncols = 3
                nrows = (len(available) + ncols - 1) // ncols
                fig, axes_grid = plt.subplots(nrows, ncols,
                                              figsize=(18, 5 * nrows))
                flat_axes = list(np.array(axes_grid).flatten())
                fig.suptitle(f"Training Metrics — {EXP_NAME}",
                             fontsize=15, fontweight="bold")
                for ax, (col, lbl, color) in zip(flat_axes, available):
                    ax.plot(df_res[col], color=color, lw=2)
                    ax.set_title(lbl); ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=0.3)
                for ax in flat_axes[len(available):]:
                    ax.set_visible(False)
                plt.tight_layout()
                out = OUTPUT_DIR / "fig_10_training_metrics.png"
                plt.savefig(out); plt.close(); saved.append(out)
                print("  ✓  fig_10  training metrics")
            else:
                print("  –  fig_10 skipped (no recognised metric columns in results.csv)")
        else:
            print(f"  –  fig_10 skipped (results.csv not found in {save_dir})")
    else:
        print("  –  fig_10 skipped (run after training to include metric curves)")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  Publication figures  →  {OUTPUT_DIR}")
    print(sep)
    for f in saved:
        print(f"  {f.name:<52}  {f.stat().st_size / 1024:>7.1f} KB")
    print(sep)
    print(f"  Total: {len(saved)} figures  |  300 DPI  |  PNG")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete YOLO26 crop-disease training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                     # full pipeline
  python train.py --dry-run           # 1-epoch timing validation
  python train.py --skip-negatives    # negatives already staged
  python train.py --figures-only      # regenerate figures, no training
  python train.py --no-figures        # train only, skip figures
  DRY_RUN=1 python train.py           # dry-run via env var
        """.strip(),
    )
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run 1 epoch; print epoch-time estimate")
    parser.add_argument("--epochs",         type=int, default=None,
                        help=f"Override epoch count (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--skip-negatives", action="store_true",
                        help="Skip hard-negative download/staging")
    parser.add_argument("--figures-only",   action="store_true",
                        help="Regenerate publication figures only; skip training")
    parser.add_argument("--no-figures",     action="store_true",
                        help="Train without generating figures afterwards")
    args = parser.parse_args()

    dry_run = args.dry_run or os.environ.get("DRY_RUN", "0") == "1"
    epochs  = 1 if dry_run else (args.epochs or EPOCHS_DEFAULT)

    # ── Figures-only shortcut ─────────────────────────────────────────────────
    if args.figures_only:
        print("─── Figures-only mode ───────────────────────────────────────")
        run_dir = None
        candidates = sorted(
            RUNS_DIR.glob(f"{EXP_NAME}*/results.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            run_dir = candidates[-1].parent
            print(f"  Using training results: {run_dir}")
        else:
            print("  No training results found; Fig 10 will be skipped.")
        generate_figures(save_dir=run_dir)
        return

    # ── Step 1: data_fixed.yaml ───────────────────────────────────────────────
    print("\n─── Step 1/3: data_fixed.yaml ───────────────────────────────────")
    prepare_data_yaml()

    # ── Step 2: Hard negatives ────────────────────────────────────────────────
    print("\n─── Step 2/3: Hard-negative images (OOD guard) ──────────────────")
    prepare_hard_negatives(skip=args.skip_negatives)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n─── Step 3/3: Training ──────────────────────────────────────────")
    device, batch, use_amp = resolve_device()
    is_mps   = device == "mps"
    is_multi = isinstance(device, list)

    if is_mps:
        _patch_tal_for_mps()

    n_train = len(list((DATA_DIR / "train" / "images").glob("*.*")))
    n_val   = len(list((DATA_DIR / "valid" / "images").glob("*.*")))

    last_pt  = RUNS_DIR / EXP_NAME / "weights" / "last.pt"
    resumable = last_pt.exists() and _is_resumable(last_pt)

    if not resumable and last_pt.exists():
        print(f"  ⚠  last.pt found but has no training state (weights-only export).")
        print(f"     Starting a fresh training run with full hyperparameters.")

    if resumable:
        print(f"  Resuming from: {last_pt}")
        model = YOLO(str(last_pt))
        log_startup(device, batch, n_train, n_val, "resumed", epochs, dry_run, use_amp)
        t0      = time.perf_counter()
        results = model.train(resume=True)
    else:  # fresh training (no last.pt, or last.pt is weights-only)
        if dry_run:
            print("  ⚡  DRY-RUN — 1-epoch validation + timing")
        model = YOLO(f"{MODEL_SIZE}.pt")
        log_startup(device, batch, n_train, n_val, MODEL_SIZE, epochs, dry_run, use_amp)

        # ─────────────────────────────────────────────────────────────────────
        # Non-default hyperparameter choices — one-line rationale each
        # ─────────────────────────────────────────────────────────────────────
        # device=mps/[0,1,…] : Apple Silicon GPU or all CUDA GPUs (auto-detected)
        # batch=32           : doubled from 16 — yolo26n is tiny; larger batch
        #                      gives stabler gradients and raises confidence scores
        # workers=0          : MPS requires 0; macOS fork() conflicts with Metal
        # cache="disk"       : deterministic; avoids YOLO's non-determinism warning
        # amp=False on MPS   : fp16 corrupts TAL assigner index tensors → AcceleratorError ~ep 5
        # epochs=200         : cosine LR needs room to decay on a 3k-image dataset
        # patience=25        : early-stop after 25 non-improving epochs (default 50)
        # freeze=5           : lock only 5 backbone layers (was 10); allows deeper
        #                      adaptation to crop-disease textures → higher AP
        # close_mosaic=20    : 20 clean fine-tune epochs (was 10); more stable final
        #                      weights → higher per-class confidence at inference
        # cls=0.5            : raised from 0.3 (YOLO default); higher classification
        #                      loss weight → model is more decisive about class identity
        # label_smoothing=0.05: halved from 0.1; smoothing=0.1 was training targets
        #                      to max ~0.9 confidence, pushing inference scores below
        #                      0.48. Hard negatives already handle OOD; smoothing
        #                      only needs to be mild enough to prevent overfit.
        # copy_paste=0.1     : reduced from 0.3; heavy copy-paste mixes disease
        #                      patches across images, confusing the classifier and
        #                      reducing per-class confidence
        # seed=42            : reproducible runs for fair paper comparisons
        # deterministic      : False under DDP (required by PyTorch distributed)
        # ─────────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = model.train(
            # ── Data ──────────────────────────────────────────────────────────
            data        = str(FIXED_YAML),
            imgsz       = IMG_SIZE,
            batch       = batch,
            cache       = "disk",
            workers     = 0 if is_mps else 8,

            # ── Training schedule ─────────────────────────────────────────────
            epochs      = epochs,
            patience    = PATIENCE,
            optimizer   = "MuSGD",
            lr0         = 0.01,
            lrf         = 0.01,
            momentum    = 0.937,
            weight_decay= 0.001,
            warmup_epochs   = 3,
            warmup_momentum = 0.8,
            cos_lr      = True,

            # ── Regularisation / fine-tuning ──────────────────────────────────
            freeze      = 5,
            close_mosaic= 20,

            # ── Augmentation ──────────────────────────────────────────────────
            hsv_h       = 0.015,
            hsv_s       = 0.7,
            hsv_v       = 0.4,
            degrees     = 10.0,
            translate   = 0.1,
            scale       = 0.5,
            shear       = 0.0,
            perspective = 0.0001,
            flipud      = 0.0,
            fliplr      = 0.5,
            bgr         = 0.0,
            mosaic      = 1.0,
            mixup       = 0.15,
            copy_paste  = 0.1,
            erasing     = 0.4,

            # ── Loss weights ──────────────────────────────────────────────────
            box         = 7.5,
            cls         = 0.5,
            dfl         = 1.5,

            # ── OOD / confidence ──────────────────────────────────────────────
            label_smoothing = 0.05,
            conf        = None,

            # ── Misc ──────────────────────────────────────────────────────────
            device      = device,
            project     = str(RUNS_DIR),
            name        = EXP_NAME,
            exist_ok    = True,
            pretrained  = True,
            amp         = use_amp,
            verbose     = True,
            plots       = True,
            save        = True,
            save_period = 10,
            val         = True,
            seed        = 42,
            deterministic = not is_multi,
        )

    elapsed = time.perf_counter() - t0

    if dry_run:
        log_dry_run_summary(elapsed, EPOCHS_DEFAULT, str(results.save_dir))
        print("(Figures skipped in dry-run mode — omit --dry-run to generate them)")
        return

    print(f"\n✅  Training complete!  ({elapsed / 3600:.1f} h total)")
    print(f"   Best model: {results.save_dir}/weights/best.pt")

    # ── Step 4: Publication figures ───────────────────────────────────────────
    if not args.no_figures:
        print("\n─── Generating publication figures ──────────────────────────")
        generate_figures(save_dir=Path(results.save_dir))


if __name__ == "__main__":
    main()
