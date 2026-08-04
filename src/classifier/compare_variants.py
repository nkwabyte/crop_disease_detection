#!/usr/bin/env python3
"""
compare_variants.py — evaluate the 3-class and 4-class classifiers head to head.

The question this answers: is it better to reject non-crop images by thresholding
a 3-class softmax, or by training an explicit "Other" class?

Both models are scored on identical data so the numbers are comparable:

  crop accuracy   — 3-way accuracy over genuine crop images only. The OOD model
                    must not trade crop accuracy for rejection ability, so this
                    is the guard-rail metric.
  OOD rejection   — share of non-crop images not assigned to a crop class.
                      base : top softmax probability < threshold
                      ood  : argmax == "Other"
                    Reported per source folder, because rejecting a stock photo
                    is a far easier task than rejecting an eggplant leaf, and a
                    single pooled number hides that difference.

Results are written to outputs/benchmarks/classifier_variant_comparison.{json,md}
for direct use in the paper.

Usage
-----
  python -m src.classifier.compare_variants
  python -m src.classifier.compare_variants --thresholds 0.5 0.55 0.7 0.9
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.classifier.config import (CONF_DEFAULT, CROP_CLASSES, DATASET_DIR,
                                   IMG_SIZE, OUTPUT_DIR, PROJECT_ROOT)
from src.classifier.train_classifier import VARIANTS, build_model

BENCH_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
OTHER_ID = 3

# ── Figure styling ───────────────────────────────────────────────────────────
# Two series only, so two categorical slots (blue, orange). This pair validates
# on the light surface: CVD dE 24.7, normal-vision dE 33.6, both well clear of
# the >=8 / >=15 floors. Text never wears the series colour — identity comes from
# the coloured mark beside it.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
C_BASE = "#2a78d6"      # 3-class + threshold
C_OOD = "#eb6834"       # 4-class, learned "Other"

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RowDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.rows = df[["image_path", "crop_id"]].values.tolist()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        rel, label = self.rows[i]
        img = Image.open(PROJECT_ROOT / rel).convert("RGB")
        return EVAL_TF(img), int(label)


@torch.no_grad()
def run_model(ckpt_path: Path, n_classes: int, df: pd.DataFrame, device):
    model = build_model(n_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ld = DataLoader(RowDataset(df), batch_size=128, shuffle=False,
                    num_workers=8, pin_memory=(device.type == "cuda"))
    confs, preds = [], []
    for imgs, _ in ld:
        probs = torch.softmax(model(imgs.to(device, non_blocking=True)), dim=1)
        c, p = probs.max(dim=1)
        confs += c.cpu().tolist()
        preds += p.cpu().tolist()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return confs, preds


def _style_axes(ax) -> None:
    """Recessive frame: hairline horizontal grid, no box, ink-token text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.set_axisbelow(True)


def make_figures(results: dict, best_thr_key: str, out_dir: Path) -> list[Path]:
    """Write the comparison charts. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base_thr = results["base"]["thresholds"][best_thr_key]

    # ── 1. Headline: does the learned class beat the threshold? ──────────────
    # Grouped columns: the reader compares two models on two metrics, so the
    # series ARE the subject -> categorical colour, legend + value labels.
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURFACE)
    metrics = ["Crop accuracy", "Non-crop rejection"]
    base_vals = [base_thr["crop_retention"] * 100, base_thr["ood_rejection"] * 100]
    ood_vals = [results["ood"]["crop_accuracy"] * 100,
                results["ood"]["ood_rejection"] * 100]
    x = range(len(metrics))
    w = 0.20                                   # leaves air in the band
    gap = 0.012                                # surface gap between the pair
    b1 = ax.bar([i - w / 2 - gap for i in x], base_vals, w,
                label=f"3-class + threshold {best_thr_key}", color=C_BASE)
    b2 = ax.bar([i + w / 2 + gap for i in x], ood_vals, w,
                label='4-class, learned "Other"', color=C_OOD)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=10, color=INK)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    _style_axes(ax)
    ax.set_title("Rejecting non-crop images: threshold vs a learned class",
                 fontsize=12, color=INK, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left",
              bbox_to_anchor=(0, -0.22), ncol=2)
    fig.tight_layout()
    p = out_dir / "fig_clf_01_headline_comparison.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # ── 2. Rejection per negative source ─────────────────────────────────────
    # Horizontal so the source names stay readable, and because the interesting
    # story is the spread across sources, not a single total.
    srcs = sorted(results["ood"]["per_source"], key=lambda s: results["ood"]["per_source"][s])
    if srcs:
        fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(srcs) + 2.4), dpi=200,
                               facecolor=SURFACE)
        y = range(len(srcs))
        h = 0.26
        gap = 0.012
        bb = ax.barh([i + h / 2 + gap for i in y],
                     [base_thr["per_source"].get(s, 0) * 100 for s in srcs], h,
                     label=f"3-class + threshold {best_thr_key}", color=C_BASE)
        bo = ax.barh([i - h / 2 - gap for i in y],
                     [results["ood"]["per_source"][s] * 100 for s in srcs], h,
                     label='4-class, learned "Other"', color=C_OOD)
        for bars in (bb, bo):
            for r in bars:
                ax.text(r.get_width() + 1.2, r.get_y() + r.get_height() / 2,
                        f"{r.get_width():.0f}%", va="center", ha="left",
                        fontsize=8.5, color=INK)
        ax.set_yticks(list(y))
        ax.set_yticklabels([s.replace("ood", "generic photos") for s in srcs],
                           fontsize=10, color=INK)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        _style_axes(ax)
        ax.set_title("Rejection rate by negative source",
                     fontsize=12, color=INK, pad=14, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2,
                  loc="upper left", bbox_to_anchor=(0, -0.10), ncol=2)
        fig.tight_layout()
        p = out_dir / "fig_clf_02_rejection_by_source.png"
        fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # ── 3. What the threshold actually buys (3-class model) ──────────────────
    # Both series are percentages on one axis - never a second y-scale.
    thrs = sorted(float(k) for k in results["base"]["thresholds"])
    if len(thrs) > 1:
        keep = [results["base"]["thresholds"][f"{t:.2f}"]["crop_retention"] * 100
                for t in thrs]
        rej = [results["base"]["thresholds"][f"{t:.2f}"]["ood_rejection"] * 100
               for t in thrs]
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200, facecolor=SURFACE)
        ax.plot(thrs, keep, color=C_BASE, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round",
                label="Crop retention")
        ax.plot(thrs, rej, color=C_OOD, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round",
                label="Non-crop rejection")
        # Label the endpoints only - a value on every point goes unread.
        ax.text(thrs[-1], keep[-1] + 3, f"{keep[-1]:.1f}%", ha="right",
                fontsize=9, color=INK)
        ax.text(thrs[-1], rej[-1] - 6, f"{rej[-1]:.1f}%", ha="right",
                fontsize=9, color=INK)
        ax.axvline(CONF_DEFAULT, color=GRID, linewidth=1, zorder=0)
        ax.text(CONF_DEFAULT, 103, f" shipped default {CONF_DEFAULT}", fontsize=8.5,
                color=INK_2, ha="left", va="top")
        ax.set_xlabel("Softmax confidence threshold", fontsize=10, color=INK_2)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        _style_axes(ax)
        ax.set_title("The 3-class trade-off: every point of rejection costs crop recall",
                     fontsize=12, color=INK, pad=14, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")
        fig.tight_layout()
        p = out_dir / "fig_clf_03_threshold_sweep.png"
        fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.50, CONF_DEFAULT, 0.70, 0.80, 0.90, 0.95])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # The 4-class test split contains both the crop test images and the held-out
    # negatives, so it is the single evaluation set for both models.
    test_csv = DATASET_DIR / "classifier_ood_test.csv"
    if not test_csv.exists():
        raise SystemExit(f"  x {test_csv} missing — run generate_classifier_csv.py "
                         f"--with-negatives first")
    df = pd.read_csv(test_csv)
    df = df[[(PROJECT_ROOT / p).exists() for p in df["image_path"]]].reset_index(drop=True)
    crops = df[df["crop_id"] != OTHER_ID].reset_index(drop=True)
    others = df[df["crop_id"] == OTHER_ID].reset_index(drop=True)
    print(f"  test set: {len(crops):,} crop images, {len(others):,} non-crop images")
    if "source" in others.columns:
        for s, n in others["source"].value_counts().items():
            print(f"    {s:<12} {n:5d}")

    base_ck = OUTPUT_DIR / "best.pth"
    ood_ck = VARIANTS["ood"]["dir"] / "best.pth"
    for p in (base_ck, ood_ck):
        if not p.exists():
            raise SystemExit(f"  x checkpoint missing: {p}")

    results: dict = {"n_crop_test": len(crops), "n_ood_test": len(others)}

    # ── base: 3-class + confidence threshold ────────────────────────────────
    print("\n  [base] 3-class model")
    bc_conf, bc_pred = run_model(base_ck, 3, crops, device)
    bo_conf, bo_pred = run_model(base_ck, 3, others, device)
    correct = [p == t for p, t in zip(bc_pred, crops["crop_id"].tolist())]
    results["base"] = {
        "crop_accuracy_no_threshold": sum(correct) / len(correct),
        "thresholds": {},
    }
    print(f"    crop accuracy (no threshold): {sum(correct) / len(correct) * 100:.2f}%")
    for thr in args.thresholds:
        kept = sum(1 for c, ok in zip(bc_conf, correct) if c >= thr and ok) / len(correct)
        rej = sum(1 for c in bo_conf if c < thr) / len(bo_conf)
        per_src = {}
        if "source" in others.columns:
            by = defaultdict(list)
            for c, s in zip(bo_conf, others["source"]):
                by[s].append(c)
            per_src = {s: sum(1 for c in v if c < thr) / len(v) for s, v in by.items()}
        results["base"]["thresholds"][f"{thr:.2f}"] = {
            "crop_retention": kept, "ood_rejection": rej, "per_source": per_src}
        print(f"    thr {thr:.2f}: crop retention {kept * 100:6.2f}%  "
              f"OOD rejection {rej * 100:6.2f}%")

    # ── ood: 4-class with a learned "Other" ─────────────────────────────────
    print("\n  [ood] 4-class model")
    oc_conf, oc_pred = run_model(ood_ck, 4, crops, device)
    oo_conf, oo_pred = run_model(ood_ck, 4, others, device)
    crop_ok = sum(1 for p, t in zip(oc_pred, crops["crop_id"].tolist()) if p == t)
    crops_called_other = sum(1 for p in oc_pred if p == OTHER_ID)
    ood_rej = sum(1 for p in oo_pred if p == OTHER_ID) / len(oo_pred)
    per_src = {}
    if "source" in others.columns:
        by = defaultdict(list)
        for p, s in zip(oo_pred, others["source"]):
            by[s].append(p)
        per_src = {s: sum(1 for p in v if p == OTHER_ID) / len(v) for s, v in by.items()}
    results["ood"] = {
        "crop_accuracy": crop_ok / len(oc_pred),
        "crops_misrouted_to_other": crops_called_other / len(oc_pred),
        "ood_rejection": ood_rej,
        "per_source": per_src,
    }
    print(f"    crop accuracy: {crop_ok / len(oc_pred) * 100:.2f}%   "
          f"(crops wrongly called Other: {crops_called_other / len(oc_pred) * 100:.2f}%)")
    print(f"    OOD rejection: {ood_rej * 100:.2f}%")
    for s, v in sorted(per_src.items()):
        print(f"      {s:<12} {v * 100:6.2f}%")

    # ── write artifacts ─────────────────────────────────────────────────────
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "classifier_variant_comparison.json").write_text(
        json.dumps(results, indent=2))

    best_thr = max(results["base"]["thresholds"].items(),
                   key=lambda kv: kv[1]["crop_retention"] + kv[1]["ood_rejection"])
    sources = sorted(results["ood"]["per_source"])
    md = ["# Classifier rejection: threshold vs learned 'Other' class", "",
          f"Test set: {len(crops):,} crop images, {len(others):,} non-crop images.", "",
          "| Model | Crop accuracy | OOD rejection |", "| --- | --- | --- |",
          f"| 3-class, no threshold | {results['base']['crop_accuracy_no_threshold'] * 100:.2f}% | 0.00% |",
          f"| 3-class, threshold {best_thr[0]} | {best_thr[1]['crop_retention'] * 100:.2f}% | {best_thr[1]['ood_rejection'] * 100:.2f}% |",
          f"| 4-class, learned Other | {results['ood']['crop_accuracy'] * 100:.2f}% | {results['ood']['ood_rejection'] * 100:.2f}% |",
          "", "## Rejection by negative source", "",
          "| Source | 3-class @ " + best_thr[0] + " | 4-class |", "| --- | --- | --- |"]
    for s in sources:
        b = best_thr[1]["per_source"].get(s, float("nan")) * 100
        o = results["ood"]["per_source"][s] * 100
        md.append(f"| {s} | {b:.2f}% | {o:.2f}% |")
    md += ["", "Generated by `src/classifier/compare_variants.py`."]
    (BENCH_DIR / "classifier_variant_comparison.md").write_text("\n".join(md))
    print(f"\n  wrote {BENCH_DIR / 'classifier_variant_comparison.json'}")
    print(f"  wrote {BENCH_DIR / 'classifier_variant_comparison.md'}")

    for p in make_figures(results, best_thr[0], BENCH_DIR / "figures"):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
