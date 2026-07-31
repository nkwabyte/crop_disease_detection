#!/usr/bin/env python3
"""
compare_models.py — cross-model benchmark aggregator for the research paper.

Reads whatever each pipeline has already saved and produces a single, unified
comparison of the detection models (YOLO26, Faster RCNN v2, SE-FPN final, ViTDet,
and the ablation's selected config), plus a note on the Stage-1 classifier.

Nothing here re-runs a model — it only aggregates saved artifacts, so it is safe to
run any time and degrades gracefully when a model has not been trained yet:

  • outputs/benchmarks/<model>.json        ← written by each detector's final eval
                                             (map50, per_class_ap, num_params)
  • outputs/alt_fasterrcnn_output/results.json     ← ablation study (best config)
  • outputs/yolo_output/ | runs/*/results.csv      ← Ultralytics training metrics
  • outputs/classifier_output/metrics_history.json ← Stage-1 classifier (accuracy/F1)

Outputs (to outputs/benchmarks/):
  • comparison.json          — unified machine-readable table
  • comparison_table.md      — paper-ready markdown table
  • fig_map_comparison.png   — mAP@0.5 bar chart across detection models
  • fig_params_vs_map.png    — model size vs accuracy scatter
  • fig_per_class_heatmap.png — per-class AP@0.5 heatmap (models × 23 classes)

Usage
-----
  python -m src.benchmark.compare_models
  python -m src.benchmark.compare_models --out-dir outputs/benchmarks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS      = PROJECT_ROOT / "outputs"
BENCH_DIR    = OUTPUTS / "benchmarks"

# 23 disease classes in the canonical (label_map.json / integer_label) order — used
# only to align per-class heatmap columns when models report different name spellings.
CANONICAL_CLASSES = [
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Healthy", "Corn Streak",
    "Corn Northern Leaf Blight", "Pepper Leaf Curl", "Pepper Cercospora",
    "Pepper Leaf Blight", "Pepper Bacterial Spot", "Pepper Leaf Mosaic",
    "Pepper Healthy", "Pepper Fusarium", "Pepper Septoria", "Pepper Late Blight",
    "Pepper Early Blight", "Tomato Late Blight", "Tomato Early Blight",
    "Tomato Bacterial Spot", "Tomato Septoria", "Tomato Fusarium",
    "Tomato Leaf Curl", "Tomato Healthy", "Tomato Mosaic",
]


# ── Collectors ──────────────────────────────────────────────────────────────────

def _collect_benchmark_jsons() -> list[dict]:
    """Read every outputs/benchmarks/<model>.json written by a detector's final eval."""
    records = []
    if not BENCH_DIR.exists():
        return records
    for p in sorted(BENCH_DIR.glob("*.json")):
        if p.name in {"comparison.json"}:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            if "map50" in d and "model_name" in d:
                records.append({
                    "model_name": d["model_name"],
                    "architecture": d.get("architecture", d["model_name"]),
                    "map50": d.get("map50"),
                    "num_params": d.get("num_params"),
                    "per_class_ap": d.get("per_class_ap") or {},
                    "source": p.name,
                })
        except Exception as exc:
            print(f"  ⚠  could not read {p.name}: {exc}")
    return records


def _collect_yolo() -> Optional[dict]:
    """Pull best mAP@0.5 from an Ultralytics results.csv (if a YOLO run exists)."""
    candidates = list((PROJECT_ROOT / "runs").glob("*/results.csv")) if (PROJECT_ROOT / "runs").exists() else []
    candidates += list((OUTPUTS / "yolo_output").glob("**/results.csv")) if (OUTPUTS / "yolo_output").exists() else []
    best = None
    used = None
    for csv_path in candidates:
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            col = next((c for c in (rows[0].keys() if rows else [])
                        if "mAP50" in c and "50-95" not in c), None)
            if not col:
                continue
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
            if vals:
                m = max(vals)
                if best is None or m > best:
                    best, used = m, csv_path
        except Exception:
            continue
    if best is None:
        return None
    return {
        "model_name": "yolo26n",
        "architecture": "YOLO26n (Ultralytics)",
        "map50": round(best, 5),
        "num_params": None,
        "per_class_ap": {},
        "source": str(used.relative_to(PROJECT_ROOT)),
    }


def _collect_alt_selected() -> Optional[dict]:
    """Pull the ablation study's selected config (resnet50_300) from results.json."""
    path = OUTPUTS / "alt_fasterrcnn_output" / "results.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    def _extract(entry: dict) -> Optional[float]:
        for k in ("map50", "best_map", "mAP50", "val_map50"):
            if isinstance(entry.get(k), (int, float)):
                return float(entry[k])
        return None

    entry = None
    if isinstance(data, dict):
        entry = data.get("resnet50_300") or data.get("configs", {}).get("resnet50_300")
        if entry is None:  # fall back to the best-scoring config
            best_id, best_m = None, None
            for cid, e in (data.get("configs", data)).items():
                if isinstance(e, dict):
                    m = _extract(e)
                    if m is not None and (best_m is None or m > best_m):
                        best_id, best_m, entry = cid, m, e
    if not isinstance(entry, dict):
        return None
    m = _extract(entry)
    if m is None:
        return None
    return {
        "model_name": "fasterrcnn_ablation_resnet50_300",
        "architecture": "Faster RCNN (ablation, selected config resnet50_300)",
        "map50": round(m, 5),
        "num_params": entry.get("num_params"),
        "per_class_ap": entry.get("per_class_ap") or {},
        "source": "alt_fasterrcnn_output/results.json",
    }


def _collect_classifier() -> Optional[dict]:
    """Stage-1 classifier best accuracy/F1 (reported separately — different task)."""
    path = OUTPUTS / "classifier_output" / "metrics_history.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            h = json.load(f)
    except Exception:
        return None
    def _best(*keys):
        for k in keys:
            v = h.get(k)
            if isinstance(v, list) and v:
                nums = [x for x in v if isinstance(x, (int, float))]
                if nums:
                    return round(max(nums), 5)
        return None
    return {
        "model_name": "efficientnet_b2_classifier",
        "architecture": "EfficientNet-B2 (Stage-1 crop classifier)",
        "best_val_acc": _best("valid_acc", "val_acc", "val_accuracy"),
        "best_val_f1": _best("valid_f1", "val_f1", "val_macro_f1", "f1"),
        "source": "classifier_output/metrics_history.json",
    }


# ── Aggregation + outputs ───────────────────────────────────────────────────────

def _dedupe(records: list[dict]) -> list[dict]:
    """Prefer the outputs/benchmarks/<model>.json record if the same model appears twice."""
    by_name: dict[str, dict] = {}
    for r in records:
        by_name.setdefault(r["model_name"], r)
    return list(by_name.values())


def build_comparison() -> dict:
    detectors = _collect_benchmark_jsons()
    for extra in (_collect_yolo(), _collect_alt_selected()):
        if extra:
            detectors.append(extra)
    detectors = _dedupe(detectors)
    detectors.sort(key=lambda r: (r["map50"] is None, -(r["map50"] or 0)))
    classifier = _collect_classifier()
    return {"detectors": detectors, "classifier": classifier}


def write_markdown_table(comp: dict, out: Path) -> None:
    lines = ["# Model Benchmark — Crop Disease Detection\n",
             "## Detection models (Stage 2)\n",
             "| Model | Architecture | mAP@0.5 | Params (M) | Source |",
             "| ----- | ------------ | ------- | ---------- | ------ |"]
    for r in comp["detectors"]:
        pm = f"{r['num_params'] / 1e6:.1f}" if r.get("num_params") else "—"
        mp = f"{r['map50']:.3f}" if r.get("map50") is not None else "—"
        lines.append(f"| `{r['model_name']}` | {r['architecture']} | {mp} | {pm} | {r.get('source','')} |")
    if not comp["detectors"]:
        lines.append("| _(no trained detectors found — train a model first)_ | | | | |")
    c = comp.get("classifier")
    if c:
        lines += ["\n## Stage-1 classifier (reported separately — classification task)\n",
                  "| Model | Best val acc | Best val F1 |",
                  "| ----- | ------------ | ----------- |",
                  f"| {c['architecture']} | {c.get('best_val_acc','—')} | {c.get('best_val_f1','—')} |"]
    out.write_text("\n".join(lines) + "\n")


def make_figures(comp: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"  ⚠  matplotlib unavailable, skipping figures: {exc}")
        return

    dets = [d for d in comp["detectors"] if d.get("map50") is not None]
    if not dets:
        print("  (no detector metrics yet — figures skipped)")
        return

    plt.rcParams.update({"savefig.dpi": 300, "font.size": 10})
    names = [d["model_name"] for d in dets]
    maps  = [d["map50"] for d in dets]

    # mAP bar chart
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(dets)), 4.2))
    bars = ax.bar(names, maps, color="#3E6FE0")
    ax.set_ylabel("mAP@0.5"); ax.set_title("Cross-model mAP@0.5", fontweight="bold")
    ax.set_ylim(0, max(0.05, max(maps) * 1.2))
    for b, m in zip(bars, maps):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(out_dir / "fig_map_comparison.png", bbox_inches="tight"); plt.close(fig)

    # params vs mAP scatter
    pts = [(d["num_params"] / 1e6, d["map50"], d["model_name"])
           for d in dets if d.get("num_params")]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for x, y, n in pts:
            ax.scatter(x, y, s=70, color="#E8684A")
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Parameters (M)"); ax.set_ylabel("mAP@0.5")
        ax.set_title("Model size vs accuracy", fontweight="bold")
        fig.tight_layout(); fig.savefig(out_dir / "fig_params_vs_map.png", bbox_inches="tight")
        plt.close(fig)

    # per-class AP heatmap
    with_pc = [d for d in dets if d.get("per_class_ap")]
    if with_pc:
        mat = np.full((len(with_pc), len(CANONICAL_CLASSES)), np.nan)
        for i, d in enumerate(with_pc):
            # tolerate underscored class names (YOLO / RT-DETR) vs spaced (Faster R-CNN)
            pc = {k.replace("_", " "): v for k, v in d["per_class_ap"].items()}
            for j, cls in enumerate(CANONICAL_CLASSES):
                v = pc.get(cls)
                if isinstance(v, (int, float)):
                    mat[i, j] = v
        fig, ax = plt.subplots(figsize=(13, 1.1 + 0.6 * len(with_pc)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(with_pc))); ax.set_yticklabels([d["model_name"] for d in with_pc])
        ax.set_xticks(range(len(CANONICAL_CLASSES)))
        ax.set_xticklabels(CANONICAL_CLASSES, rotation=90, fontsize=7)
        ax.set_title("Per-class AP@0.5 (models × classes)", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="AP@0.5")
        fig.tight_layout(); fig.savefig(out_dir / "fig_per_class_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures written → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model benchmark aggregator")
    parser.add_argument("--out-dir", default=str(BENCH_DIR), help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="tables only, no figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp = build_comparison()
    print(f"Detectors found : {len(comp['detectors'])}")
    for d in comp["detectors"]:
        print(f"  • {d['model_name']:<34} mAP@0.5={d['map50']}")
    if comp.get("classifier"):
        print(f"Classifier      : {comp['classifier']['architecture']}")

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
    write_markdown_table(comp, out_dir / "comparison_table.md")
    print(f"Wrote {out_dir/'comparison.json'} and {out_dir/'comparison_table.md'}")

    if not args.no_figures:
        make_figures(comp, out_dir)


if __name__ == "__main__":
    main()
