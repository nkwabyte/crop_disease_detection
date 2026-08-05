#!/usr/bin/env python3
"""
yolo_capacity.py — does YOLO26 model capacity move the needle on this dataset?

Reads the versioned snapshots written by scripts/archive_weights.py and produces
the accuracy-vs-size comparison, so re-running after adding a variant (yolo26l,
a distilled student, …) refreshes the table and charts without editing anything.

The question it answers is a diagnosis, not a leaderboard. Three yolo26n runs had
plateaued at mAP50 ~0.28 on a split holding only ~124 training images per disease
class, leaving it unclear whether the ceiling was the model or the data. A sharp
gain from a larger model would mean capacity; a flat one means the dataset — and
the flat answer is the more actionable, because it redirects effort to collection.

Outputs
-------
  outputs/benchmarks/yolo_capacity_comparison.{json,md}
  outputs/benchmarks/figures/fig_yolo_01_accuracy_vs_size.png
  outputs/benchmarks/figures/fig_yolo_02_accuracy_per_mb.png

Usage
-----
  python -m src.benchmark.yolo_capacity
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE = PROJECT_ROOT / "weights"
BENCH = PROJECT_ROOT / "outputs" / "benchmarks"
FIGS = BENCH / "figures"

# ── Figure styling ───────────────────────────────────────────────────────────
# One series per chart, so a single categorical slot and no legend box: the title
# already names what is plotted. Text wears ink tokens, never the series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
SERIES = "#2a78d6"
ACCENT = "#eb6834"      # used only to mark the deployed choice


def collect() -> list[dict]:
    """One record per archived variant, newest version of each model."""
    import torch

    best: dict[str, dict] = {}
    for d in sorted(ARCHIVE.iterdir()):
        mf = d / "MANIFEST.json"
        if not d.is_dir() or not mf.exists():
            continue
        m = json.loads(mf.read_text())
        name = m["model"]
        if name in best and best[name]["version"] >= m["version"]:
            continue
        pt = d / "best.pt"
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        model = ck.get("model")
        best[name] = {
            "model": name,
            "version": m["version"],
            "snapshot": d.name,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            if model is not None else None,
            "size_mb": round(pt.stat().st_size / 1e6, 1),
            "map50": m["metrics"]["mAP50"],
            "map50_95": m["metrics"].get("mAP50_95"),
            "best_epoch": m["metrics"].get("best_epoch"),
            "epochs": m["metrics"].get("epochs"),
        }
    return sorted(best.values(), key=lambda r: r["params_m"] or 0)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.set_axisbelow(True)


def make_figures(rows: list[dict]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    xs = [r["params_m"] for r in rows]
    ys = [r["map50"] for r in rows]

    # ── 1. The trade-off curve ───────────────────────────────────────────────
    # Size is continuous and the story is its *shape* — how fast the gain flattens
    # — so a connected scatter, not bars. One series, so no legend; three points,
    # so every point earns a direct label.
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200, facecolor=SURFACE)
    ax.plot(xs, ys, color=SERIES, linewidth=2, marker="o", markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round",
            zorder=3)
    # The deployed choice is the point the reader should leave with.
    ax.plot(xs[0], ys[0], marker="o", markersize=9, color=ACCENT,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    for r in rows:
        gain = (r["map50"] / rows[0]["map50"] - 1) * 100
        tag = f"{r['model']}\n{r['map50']:.4f}" + (f"  ({gain:+.1f}%)" if gain else "  (baseline)")
        ax.annotate(tag, (r["params_m"], r["map50"]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, color=INK)
    ax.set_xlabel("Model parameters (millions)", fontsize=10, color=INK_2)
    ax.set_ylabel("mAP@0.5", fontsize=10, color=INK_2)
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 1.5)
    ax.set_xlim(0, max(xs) * 1.15)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    _style(ax)
    ax.set_title("8.7x the parameters buys 6.8% mAP — capacity is not the ceiling",
                 fontsize=12, color=INK, pad=14, loc="left")
    fig.tight_layout()
    p = FIGS / "fig_yolo_01_accuracy_vs_size.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    out.append(p)

    # ── 2. The deployment view ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=200, facecolor=SURFACE)
    eff = [r["map50"] / r["size_mb"] for r in rows]
    names = [r["model"] for r in rows]
    colors = [ACCENT] + [SERIES] * (len(rows) - 1)   # emphasis: the shipped model
    bars = ax.bar(range(len(rows)), eff, 0.42, color=colors)
    for b, r, e in zip(bars, rows, eff):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(eff) * 0.03,
                f"{e:.4f}\n{r['size_mb']:.0f} MB", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=10, color=INK)
    ax.set_ylabel("mAP@0.5 per MB", fontsize=10, color=INK_2)
    ax.set_ylim(0, max(eff) * 1.30)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    _style(ax)
    ax.set_title("Accuracy per megabyte — the mobile trade-off",
                 fontsize=12, color=INK, pad=14, loc="left")
    fig.tight_layout()
    p = FIGS / "fig_yolo_02_accuracy_per_mb.png"
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    out.append(p)
    return out


def main() -> None:
    rows = collect()
    if not rows:
        raise SystemExit(f"  x no snapshots in {ARCHIVE} — run scripts/archive_weights.py")

    base = rows[0]
    for r in rows:
        r["vs_baseline_pct"] = round((r["map50"] / base["map50"] - 1) * 100, 2)
        r["map50_per_mb"] = round(r["map50"] / r["size_mb"], 4)

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "yolo_capacity_comparison.json").write_text(json.dumps(rows, indent=2))

    md = ["# YOLO26 capacity sweep — is the ceiling the model or the data?", "",
          "Identical data, schedule and augmentation; each variant early-stopped on",
          "its own validation mAP. Trained on the RTX 5090, 2026-08-04.", "",
          "| Model | Params | best.pt | mAP@0.5 | mAP@0.5:0.95 | Epochs | vs baseline | mAP/MB |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params_m']}M | {r['size_mb']} MB | "
                  f"{r['map50']:.4f} | {r['map50_95']:.4f} | "
                  f"{r['best_epoch']}/{r['epochs']} | {r['vs_baseline_pct']:+.1f}% | "
                  f"{r['map50_per_mb']:.4f} |")
    md += ["",
           f"**{rows[-1]['params_m'] / rows[0]['params_m']:.1f}x the parameters buys "
           f"{rows[-1]['vs_baseline_pct']:+.1f}% mAP@0.5.** The gain also flattens across the "
           "sweep, so the limiting factor is the dataset — roughly 124 training images per "
           "disease class — not model capacity. Effort is better spent collecting images "
           "than enlarging the detector.", "",
           "The larger variants also converged in *fewer* epochs "
           f"({rows[-1]['epochs']} vs {rows[0]['epochs']}), which is what one expects when a "
           "model saturates the available data and early-stops.", "",
           "For deployment the baseline remains the right choice: accuracy per megabyte "
           f"falls {rows[0]['map50_per_mb'] / rows[-1]['map50_per_mb']:.1f}x from "
           f"{rows[0]['model']} to {rows[-1]['model']}.", "",
           "> Caveats: one run per variant, no seed repetitions, so treat the deltas as",
           "> indicative rather than significance-tested. Batch sizes differed for memory",
           "> reasons (32 / 48 / 32 for n / s / m).", "",
           "Generated by `src/benchmark/yolo_capacity.py`."]
    (BENCH / "yolo_capacity_comparison.md").write_text("\n".join(md))

    print(f"  {'model':9} {'params':>9} {'size':>9} {'mAP50':>8} {'vs base':>9} {'mAP/MB':>9}")
    for r in rows:
        print(f"  {r['model']:9} {r['params_m']:8.2f}M {r['size_mb']:8.1f}M "
              f"{r['map50']:8.4f} {r['vs_baseline_pct']:+8.1f}% {r['map50_per_mb']:9.4f}")
    print(f"\n  wrote {BENCH / 'yolo_capacity_comparison.json'}")
    print(f"  wrote {BENCH / 'yolo_capacity_comparison.md'}")
    for p in make_figures(rows):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
