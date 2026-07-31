#!/usr/bin/env python3
"""
latency.py — size + inference-latency benchmark for the exported model artifacts.

Loads every exported `.pte` (ExecuTorch / XNNPACK) and `.onnx` (ONNX Runtime) it can
find and reports file size and per-image CPU latency. This is the deployment-cost axis
that complements the accuracy benchmark (`compare_models.py`): it shows which model is
actually cheap enough to ship to farmers.

Measured on THIS host's CPU. Numbers indicate the *ordering* and rough magnitude you
will see on a mid-range mobile CPU (both paths use the same XNNPACK kernels the phone
uses), not exact on-device timings — for those, profile the .pte on the target device.

Usage
-----
  python -m src.benchmark.latency
  python -m src.benchmark.latency --runs 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS      = PROJECT_ROOT / "outputs"
MODELS_DIR   = PROJECT_ROOT / "models"
BENCH_DIR    = OUTPUTS / "benchmarks"


def _input_size_for(path: Path) -> int:
    """Infer the square input size: classifier = 260, detectors = 640, with a metadata
    override when a sibling *metadata*.yaml declares input_size."""
    try:
        import yaml
        for meta in path.parent.glob("*metadata*.yaml"):
            d = yaml.safe_load(meta.read_text())
            if isinstance(d, dict) and d.get("input_size"):
                # only trust metadata that plausibly matches this artifact family
                if ("classifier" in path.name) == ("classifier" in meta.name.lower() or
                                                    d.get("task") == "image_classification"):
                    return int(d["input_size"])
    except Exception:
        pass
    return 260 if "classifier" in path.name else 640


def _discover() -> list[Path]:
    arts: list[Path] = []
    for base in [MODELS_DIR, *sorted(OUTPUTS.glob("*/models"))]:
        if base.exists():
            arts += sorted(base.glob("*.pte"))
            arts += sorted(base.glob("*.onnx"))
    return arts


def _bench_pte(path: Path, shape, runs: int, warmup: int) -> list[float]:
    import executorch.kernels.quantized  # noqa: F401  (in case of quantized .pte)
    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    m = _load_for_executorch(str(path))
    x = torch.randn(*shape)
    for _ in range(warmup):
        m.forward((x,))
    out = []
    for _ in range(runs):
        t = time.perf_counter(); m.forward((x,)); out.append((time.perf_counter() - t) * 1000)
    return out


def _bench_onnx(path: Path, shape, runs: int, warmup: int) -> list[float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    dims = [d if isinstance(d, int) and d > 0 else s for d, s in zip(ishape, shape)]
    x = np.random.randn(*dims).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {iname: x})
    out = []
    for _ in range(runs):
        t = time.perf_counter(); sess.run(None, {iname: x}); out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Size + latency benchmark of exported artifacts")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out-dir", default=str(BENCH_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _discover()
    if not artifacts:
        print("No .pte / .onnx artifacts found. Export a model first."); return

    rows = []
    for path in artifacts:
        size_mb = path.stat().st_size / 1_048_576
        s = _input_size_for(path)
        shape = (1, 3, s, s)
        fmt = path.suffix.lstrip(".")
        try:
            times = _bench_pte(path, shape, args.runs, args.warmup) if fmt == "pte" \
                else _bench_onnx(path, shape, args.runs, args.warmup)
            med, p90 = statistics.median(times), np.percentile(times, 90)
            print(f"  {path.name:<38} {fmt:<4} {size_mb:6.1f} MB  {med:7.1f} ms (median)")
            rows.append({"name": path.name, "dir": str(path.parent.relative_to(PROJECT_ROOT)),
                         "format": fmt, "input": f"1x3x{s}x{s}", "size_mb": round(size_mb, 2),
                         "latency_ms_median": round(med, 2), "latency_ms_p90": round(float(p90), 2)})
        except Exception as exc:
            print(f"  {path.name:<38} {fmt:<4} {size_mb:6.1f} MB  [WARN] {type(exc).__name__}: {str(exc)[:50]}")
            rows.append({"name": path.name, "dir": str(path.parent.relative_to(PROJECT_ROOT)),
                         "format": fmt, "input": f"1x3x{s}x{s}", "size_mb": round(size_mb, 2),
                         "latency_ms_median": None, "latency_ms_p90": None,
                         "error": type(exc).__name__})

    payload = {"host_note": "measured on host CPU (ExecuTorch XNNPACK / ONNX Runtime); "
                            "indicative of mobile-CPU ordering, not exact device numbers",
               "runs": args.runs, "artifacts": rows}
    (out_dir / "latency.json").write_text(json.dumps(payload, indent=2))
    _write_table(rows, out_dir / "latency_table.md")
    _make_figure(rows, out_dir)
    print(f"\nWrote {out_dir/'latency.json'}, latency_table.md, fig_latency.png")


def _write_table(rows, out: Path) -> None:
    lines = ["# Model Size + Latency (host CPU — indicative of mobile ordering)\n",
             "| Artifact | Format | Input | Size (MB) | Latency ms (median) | p90 |",
             "| -------- | ------ | ----- | --------- | ------------------- | --- |"]
    for r in sorted(rows, key=lambda x: (x["latency_ms_median"] is None, x["latency_ms_median"] or 0)):
        lat = "—" if r["latency_ms_median"] is None else f"{r['latency_ms_median']:.1f}"
        p90 = "—" if r["latency_ms_p90"] is None else f"{r['latency_ms_p90']:.1f}"
        lines.append(f"| `{r['name']}` | {r['format']} | {r['input']} | {r['size_mb']:.1f} | {lat} | {p90} |")
    out.write_text("\n".join(lines) + "\n")


def _make_figure(rows, out_dir: Path) -> None:
    ok = [r for r in rows if r["latency_ms_median"] is not None]
    if not ok:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ok = sorted(ok, key=lambda x: x["latency_ms_median"])
    names = [r["name"].replace("crop_disease_", "").replace("crop_", "") for r in ok]
    plt.rcParams.update({"savefig.dpi": 300, "font.size": 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.5 + 0.5 * len(ok)))
    yp = range(len(ok))
    a1.barh(list(yp), [r["size_mb"] for r in ok], color="#5B8FF9")
    a1.set_yticks(list(yp)); a1.set_yticklabels(names, fontsize=8)
    a1.invert_yaxis(); a1.set_xlabel("Size (MB)"); a1.set_title("Artifact size", fontweight="bold")
    a2.barh(list(yp), [r["latency_ms_median"] for r in ok], color="#E8684A")
    a2.set_yticks(list(yp)); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel("Latency (ms/img, median)"); a2.set_title("CPU latency", fontweight="bold")
    fig.suptitle("Exported model size + latency (host CPU)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_latency.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
