#!/usr/bin/env python3
"""
archive_weights.py — snapshot a trained model's weights into a versioned folder.

Training overwrites runs/<exp>/weights/best.pt in place, so a rerun that turns out
worse leaves nothing to fall back to. This copies the current weights into an
immutable, self-describing directory:

  weights/yolo26n_v1_20260804_map0.2904/
      best.pt  last.pt  results.csv  <exported>.pte
      MANIFEST.json   model, date, metrics, epochs, git commit, source paths

The version number auto-increments per model name, so archiving the same model
twice never clobbers the earlier snapshot. Restoring is a plain copy back.

Usage
-----
  python scripts/archive_weights.py --model yolo26n
  python scripts/archive_weights.py --model yolo26s --run runs/crop_disease_yolo26
  python scripts/archive_weights.py --list
  python scripts/archive_weights.py --restore yolo26n_v1_20260804_map0.2904
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "weights"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _best_metrics(results_csv: Path) -> dict:
    """Best row by mAP50, so the snapshot name carries the number that matters."""
    if not results_csv.exists():
        return {}
    import csv
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        return {}
    key = next((k for k in rows[0] if "mAP50(B)" in k), None)
    if not key:
        return {"epochs": len(rows)}
    best = max(rows, key=lambda r: float(r[key] or 0))
    k95 = next((k for k in rows[0] if "mAP50-95(B)" in k), None)
    return {
        "epochs": len(rows),
        "best_epoch": int(float(best.get("epoch", 0))),
        "mAP50": round(float(best[key]), 5),
        "mAP50_95": round(float(best[k95]), 5) if k95 else None,
    }


def do_list() -> int:
    if not ARCHIVE.exists() or not any(ARCHIVE.iterdir()):
        print("  no archived weights yet")
        return 0
    print(f"  {'snapshot':52} {'mAP50':>8}  {'epochs':>6}  commit")
    for d in sorted(ARCHIVE.iterdir()):
        if not d.is_dir():
            continue
        m = {}
        mf = d / "MANIFEST.json"
        if mf.exists():
            m = json.loads(mf.read_text())
        met = m.get("metrics", {})
        print(f"  {d.name:52} {str(met.get('mAP50', '-')):>8}  "
              f"{str(met.get('epochs', '-')):>6}  {m.get('git_commit', '-')}")
    return 0


def do_restore(name: str, run_dir: Path) -> int:
    src = ARCHIVE / name
    if not src.is_dir():
        print(f"  x no such snapshot: {src}")
        return 1
    dst = run_dir / "weights"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("best.pt", "last.pt"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
            print(f"  restored {f} -> {dst / f}")
    print("  Re-export the .pte if you intend to deploy this snapshot:")
    print("    bash scripts/export/yolo.sh")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="short model name, e.g. yolo26n")
    ap.add_argument("--run", default="runs/crop_disease_yolo26",
                    help="training run directory holding weights/ and results.csv")
    ap.add_argument("--pte", default="models/crop_disease_yolo26.pte",
                    help="exported artifact to include in the snapshot, if present")
    ap.add_argument("--list", action="store_true", help="list existing snapshots")
    ap.add_argument("--restore", metavar="SNAPSHOT", help="copy a snapshot back into --run")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / args.run
    if args.list:
        return do_list()
    if args.restore:
        return do_restore(args.restore, run_dir)
    if not args.model:
        ap.error("--model is required (or use --list / --restore)")

    weights = run_dir / "weights"
    best = weights / "best.pt"
    if not best.exists():
        print(f"  x no best.pt at {best}")
        return 1

    metrics = _best_metrics(run_dir / "results.csv")
    tag = f"_map{metrics['mAP50']:.4f}" if metrics.get("mAP50") is not None else ""
    date = datetime.fromtimestamp(best.stat().st_mtime).strftime("%Y%m%d")

    existing = [d.name for d in ARCHIVE.iterdir()] if ARCHIVE.exists() else []
    ver = 1 + max((int(m.group(1)) for d in existing
                   if (m := re.match(rf"{re.escape(args.model)}_v(\d+)_", d))), default=0)

    dest = ARCHIVE / f"{args.model}_v{ver}_{date}{tag}"
    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in ("best.pt", "last.pt"):
        if (weights / f).exists():
            shutil.copy2(weights / f, dest / f)
            copied.append(f)
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", dest / "results.csv")
        copied.append("results.csv")
    pte = PROJECT_ROOT / args.pte
    if pte.exists():
        shutil.copy2(pte, dest / pte.name)
        copied.append(pte.name)

    manifest = {
        "model": args.model,
        "version": ver,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "trained_at": datetime.fromtimestamp(best.stat().st_mtime).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_run": str(run_dir.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "files": copied,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"  archived -> weights/{dest.name}")
    for f in copied:
        print(f"      {f}")
    if metrics:
        print(f"  metrics: mAP50={metrics.get('mAP50')} "
              f"mAP50-95={metrics.get('mAP50_95')} "
              f"best_epoch={metrics.get('best_epoch')}/{metrics.get('epochs')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
