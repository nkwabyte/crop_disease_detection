#!/usr/bin/env python3
"""
export_rtdetr.py — export the trained RT-DETR to ONNX + ExecuTorch (.pte) for the app.

RT-DETR inference is static-shape (fixed object queries, no NMS), so unlike the
two-stage ViT/Swin detectors it can export as a *full end-to-end* ExecuTorch model —
the app can consume the .pte directly, like it does for YOLO26.

Mirrors src/yolo/export_yolo.py so the mobile-integration story is identical.

Usage:  python -m src.rtdetr.export_rtdetr
"""

import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import RTDETR

from src.rtdetr.config import (
    RUNS_DIR, EXP_NAME, MODELS_DIR, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES,
)


def _best_checkpoint() -> Path:
    best = RUNS_DIR / EXP_NAME / "weights" / "best.pt"
    if best.exists():
        return best
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        print(f"⚠️ Using fallback checkpoint: {candidates[-1]}")
        return candidates[-1]
    raise FileNotFoundError("No trained RT-DETR checkpoint found. Train the model first.")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    best = _best_checkpoint()
    print(f"Loading RT-DETR from: {best}")
    model = RTDETR(str(best))

    assert len(model.names) == len(CLASS_NAMES), (
        f"Class count mismatch: model has {len(model.names)}, expected {len(CLASS_NAMES)}")

    # ── ONNX (universal fallback) ─────────────────────────────────────────────
    print("\nExporting to ONNX …")
    try:
        onnx_res = model.export(format="onnx", imgsz=IMG_SIZE, dynamic=False,
                                simplify=True, opset=17, half=False)
        dst_onnx = MODELS_DIR / "crop_disease_rtdetr.onnx"
        shutil.copy2(str(onnx_res), dst_onnx)
        print(f"✅ ONNX → {dst_onnx} ({dst_onnx.stat().st_size / 1_048_576:.1f} MB)")
    except Exception as exc:
        print(f"⚠️ ONNX export failed: {exc}")

    # ── ExecuTorch (.pte) — full model (RT-DETR is NMS-free / static-shape) ────
    print("\nExporting to ExecuTorch (.pte) …")
    dst_pte = MODELS_DIR / "crop_disease_rtdetr.pte"
    try:
        et_res = model.export(format="executorch", imgsz=IMG_SIZE, half=False)
        et_path = Path(str(et_res))
        pte_src = None
        if et_path.is_dir():
            files = list(et_path.rglob("*.pte"))
            pte_src = files[0] if files else None
        elif et_path.suffix == ".pte":
            pte_src = et_path
        if pte_src and pte_src.exists():
            shutil.copy2(pte_src, dst_pte)
            print(f"✅ ExecuTorch → {dst_pte} ({dst_pte.stat().st_size / 1_048_576:.1f} MB)")
        else:
            print(f"⚠️ Could not locate .pte in {et_path}")
    except Exception as exc:
        print(f"⚠️ ExecuTorch export failed ({exc}); ONNX remains the full-model fallback.")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "model_name": "crop_disease_rtdetr",
        "architecture": "RT-DETR-L (Ultralytics, transformer query head)",
        "task": "object_detection",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "input_size": IMG_SIZE,
        "input_channels": 3,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "crops_covered": ["Corn", "Pepper", "Tomato"],
        "notes": ("RT-DETR is anchor-free and NMS-free; apply conf_threshold at "
                  "inference before displaying results."),
        "android_integration": {
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "pte_file": "crop_disease_rtdetr.pte",
            "input_normalize": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]},
            "input_format": "NCHW_RGB",
        },
    }
    meta_path = MODELS_DIR / "rtdetr_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"\nMetadata written → {meta_path}")


if __name__ == "__main__":
    main()
