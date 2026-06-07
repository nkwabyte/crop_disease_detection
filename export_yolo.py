import shutil
import yaml
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

# ─── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
RUNS_DIR     = PROJECT_ROOT / "runs"
EXP_NAME     = "crop_disease_yolo26"

MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Try to find the best checkpoint
BEST_PT = RUNS_DIR / EXP_NAME / "weights" / "best.pt"
if not BEST_PT.exists():
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        BEST_PT = candidates[-1]
        print(f"⚠️ Using fallback checkpoint: {BEST_PT}")
    else:
        raise FileNotFoundError("No trained model found. Please train the YOLO model first.")

IMG_SIZE = 640
CONF_THRESHOLD = 0.50

CLASS_NAMES = [
    'Corn_Cercospora_Leaf_Spot', 'Corn_Common_Rust', 'Corn_Healthy',
    'Corn_Northern_Leaf_Blight', 'Corn_Streak',
    'Pepper_Bacterial_Spot', 'Pepper_Cercospora', 'Pepper_Early_Blight',
    'Pepper_Fusarium', 'Pepper_Healthy', 'Pepper_Late_Blight',
    'Pepper_Leaf_Blight', 'Pepper_Leaf_Curl', 'Pepper_Leaf_Mosaic',
    'Pepper_Septoria',
    'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Fusarium',
    'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Leaf_Curl',
    'Tomato_Mosaic', 'Tomato_Septoria'
]

def main():
    print(f"Loading YOLO model from: {BEST_PT}")
    model = YOLO(str(BEST_PT))
    
    assert len(model.names) == len(CLASS_NAMES), (
        f"Class count mismatch: model has {len(model.names)}, expected {len(CLASS_NAMES)}"
    )
    
    # ─── Export to ONNX ───────────────────────────────────────────────────────
    print("\nExporting to ONNX...")
    onnx_result = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True,
        opset=17,
        half=False,
    )
    src_onnx = Path(str(onnx_result))
    dst_onnx = MODELS_DIR / "crop_disease_yolo26.onnx"
    shutil.copy2(src_onnx, dst_onnx)
    print(f"✅ ONNX export complete → {dst_onnx} ({dst_onnx.stat().st_size / 1_048_576:.1f} MB)")

    # ─── Export to ExecuTorch ─────────────────────────────────────────────────
    print("\nExporting to ExecuTorch (.pte) for Android/Kotlin integration...")
    et_result = model.export(
        format="executorch",
        imgsz=IMG_SIZE,
        half=False,
    )
    
    et_path = Path(str(et_result))
    pte_src = None
    if et_path.is_dir():
        pte_files = list(et_path.rglob("*.pte"))
        pte_src = pte_files[0] if pte_files else None
    else:
        pte_src = et_path if et_path.suffix == ".pte" else None

    dst_pte = MODELS_DIR / "crop_disease_yolo26.pte"
    if pte_src and pte_src.exists():
        shutil.copy2(pte_src, dst_pte)
        print(f"✅ ExecuTorch export complete → {dst_pte} ({dst_pte.stat().st_size / 1_048_576:.1f} MB)")
    else:
        print(f"⚠️ Could not locate .pte file. Check: {et_path}")

    # ─── Write Metadata YAML ──────────────────────────────────────────────────
    metadata = {
        "model_name": "crop_disease_yolo26",
        "architecture": "YOLO26",
        "task": "object_detection",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "input_size": IMG_SIZE,
        "input_channels": 3,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": 0.45,
        "crops_covered": ["Corn", "Pepper", "Tomato"],
        "notes": (
            "Trained on Ghana Crop Disease Challenge v2. "
            "Apply conf_threshold at inference time before displaying results."
        ),
        "android_integration": {
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "pte_file": "crop_disease_yolo26.pte",
            "input_normalize": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]},
            "input_format": "NCHW_RGB",
        }
    }
    
    meta_path = MODELS_DIR / "yolo_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"\nMetadata written → {meta_path}")

if __name__ == "__main__":
    main()
