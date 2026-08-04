"""Export the crop classifier to ExecuTorch (.pte) for the Android app.

Run this on a machine that has `flatc` (ExecuTorch shells out to it when
serializing the XNNPACK payload). macOS: `brew install flatbuffers`.

  python -m src.classifier.export_classifier                 # 3-class (shipped)
  python -m src.classifier.export_classifier --variant ood   # 4-class + "Other"
"""
import argparse
import os
import torch
import yaml
from pathlib import Path
from datetime import datetime
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from src.classifier.train_classifier import VARIANTS, build_model
from src.classifier.config import PROJECT_ROOT, OUTPUT_DIR, NUM_CLASSES, IMG_SIZE, CROP_CLASSES

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="base",
                    help="'base' = the 3-class model the app consumes; "
                         "'ood' = the 4-class model with a learned 'Other' class")
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    class_names = list(variant["classes"])
    num_classes = len(class_names)

    project_root = PROJECT_ROOT
    output_dir = variant["dir"]
    best_pth = output_dir / "best.pth"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # The base variant keeps the filenames the Android app already loads; the
    # OOD variant is written alongside it so both can ship and be A/B'd.
    stem = "crop_classifier" if args.variant == "base" else "crop_classifier_ood"
    pte_out = models_dir / f"{stem}.pte"
    meta_path = models_dir / f"{stem}_metadata.yaml"
    if args.variant == "base":
        meta_path = models_dir / "classifier_metadata.yaml"

    if not best_pth.exists():
        print(f"[ERROR] Could not find classifier checkpoint at {best_pth}")
        return

    print(f"Variant: {args.variant}  ({num_classes} classes: {', '.join(class_names)})")
    print(f"Loading classifier from {best_pth}")
    device = torch.device("cpu")
    model = build_model(num_classes).to(device)
    ckpt = torch.load(best_pth, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Tracing model...")
    example_inputs = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)
    # Trace the model using torch.export
    try:
        ep = torch.export.export(model, example_inputs)
    except Exception as e:
        print(f"[ERROR] Tracing failed: {e}")
        return

    print("Lowering to edge dialect...")
    edge_prog = to_edge(ep)

    print("Partitioning for XNNPACK backend...")
    edge_prog = edge_prog.to_backend(XnnpackPartitioner())

    print("Exporting to ExecuTorch (.pte)...")
    exec_prog = edge_prog.to_executorch()
    
    with open(pte_out, "wb") as f:
        f.write(exec_prog.buffer)
    
    size_mb = pte_out.stat().st_size / 1_048_576
    print(f"\n[OK] ExecuTorch export complete")
    print(f"   └─ {pte_out} ({size_mb:.1f} MB)")

    print("Writing metadata YAML...")
    metadata = {
        "model_name": f"crop_classifier_effnet_b2_{args.variant}",
        "architecture": "EfficientNet-B2",
        "task": "image_classification",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "input_size": IMG_SIZE,
        "input_channels": 3,
        "num_classes": num_classes,
        "class_names": class_names,
        # The 4-class model rejects via argmax == "Other", so a confidence
        # threshold is not part of its contract.
        "conf_threshold": 0.55 if args.variant == "base" else None,
        "rejection": ("softmax confidence below conf_threshold -> unknown"
                      if args.variant == "base"
                      else "argmax == 'Other' -> unknown"),
        "android_integration": {
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "pte_file": f"{stem}.pte",
            "input_normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "input_format": "NCHW_RGB",
        }
    }
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    print(f"Metadata written → {meta_path}")

if __name__ == "__main__":
    main()
