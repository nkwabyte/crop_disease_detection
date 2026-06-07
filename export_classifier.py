import os
import torch
import yaml
from pathlib import Path
from datetime import datetime
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from train_classifier import build_model, NUM_CLASSES, IMG_SIZE, CROP_CLASSES

def main():
    project_root = Path.cwd()
    output_dir = project_root / "outputs" / "classifier_output"
    best_pth = output_dir / "best.pth"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    pte_out = models_dir / "crop_classifier.pte"
    meta_path = models_dir / "classifier_metadata.yaml"

    if not best_pth.exists():
        print(f"❌ Could not find classifier checkpoint at {best_pth}")
        return

    print(f"Loading classifier from {best_pth}")
    device = torch.device("cpu")
    model = build_model(NUM_CLASSES).to(device)
    ckpt = torch.load(best_pth, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Tracing model...")
    example_inputs = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)
    # Trace the model using torch.export
    try:
        ep = torch.export.export(model, example_inputs)
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
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
    print(f"\n✅ ExecuTorch export complete")
    print(f"   └─ {pte_out} ({size_mb:.1f} MB)")

    print("Writing metadata YAML...")
    metadata = {
        "model_name": "crop_classifier_effnet_b2",
        "architecture": "EfficientNet-B2",
        "task": "image_classification",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "input_size": IMG_SIZE,
        "input_channels": 3,
        "num_classes": NUM_CLASSES,
        "class_names": CROP_CLASSES,
        "conf_threshold": 0.55,
        "android_integration": {
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "pte_file": "crop_classifier.pte",
            "input_normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "input_format": "NCHW_RGB",
        }
    }
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    print(f"Metadata written → {meta_path}")

if __name__ == "__main__":
    main()
