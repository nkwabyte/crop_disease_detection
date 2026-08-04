#!/usr/bin/env python3
"""
quantize.py — post-training INT8 quantization to ExecuTorch .pte (XNNPACK backend).

Produces ~4× smaller `.pte` artifacts for the mobile app via PT2E static quantization.
The core `quantize_to_pte()` helper is model-agnostic (it takes a plain nn.Module); the
CLI quantizes each trained model it can find, each step guarded so a missing checkpoint
or deleted model package just skips.

Every quantized model is FIDELITY-CHECKED against its fp32 source (top-1 agreement for
the classifier, feature cosine-similarity for detector backbones). If fidelity is low
the tool warns loudly rather than shipping a silently-broken model.

Note on architectures: static PTQ preserves accuracy on CNN backbones (ResNet family:
100% top-1 agreement in testing) but COLLAPSES EfficientNet-B2 (SiLU + squeeze-excite
are PTQ-fragile — the Stage-1 classifier flattens to a constant output). For
EfficientNet, use quantization-aware training (QAT) or keep the fp32 model on device.
Size wins are real (~4×); on Apple Silicon the speed is roughly flat (the CPU-latency
win shows on mid-range Android — see `latency.py`).

Requires the `flatc` FlatBuffers compiler on PATH (XNNPACK subgraph serialization):
    brew install flatbuffers        # macOS
    # or use the copy bundled at .venv/bin/flatc

Usage
-----
  python -m src.benchmark.quantize                 # quantize everything trained
  python -m src.benchmark.quantize --only classifier
  python -m src.benchmark.quantize --calib 64      # calibration image count
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

import executorch


def _ensure_flatc_on_path() -> None:
    """XNNPACK subgraph serialization shells out to `flatc`. If it isn't on PATH, add
    the copy ExecuTorch bundles so quantization works without a separate install.
    (executorch is a namespace package, so use __path__, not __file__.)"""
    if shutil.which("flatc"):
        return
    for base in list(getattr(executorch, "__path__", [])):
        for cand in (Path(base) / "data" / "bin", Path(base).parent / "bin"):
            if (cand / "flatc").exists():
                os.environ["PATH"] = f"{cand}{os.pathsep}{os.environ.get('PATH', '')}"
                return


_ensure_flatc_on_path()

# Registers the quantized_decomposed out-variant kernels needed by to_executorch().
import executorch.kernels.quantized  # noqa: E402,F401
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import _load_for_executorch

FIDELITY_MIN = 0.90   # below this, warn that the architecture needs QAT, not PTQ

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Core helper (model-agnostic) ────────────────────────────────────────────────

def quantize_to_pte(module: torch.nn.Module, example_inputs: tuple,
                    calib_inputs, out_path: Path, per_channel: bool = True) -> int:
    """Static-quantize `module` to INT8 and lower to an XNNPACK ExecuTorch .pte.

    calib_inputs: iterable of args-tuples fed through the prepared model to collect
    activation statistics. Returns the .pte size in bytes.
    """
    module = module.eval()
    captured = export(module, example_inputs).module()

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel))
    prepared = prepare_pt2e(captured, quantizer)

    n = 0
    for ci in calib_inputs:
        prepared(*ci)
        n += 1
    if n == 0:  # never calibrated → feed the example so ranges are defined
        prepared(*example_inputs)

    converted = convert_pt2e(prepared)
    ep = export(converted, example_inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    et = edge.to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et.buffer)
    return len(et.buffer)


# ── Calibration image loading ───────────────────────────────────────────────────

def _preprocess(img_path: Path, size: int) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms.v2.functional as F
    img = Image.open(img_path).convert("RGB")
    t = F.to_image(img)
    t = F.resize(t, [size, size], antialias=True)
    t = F.to_dtype(t, torch.float32, scale=True)          # [0,1]
    t = (t.unsqueeze(0) - _IMAGENET_MEAN) / _IMAGENET_STD  # normalized [1,3,S,S]
    return t


def _calib_inputs(image_dir: Path, size: int, n: int, pattern: str = "*.jpg"):
    paths = sorted(image_dir.glob(pattern))[:n]
    if not paths:
        return []
    return [(_preprocess(p, size),) for p in paths]


def _report(tag: str, fp32_ref: Path | None, int8_path: Path, int8_bytes: int) -> None:
    int8_mb = int8_bytes / 1_048_576
    if fp32_ref and fp32_ref.exists():
        fp32_mb = fp32_ref.stat().st_size / 1_048_576
        ratio = fp32_mb / max(int8_mb, 1e-6)
        print(f"  [OK] {tag}: {fp32_mb:.1f} MB (fp32) → {int8_mb:.1f} MB (int8)  [{ratio:.1f}× smaller]")
    else:
        print(f"  [OK] {tag}: {int8_mb:.1f} MB (int8)  → {int8_path}")


# ── Fidelity check (fp32 vs int8) — never ship a silently-broken quantized model ─

def _warn_if_low(tag: str, score: float, metric: str) -> None:
    if score != score:  # nan
        return
    if score < FIDELITY_MIN:
        print(f"     [WARN]  {tag} fidelity {metric}={score:.2f} < {FIDELITY_MIN:.2f} — static PTQ "
              f"degraded this architecture; use QAT or keep the fp32 model on device.")
    else:
        print(f"     fidelity {metric}={score:.2f}  (int8 matches fp32)")


def _classifier_fidelity(model, int8_path: Path, image_dir: Path, size: int, n: int = 30) -> float:
    rt = _load_for_executorch(str(int8_path))
    paths = sorted(image_dir.glob("*.jpg"))[-n:]
    if not paths:
        return float("nan")
    agree = 0
    for p in paths:
        x = _preprocess(p, size)
        with torch.no_grad():
            a = int(model(x).argmax())
        agree += (int(rt.forward((x,))[0].argmax()) == a)
    return agree / len(paths)


def _backbone_fidelity(backbone, int8_path: Path, image_dir: Path, size: int, n: int = 6) -> float:
    rt = _load_for_executorch(str(int8_path))
    paths = sorted(image_dir.glob("*.jpg"))[-n:]
    if not paths:
        return float("nan")
    sims = []
    for p in paths:
        x = _preprocess(p, size)
        with torch.no_grad():
            ref = backbone(x)
        ref_t = ref if isinstance(ref, torch.Tensor) else ref[0]
        q = rt.forward((x,))[0]
        sims.append(float(torch.nn.functional.cosine_similarity(
            ref_t.flatten(), q.flatten(), dim=0)))
    return sum(sims) / len(sims)


# ── Per-model commands (each guarded by the caller) ─────────────────────────────

def quantize_classifier(n_calib: int) -> None:
    from src.classifier.train_classifier import build_model
    from src.classifier.config import NUM_CLASSES, IMG_SIZE, OUTPUT_DIR, DATASET_DIR
    ckpt_path = OUTPUT_DIR / "best.pth"
    if not ckpt_path.exists():
        print("  – classifier: no best.pth, skipping"); return
    model = build_model(NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    calib = _calib_inputs(DATASET_DIR / "train", IMG_SIZE, n_calib)
    ex = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)
    out = MODELS_DIR / "crop_classifier_int8.pte"
    size = quantize_to_pte(model, ex, calib, out)
    _report("classifier", MODELS_DIR / "crop_classifier.pte", out, size)
    _warn_if_low("classifier", _classifier_fidelity(model.eval(), out, DATASET_DIR / "train", IMG_SIZE),
                 "top-1-agreement")


def _quantize_detector_backbone(pkg: str, out_name: str, tag: str, n_calib: int) -> None:
    import importlib
    mod = importlib.import_module(f"src.{pkg}.train_{pkg}")
    cfg = importlib.import_module(f"src.{pkg}.config")
    ckpt_path = cfg.CKPT_DIR / "best.pth"
    if not ckpt_path.exists():
        print(f"  – {tag}: no best.pth (train on the GPU server first), skipping"); return
    model = mod.build_model(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    backbone = mod._BackboneWrapper(model.backbone).eval()
    calib = _calib_inputs(cfg.PROJECT_ROOT / "data" / "yolo" / "train" / "images",
                          cfg.IMG_SIZE, n_calib)
    ex = (torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE),)
    out = cfg.MODELS_DIR / out_name
    size = quantize_to_pte(backbone, ex, calib, out)
    fp32 = cfg.MODELS_DIR / out_name.replace("_int8", "")
    _report(tag, fp32, out, size)
    _warn_if_low(tag, _backbone_fidelity(backbone, out,
                 cfg.PROJECT_ROOT / "data" / "yolo" / "train" / "images", cfg.IMG_SIZE),
                 "feature-cosine")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8-quantize trained models to ExecuTorch .pte")
    parser.add_argument("--only", choices=["classifier", "vit", "swin"], default=None)
    parser.add_argument("--calib", type=int, default=64, help="calibration image count")
    args = parser.parse_args()

    jobs = {
        "classifier": lambda: quantize_classifier(args.calib),
        "vit":  lambda: _quantize_detector_backbone("vit", "crop_disease_vit_backbone_int8.pte",
                                                    "vit-backbone", args.calib),
        "swin": lambda: _quantize_detector_backbone("swin", "crop_disease_swin_backbone_int8.pte",
                                                    "swin-backbone", args.calib),
    }
    selected = [args.only] if args.only else list(jobs)
    print(f"Quantizing: {', '.join(selected)}  (calib={args.calib} images)\n")
    for name in selected:
        try:
            jobs[name]()
        except Exception as exc:
            print(f"  [WARN]  {name}: {type(exc).__name__}: {exc}")
    print("\nDone. Quantized .pte files sit next to their fp32 counterparts.")


if __name__ == "__main__":
    main()
