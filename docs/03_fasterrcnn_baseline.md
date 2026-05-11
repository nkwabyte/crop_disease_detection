# Faster RCNN v2 Baseline Detector

**Script:** `train_fasterrcnn.py`  
**Output directory:** `fasterrcnn_output/`  
**Role:** Two-stage detection baseline; reference point for SE-FPN ablation

---

## 1. Overview

This script implements a clean, production-quality fine-tuning pipeline for `fasterrcnn_resnet50_fpn_v2` — the improved version of the classic Faster RCNN detector from torchvision. It is the reference baseline against which the SE-FPN final model is compared.

It supersedes the original `detector-torch.ipynb` notebook with the following corrections and additions:

| Issue in notebook | Fix in `train_fasterrcnn.py` |
|---|---|
| `fasterrcnn_resnet50_fpn` (v1, COCO AP 37.0) | `fasterrcnn_resnet50_fpn_v2` (COCO AP 46.7) |
| `num_classes=23` — missing background class | `num_classes=24` — 23 diseases (1–23) + background (0) |
| Labels decremented by 1 (bug) | Labels kept as 1–23 (Faster RCNN requires ≥ 1) |
| 4 fixed epochs, no resume | 30 epochs, early stopping, full resume support |
| No augmentation | `torchvision.transforms.v2` joint image + bbox augmentation |
| Hard-coded `workers=4` (crashes MPS) | `workers=0` on MPS, `workers=8` on CUDA |
| `pin_memory` not device-aware | `pin_memory=True` on CUDA, `False` on MPS |
| No mixed-precision training | CUDA: `torch.autocast` + `GradScaler`; MPS: float32 |
| No OOD guard | 200 hard-negative images with empty annotation targets |
| TorchScript export only | ExecuTorch + ONNX + TorchScript mobile (`.ptl`) |
| No publication figures | 11 publication figures |

---

## 2. Architecture

| Component | Detail |
|---|---|
| Detector | Faster RCNN (two-stage: RPN + RoI Head) |
| Backbone | ResNet-50 with FPN (Feature Pyramid Network) |
| Backbone version | `fasterrcnn_resnet50_fpn_v2` — improved COCO pretraining |
| FPN output levels | P2–P6 (5 levels) |
| RPN | Sliding-window anchor-based region proposal network |
| RoI pooling | RoI Align |
| Classifier head | Two FC layers (1024-d hidden) → 24-class softmax + bbox regression |
| num_classes | 24 (0 = background, 1–23 = disease classes) |
| Input resolution | 640 × 640 px |
| Parameters (approx.) | ~43 M |

### Backbone weight initialisation

Pre-trained on COCO using `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT`. The backbone is frozen for the first 5 epochs to prevent overwriting its high-quality ImageNet/COCO feature representations before the detection head has stabilised.

---

## 3. Training Pipeline

### Step 1 — Hard negatives

Downloads 200 diverse non-crop images from `picsum.photos` (seeded with 42 for reproducibility) into `data/negatives/`. Each image is given an empty annotation target:
```python
{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
```
The RPN assigns all its proposals to background on these images, training the model to suppress detections on OOD scenes.

### Step 2 — Training

Fine-tunes the pre-trained model for up to 30 epochs. Separate parameter groups are used:
- **Backbone params:** LR = `LR₀ × 0.1` — slower updates preserve pre-trained features
- **Head and RPN params:** LR = `LR₀` — faster updates for the new task-specific layers

### Step 3 — Figures

Generates 11 publication figures (see Section 7).

### Step 4 — Export

Exports the best checkpoint to three mobile formats (see Section 8).

---

## 4. Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Architecture | FasterRCNN-ResNet50-FPN-v2 | V2 COCO AP = 46.7 |
| num_classes | 24 | 23 diseases + background (0) |
| Image size | 640 × 640 | |
| Batch size | 4 | Safe on 24 GB unified memory |
| Effective batch | 4 | No gradient accumulation in baseline |
| Epochs | 30 | Pre-trained backbone converges quickly |
| Early stopping | patience = 8 | Evals every `EVAL_EVERY` epochs |
| Optimizer | SGD, momentum = 0.9 | Standard for Faster RCNN fine-tuning |
| LR (head) | 5e-3 | Peak after warmup |
| LR (backbone) | 5e-4 | 10× lower to preserve pretrained features |
| LR schedule | Linear warmup (3 ep) → cosine decay | |
| Backbone freeze | First 5 epochs | |
| Gradient clip | max norm = 10.0 | |
| Augmentation | HFlip + ColorJitter + GaussianBlur | `torchvision.transforms.v2` |
| Hard negatives | 200 images | |
| Eval frequency | Every 5 epochs | VOC mAP@0.5 |
| AMP | CUDA only | `torch.autocast(cuda, fp16)` + `GradScaler` |
| Workers | 0 (MPS) / 8 (CUDA) | |
| pin_memory | False (MPS) / True (CUDA) | |
| cudnn.benchmark | True (CUDA only) | Auto-tunes cuDNN for fixed 640×640 input |

### Augmentation pipeline (train split)

| Transform | Parameters |
|---|---|
| RandomHorizontalFlip | p = 0.5 |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 |
| RandomGrayscale | p = 0.1 |
| GaussianBlur | kernel 5×5, σ=[0.1, 2.0] |
| RandomPhotometricDistort | p = 0.3 |

All transforms use `torchvision.transforms.v2` which applies transforms jointly to the image and its bounding boxes, preventing label misalignment.

---

## 5. Mixed-Precision (AMP) on CUDA

On a CUDA GPU server, the training loop runs in float16 via `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss_dict = model(images, targets)
    losses    = sum(loss_dict.values())

scaler.scale(losses).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), GRAD_CLIP)
scaler.step(optimizer)
scaler.update()
```

`GradScaler` prevents float16 underflow in the gradients. This typically provides a 30–50% throughput improvement over full float32 training on CUDA.

On MPS, AMP is skipped and the model trains in float32. MPS fp16 detection losses are stable without a scaler, but the detection head's indexing operations can produce silent precision errors.

---

## 6. Resume Behaviour

A checkpoint is saved after every epoch to `fasterrcnn_output/checkpoints/last.pth`. It stores epoch number, model weights, optimizer state, scheduler state, and best mAP. Re-running the same command resumes automatically:

```bash
python train_fasterrcnn.py              # detects last.pth → resumes
python train_fasterrcnn.py --skip-negatives   # if negatives already staged
```

To force a fresh start:

```bash
rm -rf fasterrcnn_output/checkpoints
python train_fasterrcnn.py
```

---

## 7. All Commands

| Command | Purpose |
|---|---|
| `python train_fasterrcnn.py` | Full pipeline from scratch |
| `python train_fasterrcnn.py --dry-run` | 2-epoch timing estimate |
| `python train_fasterrcnn.py --skip-negatives` | Skip hard-negative download |
| `python train_fasterrcnn.py --figures-only` | Regenerate figures from `best.pth` |
| `python train_fasterrcnn.py --export-only` | Re-export mobile models from `best.pth` |
| `python train_fasterrcnn.py --no-figures` | Train without generating figures |
| `python train_fasterrcnn.py --epochs 15` | Override epoch count |
| `DRY_RUN=1 python train_fasterrcnn.py` | Dry-run via environment variable |

---

## 8. Mobile Export

After training, the best checkpoint is exported to `fasterrcnn_output/models/`:

| File | Format | Primary use case |
|---|---|---|
| `crop_disease_fasterrcnn.ptl` | TorchScript mobile | Android / iOS via LibTorch |
| `crop_disease_fasterrcnn.onnx` | ONNX | Any ONNX Runtime (universal) |
| `crop_disease_fasterrcnn_backbone.pte` | ExecuTorch | Backbone feature extractor on-device |
| `crop_disease_fasterrcnn.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec |

Re-export at any time:

```bash
python train_fasterrcnn.py --export-only
```

---

## 9. Output Figures

| File | Requires training | Contents |
|---|---|---|
| `fig_01_dataset_overview.png` | No | Images per split, hard-negative ratio |
| `fig_02_class_distribution_train.png` | No | Per-class annotation count |
| `fig_03_cross_split_distribution.png` | No | Class distribution across splits |
| `fig_04_annotation_density.png` | No | Boxes-per-image histogram |
| `fig_05_bbox_spatial_heatmap.png` | No | Bounding-box centre density maps |
| `fig_06_bbox_geometry.png` | No | Width/height scatter, aspect ratio, area |
| `fig_07_class_imbalance.png` | No | Val/Test vs Train frequency ratios |
| `fig_08_training_config.png` | No | Hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | No | LR schedule + augmentation profile |
| `fig_10_training_metrics.png` | Yes | Loss components + mAP@0.5 curves |
| `fig_11_per_class_ap.png` | Yes | Per-class AP@0.5 bar chart (23 classes) |

---

## 10. Inference

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

# Rebuild model structure
model = fasterrcnn_resnet50_fpn_v2(weights=None)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 24
)

# Load checkpoint
ckpt = torch.load("fasterrcnn_output/checkpoints/best.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image)   # PIL Image or numpy array

with torch.no_grad():
    predictions = model([img_tensor])

boxes  = predictions[0]["boxes"]    # shape [N, 4] — XYXY pixel coords
labels = predictions[0]["labels"]   # shape [N]   — 1-indexed (1=Corn Cercospora…)
scores = predictions[0]["scores"]   # shape [N]   — confidence

# Filter by confidence
keep = scores > 0.50
```

---

## 11. References

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015.
- Lin, T-Y., et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017.
- torchvision `fasterrcnn_resnet50_fpn_v2`: https://pytorch.org/vision/stable/models/faster_rcnn.html
