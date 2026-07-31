# SE-FPN Faster RCNN — Final Production Model

**Script:** `train_final.py`  
**Output directory:** `final_output/`  
**Role:** Primary research contribution; all 8 custom innovations evaluated against the `resnet50_300` ablation baseline

---

## 1. Overview

The SE-FPN model is the primary research contribution of this project. It builds on the `fasterrcnn_resnet50_fpn_v2` baseline from the ablation study and adds eight custom innovations:

1. SE channel attention gates on every FPN output level
2. K-means anchor clustering from training-set bounding-box statistics
3. EMA (Exponential Moving Average) model weights
4. Gradient accumulation (effective batch = 8)
5. SGDR warm-restart LR schedule
6. Categorised OOD hard negatives (7 semantic categories)
7. Test-Time Augmentation (horizontal-flip TTA)
8. Precision–recall curves and detection confusion matrix

Each innovation is individually motivated by published research and the ablation study findings. The model is designed to maximise per-class mAP@0.5 on the Ghana Crop Disease dataset while remaining exportable to mobile.

---

## 2. Architecture

### Base detector

Same as the `resnet50_300` ablation baseline:
- `fasterrcnn_resnet50_fpn_v2` with 300 proposals and NMS threshold 0.7
- num_classes = 24 (0 = background, 1–23 = disease classes)

### Innovation 1 — SE-FPN Channel Attention

A Squeeze-and-Excitation (SE) gate is inserted after each of the 5 FPN output levels (P2–P6). This is the core architectural contribution.

**SE block:**

```
FPN output [B, C, H, W]
    ↓ Global Average Pooling  → [B, C, 1, 1]
    ↓ FC (C → C/16, ReLU)    → [B, C/16, 1, 1]
    ↓ FC (C/16 → C, Sigmoid) → [B, C, 1, 1]
    ↓ Channel-wise multiply   → [B, C, H, W]  (re-calibrated)
```

The SE gate learns to amplify feature channels that are discriminative for disease detection (e.g., discolouration, lesion texture) and suppress background-texture channels. Applied to every FPN level, it provides multi-scale channel recalibration without adding significant parameters (~0.2% parameter increase over the baseline).

**Motivation:** Hu et al. (SE-Net, 2018) demonstrated that SE gates improve ImageNet top-1 accuracy by 0.5–3% across ResNet, Inception, and VGG architectures. For crop disease detection where the discriminative signal is often localised colour and texture patterns, channel attention is particularly well-suited.

### Innovation 2 — K-means Anchor Clustering

The default COCO anchor sizes (32, 64, 128, 256, 512 px) were designed for general-object detection. Disease lesions on crop leaves are typically smaller (10–100 px) and exhibit different aspect ratios.

1-D k-means clustering is applied to `sqrt(width × height)` of all training-set bounding boxes to derive 5 dataset-specific anchor sizes. These replace the COCO defaults in the RPN.

**Motivation:** Redmon & Farhadi (YOLOv2, 2017) showed that data-driven anchor priors improve recall substantially when the object size distribution differs from COCO.

### Innovation 3 — EMA Model Weights

An Exponential Moving Average (EMA) shadow copy of the model is maintained throughout training:

```
ema_weights ← decay × ema_weights + (1 − decay) × model_weights
```

Evaluation always uses the EMA model, not the instantaneous model. This smooths out training noise and consistently provides a 0.5–2% mAP improvement over the final instantaneous weights at no training-time cost.

- EMA decay: `0.9998`
- Shadow weights are stored on CPU to avoid additional MPS/CUDA memory pressure
- The EMA state is saved in the checkpoint so training resumes with the correct shadow

### Innovation 4 — Gradient Accumulation

Gradients are accumulated over 2 mini-batches before each optimiser step, giving an effective batch size of 8 (2 × physical batch of 4) while keeping per-step GPU memory usage at a single batch of 4.

Larger effective batch sizes are well-known to improve convergence stability for fine-tuning pre-trained models (Goyal et al., 2017).

With AMP on CUDA:
```python
# Accumulate scaled gradients over 2 steps
scaler.scale(losses / ACCUM_STEPS).backward()

# Unscale + step only at accumulation boundary
if (bi + 1) % ACCUM_STEPS == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    ema.update(model)
```

### Innovation 5 — SGDR Warm Restarts

After a linear warm-up phase (3 epochs), the learning rate follows `CosineAnnealingWarmRestarts` (T₀=12, T_mult=2):

```
Epoch  0–2:   Linear warm-up 0 → LR₀
Epoch  3–14:  Cosine annealing T₀=12  (first cycle)
Epoch 15–38:  Cosine annealing T₁=24  (second cycle, T_mult=2)
```

Warm restarts periodically escape local minima and improve final mAP compared to monotone cosine decay, particularly on datasets with fine-grained intra-class variation.

**Motivation:** Loshchilov & Hutter (SGDR, 2017).

### Innovation 6 — Categorised OOD Hard Negatives

300 hard-negative images are distributed across 7 semantic categories to provide richer OOD coverage than random images:

| Category | Count | Why included |
|---|---|---|
| Animals | 50 | Organic textures; shape ambiguity |
| People | 50 | Skin texture can visually resemble leaf surfaces |
| Cityscape | 40 | Background clutter, many small objects |
| Landscape | 40 | Green outdoor scenes; chromatic similarity to crops |
| Transport | 35 | Structured metal/plastic — tests shape invariance |
| Objects | 45 | Domestic items with varied texture |
| Indoor | 40 | Controlled lighting; diverse colour distributions |

Compared to generic random images, categorised negatives provide denser coverage of semantically distinct scene types, improving the model's ability to generalise OOD suppression.

### Innovation 7 — Test-Time Augmentation (TTA)

At final evaluation, each image is also evaluated with a horizontal flip. The two sets of predictions are merged using score-weighted batched NMS:

```
predictions = model([image, flipped_image])
merged = batched_nms(predictions[0] + flip_back(predictions[1]), score_thresh=0.5)
```

TTA improves recall on objects near image edges and suppresses spurious high-confidence detections that appear only on the original orientation.

### Innovation 8 — PR Curves and Detection Confusion Matrix

Two evaluation outputs not present in the baseline or ablation scripts:

**Precision–recall curves** (`fig_10_pr_curves.png`): One curve per class, with the mAP area shaded. These are standard in detection papers and allow readers to assess the confidence threshold trade-off for each disease class.

**Detection confusion matrix** (`fig_11_confusion_matrix.png`): Rows = ground-truth class; columns = predicted class + false-negative column. Unlike a classification confusion matrix, this includes the model's localisation performance (a correct class prediction with IoU < 0.5 is counted as a miss).

---

## 3. Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Architecture | SE-FPN + FasterRCNN-ResNet50-FPN-v2 | SE gates on all 5 FPN levels |
| num_classes | 24 | 23 diseases (1–23) + background (0) |
| Image size | 640 × 640 | |
| Batch size | 4 (effective 8) | Gradient accumulation over 2 steps |
| Epochs | 50 | SGDR benefits from longer training than the baseline |
| Early stopping | patience = 10 | |
| Optimizer | SGD, momentum = 0.9, weight decay = 5e-4 | |
| LR (peak) | 5e-3 | |
| LR schedule | Linear warmup (3 ep) → SGDR T₀=12, T_mult=2 | |
| Backbone freeze | First 5 epochs | |
| Gradient clip | 10.0 | |
| EMA decay | 0.9998 | Shadow updated every optimiser step |
| K-means anchors | 5 data-specific sizes | Replaces COCO defaults |
| Hard negatives | 300 across 7 categories | |
| Eval frequency | Every 3 epochs | Using EMA model |
| TTA at eval | Horizontal flip | Final evaluation only |
| AMP | CUDA only | `autocast` + `GradScaler`; MPS uses float32 |
| Workers | 0 (MPS) / 8 (CUDA) | |
| pin_memory | False (MPS) / True (CUDA) | |
| cudnn.benchmark | True (CUDA only) | |

---

## 4. Usage

### Quick start

```bash
python train_final.py
```

This runs 4 steps automatically: hard negatives → K-means anchors → training (50 ep, EMA, grad accum, SGDR) → TTA evaluation + 15 figures + export.

### All commands

| Command | Purpose |
|---|---|
| `python train_final.py` | Full 4-step pipeline from scratch |
| `python train_final.py --dry-run` | 2-epoch timing estimate |
| `python train_final.py --skip-negatives` | Negatives already downloaded |
| `python train_final.py --figures-only` | Regenerate all 15 figures from `best.pth` |
| `python train_final.py --export-only` | Re-export best checkpoint to mobile formats |
| `python train_final.py --no-figures` | Train without generating figures |
| `python train_final.py --no-ema` | Disable EMA (faster iteration, slightly lower mAP) |
| `python train_final.py --no-tta` | Disable TTA at final evaluation |
| `python train_final.py --epochs 60` | Override epoch count |
| `DRY_RUN=1 python train_final.py` | Dry-run via environment variable |

### Resume

```bash
python train_final.py              # detects last.pth → resumes with EMA + SGDR state
python train_final.py --skip-negatives
```

Force fresh start:

```bash
rm -rf final_output/checkpoints
python train_final.py
```

---

## 5. Checkpoint Format

```python
{
    "epoch":                int,
    "model_state_dict":     dict,      # instantaneous model weights
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,      # SGDR cycle position preserved
    "ema_state_dict":       dict,      # EMA shadow weights
    "best_map":             float,
    "history":              dict,      # full per-epoch metrics
}
```

`best.pth` always stores the EMA model's best checkpoint. Loading for inference:

```python
ckpt = torch.load("final_output/checkpoints/best.pth", map_location="cpu")
model = build_model(anchor_sizes=...)
model.load_state_dict(ckpt["model_state_dict"])
ema = EMAModel(model)
ema.load_state_dict(ckpt["ema_state_dict"])
ema.apply_shadow(model)   # replace model weights with EMA shadow
```

---

## 6. Mobile Export

| File | Format | Use case |
|---|---|---|
| `crop_disease_final.ptl` | TorchScript mobile | Android / iOS via LibTorch |
| `crop_disease_final.onnx` | ONNX | Any ONNX Runtime |
| `crop_disease_final_backbone.pte` | ExecuTorch | Backbone on-device |
| `crop_disease_final.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, anchor sizes |

---

## 7. Output Figures

### Pre-training (generated without a checkpoint)

| File | Contents |
|---|---|
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_anchor_analysis.png` | K-means cluster sizes vs COCO defaults; training bbox distribution |
| `fig_03_se_fpn_architecture.png` | SE-FPN block diagram: FPN levels + SE gates |
| `fig_04_se_attention_detail.png` | SE flow: pool → FC → sigmoid → channel scale |
| `fig_05_ema_weights.png` | EMA decay curve; model vs shadow divergence |
| `fig_06_lr_schedule.png` | Full SGDR schedule: warmup + warm-restart cosine cycles |
| `fig_07_hard_negatives.png` | OOD category breakdown (7 groups × 300 images) |
| `fig_12_gradient_accumulation.png` | Accumulation strategy vs standard batch diagram |
| `fig_14_bbox_geometry.png` | Width/height scatter, aspect-ratio, area by crop |
| `fig_15_summary.png` | One-page summary of all 8 contributions |

### Post-training (require `best.pth`)

| File | Contents |
|---|---|
| `fig_08_training_metrics.png` | 4-component loss + mAP@0.5 curves |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart (23 classes) |
| `fig_10_pr_curves.png` | Precision–recall curves per class |
| `fig_11_confusion_matrix.png` | Detection confusion matrix |
| `fig_13_cross_model_comparison.png` | SE-FPN final vs baseline: mAP, FPS, params |

---

## 8. References

- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.
- Redmon, J. & Farhadi, A. (2017). *YOLO9000: Better, Faster, Stronger*. CVPR 2017. (K-means anchors)
- Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017.
- Tan, M., et al. (2020). *EfficientDet: Scalable and Efficient Object Detection*. CVPR 2020. (EMA weights)
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. (Gradient accumulation / large batch)
- Lin, T-Y., et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017.
