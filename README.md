# Crop Disease Detection — Two-Stage Pipeline

Object detection system for diagnosing diseases in corn, pepper, and tomato crops.
23 disease/health classes across 3 crop types, trained on the Ghana Crop Disease Challenge dataset.

The system uses a **two-stage pipeline**: a crop-type classifier first identifies the leaf species, then a disease detector runs on images confirmed to be from a known crop. This prevents cross-crop false positives (e.g., mango leaves being labelled as Tomato diseases).

Five training scripts are provided:

| Script | Purpose | Architecture | Output |
| ------ | ------- | ------------ | ------ |
| `generate_classifier_csv.py` | Build classifier CSVs | — | `dataset/classifier_*.csv` |
| `train_classifier.py` | **Stage 1 — Crop classifier** | EfficientNet-B2 | `outputs/classifier_output/` |
| `train.py` | Stage 2 — YOLO26 detector | YOLO26n (ultralytics) | `outputs/yolo_output/` |
| `train_fasterrcnn.py` | Stage 2 — Faster RCNN baseline | ResNet-50-FPN-v2 | `outputs/fasterrcnn_output/` |
| `train_alt_fasterrcnn.py` | Ablation study | 7 Faster RCNN variants | `outputs/alt_fasterrcnn_output/` |
| `train_final.py` | Stage 2 — **SE-FPN final model** | ResNet-50-FPN-v2 + SE attention | `outputs/final_output/` |

## Documentation

Detailed documentation for each model is in the [`docs/`](docs/) folder:

| Document | Contents |
| -------- | -------- |
| [`docs/01_crop_classifier.md`](docs/01_crop_classifier.md) | EfficientNet-B2 crop-type classifier — architecture, training, inference API |
| [`docs/02_yolo26_detector.md`](docs/02_yolo26_detector.md) | YOLO26n detector — pipeline, MPS patch, all outputs |
| [`docs/03_fasterrcnn_baseline.md`](docs/03_fasterrcnn_baseline.md) | Faster RCNN v2 baseline — corrections from notebook, AMP, export |
| [`docs/04_fasterrcnn_ablation.md`](docs/04_fasterrcnn_ablation.md) | Ablation study — 7 configurations, findings, paper figures |
| [`docs/05_sefpn_final_model.md`](docs/05_sefpn_final_model.md) | SE-FPN final model — all 8 research contributions in detail |
| [`docs/06_two_stage_pipeline.md`](docs/06_two_stage_pipeline.md) | Pipeline integration — code examples, threshold tuning, limitations |
| [`docs/07_dataset.md`](docs/07_dataset.md) | Dataset — splits, class distribution, label conventions, citation |

---

## Project Structure

```text
crop_disease_detection/
├── generate_classifier_csv.py ← Build crop-type classifier CSVs from YOLO labels
├── train_classifier.py        ← Stage 1: EfficientNet-B2 crop-type classifier
├── train.py                   ← Stage 2: YOLO26 detector
├── train_fasterrcnn.py        ← Stage 2: Faster RCNN v2 baseline
├── train_alt_fasterrcnn.py    ← Faster RCNN ablation study (7 configurations)
├── train_final.py             ← Stage 2: SE-FPN final model (research contributions)
├── main.ipynb                 ← Kaggle-compatible YOLO training notebook
├── export.ipynb               ← ONNX + ExecuTorch export + Android/Kotlin guide
├── detector-torch.ipynb       ← Original Faster RCNN notebook (upgraded by train_fasterrcnn.py)
├── app_gradio.py              ← Gradio web demo (run after training)
├── app_streamlit.py           ← Streamlit web demo (run after training)
├── requirements.txt           ← All Python dependencies
├── data_fixed.yaml            ← Auto-generated at YOLO training time (do not edit)
│
├── docs/                      ← Detailed model documentation
│   ├── 01_crop_classifier.md  ← EfficientNet-B2 classifier — architecture, training, API
│   ├── 02_yolo26_detector.md  ← YOLO26n detector — pipeline, MPS patch, outputs
│   ├── 03_fasterrcnn_baseline.md
│   ├── 04_fasterrcnn_ablation.md
│   ├── 05_sefpn_final_model.md
│   ├── 06_two_stage_pipeline.md
│   └── 07_dataset.md
│
├── data/                     ← YOLO dataset (Roboflow YOLO format)
│   ├── main/
│   │   ├── train/            ← Training images + labels (hard negatives staged here)
│   │   ├── valid/            ← Validation images + labels
│   │   └── test/             ← Test images + labels
│   └── negatives/            ← Shared hard-negative images (all pipelines)
│
├── dataset/                  ← Faster RCNN + classifier dataset
│   ├── train/                ← Training images
│   ├── validate/             ← Validation images
│   ├── test/                 ← Test images
│   ├── final_train_labels.csv
│   ├── final_validate_labels.csv
│   ├── final_test_labels.csv
│   ├── classifier_train.csv  ← Crop-type classifier training split (2,861 images)
│   ├── classifier_valid.csv  ← Crop-type classifier validation split (1,024 images)
│   ├── classifier_test.csv   ← Crop-type classifier test split (1,016 images)
│   └── label_map.json        ← Class name → integer label (1-indexed)
│
├── runs/
│   └── crop_disease_yolo26/
│       └── weights/
│           ├── best.pt       ← Best YOLO checkpoint
│           └── last.pt       ← Latest YOLO checkpoint (resume)
│
└── outputs/                  ← All training outputs (auto-created)
    ├── classifier_output/    ← Crop-type classifier
    │   ├── best.pth          ← Best EfficientNet-B2 checkpoint
    │   ├── last.pth          ← Latest checkpoint (resume)
    │   ├── metrics_history.json
    │   └── figures/          ← Training curve, confusion matrix, class distribution
    │
    ├── yolo_output/          ← YOLO publication figures (300 DPI PNG)
    │
    ├── fasterrcnn_output/    ← Faster RCNN baseline
    │   ├── checkpoints/
    │   │   ├── best.pth
    │   │   ├── last.pth
    │   │   └── epoch_NNNN.pth
    │   ├── models/
    │   │   ├── crop_disease_fasterrcnn.ptl       ← TorchScript mobile (primary)
    │   │   ├── crop_disease_fasterrcnn.onnx      ← ONNX (universal fallback)
    │   │   ├── crop_disease_fasterrcnn_backbone.pte
    │   │   ├── crop_disease_fasterrcnn.pte
    │   │   └── model_metadata.yaml
    │   ├── metrics_history.json
    │   └── fig_01_*.png … fig_11_*.png
    │
    ├── alt_fasterrcnn_output/  ← Ablation study
    │   ├── checkpoints/{config_id}/  last.pth  best.pth  epoch_NNNN.pth
    │   ├── models/{config_id}_best.pth
    │   ├── figures/
    │   │   ├── fig_arch_01_pipeline.png … fig_arch_05_fpn_structure.png
    │   │   ├── fig_cmp_01_map_bar.png … fig_cmp_08_nms_anchor_ablation.png
    │   │   ├── fig_tbl_01_main_results.png  fig_tbl_02_speed_comparison.png
    │   │   └── table_ablation.tex
    │   └── results.json
    │
    └── final_output/         ← SE-FPN final model
        ├── checkpoints/
        │   ├── best.pth      ← Best checkpoint (EMA weights included)
        │   ├── last.pth
        │   └── epoch_NNNN.pth
        ├── models/
        │   ├── crop_disease_final.ptl            ← TorchScript mobile (primary)
        │   ├── crop_disease_final.onnx           ← ONNX (universal fallback)
        │   ├── crop_disease_final_backbone.pte
        │   ├── crop_disease_final.pte
        │   └── model_metadata.yaml
        ├── metrics_history.json
        └── fig_01_*.png … fig_15_*.png
```

---

## Crop-Type Classifier — `train_classifier.py`

Stage 1 of the two-stage pipeline. An EfficientNet-B2 fine-tuned to classify a leaf
image as Corn, Pepper, or Tomato before the disease detector runs. Any image whose
maximum class probability falls below the confidence threshold (default 0.55) is
rejected as an unknown/non-crop leaf.

### Quick start (Classifier)

```bash
# Step 1 — generate the classifier CSVs from the existing YOLO label files
python generate_classifier_csv.py

# Step 2 — train the classifier (40 epochs, early stopping, EfficientNet-B2)
python train_classifier.py
```

Classifier CSVs are derived from the YOLO label files already in `data/main/`.
Running `generate_classifier_csv.py` takes under a minute and produces:

- `dataset/classifier_train.csv` (2,861 images)
- `dataset/classifier_valid.csv` (1,024 images)
- `dataset/classifier_test.csv` (1,016 images)

Hard-negative images (empty `.txt` labels, no crop assignment) are excluded
automatically.

### All commands (Classifier)

| Command | When to use |
| ------- | ----------- |
| `python generate_classifier_csv.py` | Build CSVs — run once before training |
| `python train_classifier.py` | Full training from scratch (40 epochs) |
| `python train_classifier.py --dry-run` | 2-epoch timing estimate |
| `python train_classifier.py --batch-size 32` | Reduce batch size if OOM |
| `python train_classifier.py --epochs 60` | Override epoch count |
| `python train_classifier.py --workers 4` | Override DataLoader worker count |

### Resume after interruption (Classifier)

Training saves `outputs/classifier_output/last.pth` after every epoch. Re-running the same
command resumes automatically:

```bash
python train_classifier.py   # detects last.pth → resumes
```

To force a fresh start:

```bash
rm outputs/classifier_output/last.pth outputs/classifier_output/best.pth
python train_classifier.py
```

### Training configuration (Classifier)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | EfficientNet-B2 | ImageNet pretrained; 3-class head |
| num_classes | 3 | Corn / Pepper / Tomato |
| Image size | 260 × 260 | EfficientNet-B2 native resolution |
| Batch size | 64 | MPS and CUDA; reduce if OOM |
| Epochs | 40 | Early stopping patience = 8 |
| Optimizer | AdamW (weight decay = 1e-4) | |
| Learning rate | 3e-4 → cosine decay | Linear warmup for 3 epochs |
| Label smoothing | ε = 0.10 | Reduces overconfidence on the imbalanced dataset |
| Gradient clip | 1.0 | |
| AMP | CUDA (fp16 GradScaler) / MPS (fp16 autocast, no scaler) | |
| Workers | 8 (CUDA) / 4 (MPS) | 0 not required for classifier on MPS |
| pin_memory | True (CUDA) / False (MPS) | |
| Device | MPS / CUDA / CPU | Auto-detected |

### Output files (Classifier)

| File | Contents |
| ---- | -------- |
| `outputs/classifier_output/best.pth` | Best checkpoint (val accuracy) |
| `outputs/classifier_output/last.pth` | Latest checkpoint (resume) |
| `outputs/classifier_output/metrics_history.json` | Per-epoch loss, accuracy, F1 |
| `outputs/classifier_output/figures/` | Training curves, confusion matrix, class distribution |

### Using the classifier in inference

```python
from train_classifier import CropClassifier

clf = CropClassifier()                # loads outputs/classifier_output/best.pth
crop, conf, yolo_ids = clf.predict("leaf.jpg")

# Returns:
#   crop     — "Corn" | "Pepper" | "Tomato" | "unknown"
#   conf     — float  (max softmax probability)
#   yolo_ids — list of YOLO class IDs allowed for this crop (empty if unknown)
```

Change the confidence threshold:

```python
clf = CropClassifier(confidence_threshold=0.65)
# or after construction:
clf.threshold = 0.65
```

---

## Two-Stage Pipeline

Combine the crop-type classifier (Stage 1) with any detector (Stage 2) to prevent
cross-crop false positives. The classifier rejects non-crop leaves before the detector runs.

```text
Input image
    │
    ▼
Stage 1: EfficientNet-B2 (Corn / Pepper / Tomato)
    ├── confidence < 0.55 ─────────────► Rejected — "Unknown crop"
    └── confidence ≥ 0.55
            │
            ▼  crop label + allowed YOLO class IDs
Stage 2: YOLO26 or SE-FPN Faster RCNN
    │  (output filtered to crop-specific classes only)
    ▼
Disease bounding boxes + labels
```

### Two-stage usage example (YOLO)

```python
from train_classifier import CropClassifier
from ultralytics import YOLO

clf  = CropClassifier()
yolo = YOLO("runs/crop_disease_yolo26/weights/best.pt")

def detect_disease(image_path: str, conf_thresh: float = 0.50) -> dict:
    crop, crop_conf, allowed_ids = clf.predict(image_path)

    if crop == "unknown":
        return {"status": "rejected", "crop": None, "diseases": []}

    results  = yolo(image_path, conf=conf_thresh)[0]
    diseases = [
        {
            "class_id":   int(b.cls),
            "class_name": yolo.names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox_xyxy":  b.xyxy[0].tolist(),
        }
        for b in results.boxes
        if int(b.cls) in allowed_ids
    ]
    return {"status": "detected", "crop": crop, "crop_conf": crop_conf, "diseases": diseases}
```

See [docs/06_two_stage_pipeline.md](docs/06_two_stage_pipeline.md) for the SE-FPN Faster RCNN
integration, threshold tuning guidance, and known limitations.

---

## YOLO26 Pipeline — `train.py`

### Quick start

```bash
pip install -r requirements.txt
python train.py
```

This runs four steps automatically:

1. Writes `data_fixed.yaml` with absolute paths
2. Downloads 300 hard-negative (non-crop) images and stages them in the training split
3. Trains YOLO26 for up to 200 epochs (early-stops after 25 non-improving epochs)
4. Saves publication figures to `outputs/yolo_output/`

### All commands

| Command | When to use |
| ------- | ----------- |
| `python train.py` | Full pipeline from scratch |
| `python train.py --dry-run` | Estimate total training time (1 epoch) |
| `python train.py --skip-negatives` | Negatives already downloaded |
| `python train.py --figures-only` | Regenerate charts without re-training |
| `python train.py --no-figures` | Train only, skip figure generation |
| `python train.py --epochs 50` | Override epoch count |
| `DRY_RUN=1 python train.py` | Dry-run via environment variable |

### Resume after interruption

Training auto-resumes if `runs/crop_disease_yolo26/weights/last.pt` exists:

```bash
python train.py              # detects last.pt → resumes automatically
python train.py --skip-negatives   # if negatives are already staged
```

To force a fresh start:

```bash
rm -rf runs/crop_disease_yolo26
python train.py
```

---

## Faster RCNN v2 Pipeline — `train_fasterrcnn.py`

Upgraded from `detector-torch.ipynb` with the following changes:

| Before (notebook) | After (`train_fasterrcnn.py`) |
| ----------------- | ----------------------------- |
| `fasterrcnn_resnet50_fpn` v1 | `fasterrcnn_resnet50_fpn_v2` (COCO AP 37.0 → 46.7) |
| `num_classes=23` (bug — missing background) | `num_classes=24` (23 diseases + class 0 background) |
| Labels decremented by 1 (bug) | Labels kept 1–23 as-is (Faster RCNN requires ≥ 1) |
| 4 fixed epochs, no resume | 30 epochs, early stopping, full resume from `last.pth` |
| No augmentation | `torchvision.transforms.v2` joint image + bounding-box augmentation |
| `workers=4` (crashes MPS) | `workers=0` (required on macOS Metal) |
| No OOD guard | 200 hard-negative images with empty annotations |
| TorchScript export only | ExecuTorch + ONNX + TorchScript mobile (`.ptl`) export |
| No publication figures | 11 publication figures to `outputs/fasterrcnn_output/` |
| Kaggle-only paths | Local paths, fully portable |

### Quick start (Faster RCNN)

```bash
python train_fasterrcnn.py
```

This runs four steps automatically:

1. Downloads 200 hard-negative (non-crop) images into `data/negatives/`
2. Trains FasterRCNN-ResNet50-FPN-v2 for up to 30 epochs with early stopping
3. Saves 11 publication figures to `outputs/fasterrcnn_output/`
4. Exports the best checkpoint to `outputs/fasterrcnn_output/models/`

### All commands (Faster RCNN)

| Command | When to use |
| ------- | ----------- |
| `python train_fasterrcnn.py` | Full pipeline from scratch |
| `python train_fasterrcnn.py --dry-run` | Estimate training time (2 epochs) |
| `python train_fasterrcnn.py --skip-negatives` | Negatives already downloaded |
| `python train_fasterrcnn.py --figures-only` | Regenerate publication figures only |
| `python train_fasterrcnn.py --export-only` | Re-export best checkpoint to mobile formats |
| `python train_fasterrcnn.py --no-figures` | Train without generating figures |
| `python train_fasterrcnn.py --epochs 10` | Override epoch count |
| `DRY_RUN=1 python train_fasterrcnn.py` | Dry-run via environment variable |

### Resume after interruption (Faster RCNN)

Training saves `outputs/fasterrcnn_output/checkpoints/last.pth` after every epoch.
Re-running the same command resumes automatically:

```bash
python train_fasterrcnn.py              # detects last.pth → resumes
python train_fasterrcnn.py --skip-negatives   # if negatives are already staged
```

The checkpoint stores epoch number, model weights, optimizer state, scheduler
state, and best mAP so the learning-rate schedule continues correctly.

To force a fresh start:

```bash
rm -rf outputs/fasterrcnn_output/checkpoints
python train_fasterrcnn.py
```

### Training configuration

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | FasterRCNN-ResNet50-FPN-v2 | V2 weights; COCO AP 46.7 |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 × 640 | Matches dataset images |
| Batch size | 4 | Safe on 24 GB unified memory |
| Epochs | 30 | Pretrained backbone converges quickly |
| Early stopping | patience = 8 | Stops after 8 non-improving eval epochs |
| Optimizer | SGD (momentum = 0.9) | Standard for Faster RCNN fine-tuning |
| Learning rate | 5e-3 → cosine decay | Linear warmup for 3 epochs |
| Backbone freeze | First 5 epochs | Prevents overwriting pretrained features too early |
| Gradient clip | 10.0 | Prevents exploding gradients |
| Augmentation | HFlip + ColorJitter + Blur | `torchvision.transforms.v2` — no extra deps |
| Hard negatives | 200 images | Diverse non-crop images; empty annotation |
| Eval frequency | Every 5 epochs | VOC mAP@0.5; full eval at end of training |
| Workers | 0 | Required on macOS MPS |
| Device | MPS (M4 Pro) | Auto-detected; falls back to CUDA or CPU |

### Mobile export

After training, three formats are exported to `outputs/fasterrcnn_output/models/`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_fasterrcnn.ptl` | TorchScript mobile | Android / iOS via LibTorch (**primary**) |
| `crop_disease_fasterrcnn.onnx` | ONNX | Any ONNX Runtime (universal fallback) |
| `crop_disease_fasterrcnn_backbone.pte` | ExecuTorch | Backbone feature extractor on-device |
| `crop_disease_fasterrcnn.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec |

Re-export at any time without re-training:

```bash
python train_fasterrcnn.py --export-only
```

---

## Faster RCNN Ablation Study — `train_alt_fasterrcnn.py`

Mirrors the comparative analysis style of the original Faster RCNN paper (Ren et al., 2015)
to justify the final model selection for publication. Trains and evaluates 7 configurations
across backbone depth, RPN proposal count, NMS policy, and anchor scale.

### Configurations

| # | Config ID | Backbone | Proposals | NMS Thresh | Anchor Sizes | Role |
| --- | --------- | -------- | --------- | ---------- | ------------ | ---- |
| 1 | `mobilenet_300` | MobileNetV3-FPN (~19M params) | 300 | 0.7 | default | Lightweight baseline |
| 2 | `resnet50_100` | ResNet50-FPN-v2 (~43M params) | 100 | 0.7 | default | Low-proposal ablation |
| 3 | `resnet50_300` | ResNet50-FPN-v2 (~43M params) | 300 | 0.7 | default | **Selected model ★** |
| 4 | `resnet50_1000` | ResNet50-FPN-v2 (~43M params) | 1000 | 0.7 | default | High-proposal ablation |
| 5 | `resnet50_no_nms` | ResNet50-FPN-v2 (~43M params) | 300 | 1.0 | default | NMS disabled |
| 6 | `resnet50_small_anchors` | ResNet50-FPN-v2 (~43M params) | 300 | 0.7 | 16–256 px | Disease-lesion optimised |
| 7 | `resnet101_300` | ResNet101-FPN (~60M params) | 300 | 0.7 | default | Heavier backbone |

### Quick start (Ablation)

```bash
# Architecture figures (no training needed — useful for paper diagrams)
python train_alt_fasterrcnn.py --arch-figures

# Train all 7 configurations sequentially
python train_alt_fasterrcnn.py

# Train a single configuration (e.g. the selected baseline)
python train_alt_fasterrcnn.py --configs resnet50_300

# Quick timing estimate (2 epochs per config)
python train_alt_fasterrcnn.py --dry-run

# Regenerate all figures from existing results.json
python train_alt_fasterrcnn.py --figures-only
```

### All commands (Ablation)

| Command | When to use |
| ------- | ----------- |
| `python train_alt_fasterrcnn.py` | Train all 7 configs |
| `python train_alt_fasterrcnn.py --configs ID [ID ...]` | Train specific config(s) |
| `python train_alt_fasterrcnn.py --epochs 10` | Override epoch count per config |
| `python train_alt_fasterrcnn.py --dry-run` | 2-epoch timing estimate per config |
| `python train_alt_fasterrcnn.py --arch-figures` | Architecture diagrams only (instant) |
| `python train_alt_fasterrcnn.py --figures-only` | All figures from existing results.json |
| `python train_alt_fasterrcnn.py --skip-negatives` | Reuse cached hard-negative images |
| `python train_alt_fasterrcnn.py --no-figures` | Train without generating figures |

### Resume after interruption (Ablation)

Each configuration saves its own checkpoint to
`outputs/alt_fasterrcnn_output/checkpoints/{config_id}/last.pth`. Re-running the same command
resumes each config from where it left off automatically.

```bash
python train_alt_fasterrcnn.py          # resumes each unfinished config
python train_alt_fasterrcnn.py --configs resnet101_300   # resume one config
```

To force a fresh start for a specific config:

```bash
rm -rf outputs/alt_fasterrcnn_output/checkpoints/resnet101_300
python train_alt_fasterrcnn.py --configs resnet101_300
```

### Training configuration (Ablation)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Epochs per config | 15 | Shorter than final pipeline; ablation is comparative |
| Early stopping | patience = 5 | Halts 5 evals after last improvement |
| Batch size | 4 | Same as final pipeline |
| Optimizer | SGD (momentum = 0.9) | Identical across all configs for fair comparison |
| Learning rate | 5e-3 → cosine decay | Linear warmup for 2 epochs |
| Gradient clip | 10.0 | |
| Hard negatives | 100 images | Shared cache with `train_fasterrcnn.py` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 |
| Speed benchmark | 100 runs, batch=1 | Full model + backbone-only latency |
| Seed | 42 | Fixed across all configs |

### Generated figures

**Architecture diagrams** (generated without training — run `--arch-figures`):

| File | Contents |
| ---- | -------- |
| `fig_arch_01_pipeline.png` | End-to-end detection pipeline (Input → Backbone → RPN → RoI → Output) |
| `fig_arch_02_backbone_comparison.png` | Parameter count, COCO mAP, and depth comparison for 3 backbones |
| `fig_arch_03_rpn_detail.png` | RPN architecture: anchor generation, cls/reg heads, NMS |
| `fig_arch_04_anchor_visualization.png` | Default vs small anchor box scales at 3 aspect ratios |
| `fig_arch_05_fpn_structure.png` | Feature Pyramid Network with bottom-up and top-down pathways |

**Performance comparisons** (generated after training):

| File | Contents |
| ---- | -------- |
| `fig_cmp_01_map_bar.png` | mAP@0.5 bar chart across all 7 configurations |
| `fig_cmp_02_speed_accuracy.png` | FPS vs mAP scatter plot (speed–accuracy trade-off) |
| `fig_cmp_03_convergence.png` | Training loss and validation mAP curves per config |
| `fig_cmp_04_proposal_ablation.png` | mAP and FPS vs proposal count (100 / 300 / 1000) |
| `fig_cmp_05_radar.png` | Normalised multi-metric radar chart |
| `fig_cmp_06_params_vs_map.png` | Model complexity vs detection performance |
| `fig_cmp_07_inference_breakdown.png` | Backbone vs RPN+Head latency stacked bar |
| `fig_cmp_08_nms_anchor_ablation.png` | NMS-off vs NMS-on, and small vs default anchors |
| `fig_tbl_01_main_results.png` | Paper-style Table 1: all configurations |
| `fig_tbl_02_speed_comparison.png` | Paper-style Table 2: latency breakdown |
| `table_ablation.tex` | LaTeX source for inclusion in the paper |

---

## Final SE-FPN Model — `train_final.py`

The primary research contribution of this project. Builds on the Faster RCNN v2 baseline
with eight custom innovations designed to maximise per-class AP on the Ghana Crop Disease
dataset. Each innovation is individually motivated by prior work and evaluated against the
ablation study baseline.

### Custom research contributions

| # | Innovation | Description | Prior work |
| --- | ---------- | ----------- | ---------- |
| 1 | **SE-FPN channel attention** | Squeeze-and-Excitation gates applied to every FPN output level — amplifies disease-discriminative channels, suppresses background texture | Hu et al., SE-Net (2018) |
| 2 | **K-means anchor clustering** | Anchor sizes derived from training-set bbox statistics (1-D k-means on √area) replacing generic COCO priors | Redmon & Farhadi, YOLOv2 (2017) |
| 3 | **EMA model weights** | Shadow copy of model parameters updated after every optimiser step; evaluation uses the EMA model. Typical gain: +0.5–2 % mAP | Tan et al., EfficientDet (2020) |
| 4 | **Gradient accumulation** | 2 mini-batches accumulated per optimiser step → effective batch size 8 without extra VRAM | Standard technique |
| 5 | **SGDR warm restarts** | `CosineAnnealingWarmRestarts` (T₀=12, T_mult=2) after a linear warm-up phase — avoids monotone cosine decay plateaus | Loshchilov & Hutter (2017) |
| 6 | **Categorised OOD hard negatives** | 300 images across 7 semantic categories (animals, people, cityscape, landscape, transport, objects, indoor) — richer OOD diversity than generic random images | Standard hard-negative mining |
| 7 | **Test-Time Augmentation** | Horizontal-flip TTA at final evaluation; predictions merged via score-weighted batched NMS | Standard TTA |
| 8 | **PR curves + confusion matrix** | Precision–recall curves and detection confusion matrix (GT × predicted) as novel evaluation outputs not present in baseline or ablation scripts | Evaluation best practice |

### Quick start (Final model)

```bash
python train_final.py
```

This runs four steps automatically:

1. Downloads 300 categorised hard-negative images into `data/negatives/`
2. Computes dataset-specific k-means anchors from training CSV
3. Trains SE-FPN Faster RCNN for up to 50 epochs with EMA, gradient accumulation, and SGDR
4. Evaluates with TTA, generates 15 publication figures, and exports to `outputs/final_output/models/`

### All commands (Final model)

| Command | When to use |
| ------- | ----------- |
| `python train_final.py` | Full 4-step pipeline from scratch |
| `python train_final.py --dry-run` | 2-epoch timing estimate |
| `python train_final.py --skip-negatives` | Negatives already downloaded |
| `python train_final.py --figures-only` | Regenerate all figures from `best.pth` |
| `python train_final.py --export-only` | Re-export best checkpoint to mobile formats |
| `python train_final.py --no-figures` | Train without generating figures |
| `python train_final.py --no-ema` | Disable EMA (faster iteration, lower mAP) |
| `python train_final.py --no-tta` | Disable TTA at final evaluation |
| `python train_final.py --epochs 60` | Override epoch count |
| `DRY_RUN=1 python train_final.py` | Dry-run via environment variable |

### Resume after interruption (Final model)

Training saves `outputs/final_output/checkpoints/last.pth` after every epoch, including
the EMA shadow weights and SGDR scheduler state. Re-running the same command
resumes automatically:

```bash
python train_final.py              # detects last.pth → resumes
python train_final.py --skip-negatives   # if negatives are already staged
```

To force a fresh start:

```bash
rm -rf outputs/final_output/checkpoints
python train_final.py
```

### Final model — training configuration

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | SE-FPN + FasterRCNN-ResNet50-FPN-v2 | SE gates on all 5 FPN output levels |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 × 640 | |
| Batch size | 4 (effective 8) | Gradient accumulation over 2 steps |
| Epochs | 50 | More than baseline — SGDR benefits from longer training |
| Early stopping | patience = 10 | |
| Optimizer | SGD (momentum = 0.9, weight decay = 5e-4) | |
| Learning rate | 5e-3 (peak) | Linear warmup 3 epochs → SGDR T₀=12, T_mult=2 |
| Backbone freeze | First 5 epochs | Prevents corrupting pretrained features |
| Gradient clip | 10.0 | |
| EMA decay | 0.9998 | Shadow weights updated after every optimiser step |
| Anchors | K-means from training set | 5 dataset-specific sizes replace COCO defaults |
| Hard negatives | 300 across 7 categories | Stored in `data/negatives/` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 using EMA model |
| TTA at eval | Horizontal flip | Merged via batched NMS |
| Workers | 0 | Required on macOS MPS |
| Device | MPS (M4 Pro) | Auto-detected; falls back to CUDA or CPU |

### Final model — mobile export

After training, four formats are exported to `outputs/final_output/models/`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_final.ptl` | TorchScript mobile | Android / iOS via LibTorch (**primary**) |
| `crop_disease_final.onnx` | ONNX | Any ONNX Runtime (universal fallback) |
| `crop_disease_final_backbone.pte` | ExecuTorch | Backbone feature extractor on-device |
| `crop_disease_final.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec, anchor sizes |

Re-export at any time without re-training:

```bash
python train_final.py --export-only
```

### Final model — generated figures

**Pre-training figures** (generated before training — run `--figures-only` with no checkpoint for these):

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_anchor_analysis.png` | K-means cluster sizes vs COCO defaults; training bbox distribution |
| `fig_03_se_fpn_architecture.png` | SE-FPN block diagram showing FPN levels + SE gates |
| `fig_04_se_attention_detail.png` | SE squeeze-and-excitation flow: pool → FC → sigmoid → scale |
| `fig_05_ema_weights.png` | EMA decay curve; model vs shadow weight divergence illustration |
| `fig_06_lr_schedule.png` | Full SGDR schedule: linear warmup + warm-restart cosine cycles |
| `fig_07_hard_negatives.png` | OOD category breakdown (7 semantic groups, 300 images) |
| `fig_12_gradient_accumulation.png` | Gradient accumulation strategy vs standard batch diagram |
| `fig_14_bbox_geometry.png` | Width/height scatter, aspect-ratio, area-by-crop analysis |
| `fig_15_summary.png` | Summary panel showing all 8 custom contributions |

**Post-training figures** (require `outputs/final_output/checkpoints/best.pth`):

| File | Contents |
| ---- | -------- |
| `fig_08_training_metrics.png` | 4-component loss + mAP@0.5 curves across epochs |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart (23 classes) |
| `fig_10_pr_curves.png` | Precision–recall curves per class (with mAP area shaded) |
| `fig_11_confusion_matrix.png` | Detection confusion matrix: rows = GT, cols = predicted + FN/FP |
| `fig_13_cross_model_comparison.png` | SE-FPN final vs Faster RCNN v2 baseline: mAP, FPS, params |

---

## Publication Figures

### YOLO26 — `outputs/yolo_output/`

Generated by `python train.py` or `python train.py --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_class_distribution_train.png` | Per-class annotation count (training set) |
| `fig_03_cross_split_distribution.png` | Absolute + normalised distribution across splits |
| `fig_04_annotation_density.png` | Boxes-per-image histograms (train + val) |
| `fig_05_bbox_spatial_heatmap.png` | Bounding-box centre heatmaps per crop type |
| `fig_06_bbox_size_analysis.png` | Width/height scatter, aspect ratio, area by crop |
| `fig_07_class_imbalance.png` | Val/Test vs Train frequency ratios per class |
| `fig_08_training_config.png` | Training hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | Cosine LR schedule + augmentation profile |
| `fig_10_training_metrics.png` | Loss and mAP curves (only after training completes) |

### Faster RCNN v2 — `outputs/fasterrcnn_output/`

Generated by `python train_fasterrcnn.py` or `python train_fasterrcnn.py --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_class_distribution_train.png` | Per-class annotation count (training set) |
| `fig_03_cross_split_distribution.png` | Absolute + normalised distribution across splits |
| `fig_04_annotation_density.png` | Boxes-per-image histograms (train + val) |
| `fig_05_bbox_spatial_heatmap.png` | Bounding-box centre heatmaps per crop type |
| `fig_06_bbox_geometry.png` | Width/height scatter, aspect ratio, area by crop |
| `fig_07_class_imbalance.png` | Val/Test vs Train frequency ratios per class |
| `fig_08_training_config.png` | Faster RCNN hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | Cosine LR schedule + v2 augmentation profile |
| `fig_10_training_metrics.png` | 4-component loss + mAP curves (post-training) |
| `fig_11_per_class_ap.png` | Per-class AP@0.5 bar chart (post-training evaluation) |

### SE-FPN Final model — `outputs/final_output/`

Generated by `python train_final.py` or `python train_final.py --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_anchor_analysis.png` | K-means cluster sizes vs COCO defaults; training bbox distribution |
| `fig_03_se_fpn_architecture.png` | SE-FPN block diagram: FPN levels + SE channel-attention gates |
| `fig_04_se_attention_detail.png` | SE flow: global avg pool → FC → ReLU → FC → sigmoid → channel scale |
| `fig_05_ema_weights.png` | EMA decay curve; model vs shadow weight divergence illustration |
| `fig_06_lr_schedule.png` | Full SGDR schedule: linear warmup (3 ep) + warm-restart cosine cycles |
| `fig_07_hard_negatives.png` | OOD category breakdown (7 groups × proportional counts = 300 images) |
| `fig_08_training_metrics.png` | 4-component loss + mAP@0.5 across all epochs (post-training) |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart for all 23 classes (post-training) |
| `fig_10_pr_curves.png` | Precision–recall curves per class with mAP area shaded (post-training) |
| `fig_11_confusion_matrix.png` | Detection confusion matrix: GT rows × predicted cols, FN/FP margins |
| `fig_12_gradient_accumulation.png` | Gradient accumulation strategy vs standard single-step batch diagram |
| `fig_13_cross_model_comparison.png` | SE-FPN final vs Faster RCNN v2 baseline: mAP, FPS, param count |
| `fig_14_bbox_geometry.png` | Width/height scatter, aspect-ratio distribution, area by crop type |
| `fig_15_summary.png` | One-page summary panel covering all 8 custom research contributions |

---

## Classes (23 total)

Both pipelines detect the same 23 disease classes.

| Label (YOLO) | Label (FasterRCNN) | Class | Crop |
| ------------ | ------------------ | ----- | ---- |
| 0 | 1 | Cercospora Leaf Spot | Corn |
| 1 | 2 | Common Rust | Corn |
| 2 | 3 | Healthy | Corn |
| 3 | 4 | Streak | Corn |
| 4 | 5 | Northern Leaf Blight | Corn |
| 5 | 6 | Leaf Curl | Pepper |
| 6 | 7 | Cercospora | Pepper |
| 7 | 8 | Leaf Blight | Pepper |
| 8 | 9 | Bacterial Spot | Pepper |
| 9 | 10 | Leaf Mosaic | Pepper |
| 10 | 11 | Healthy | Pepper |
| 11 | 12 | Fusarium | Pepper |
| 12 | 13 | Septoria | Pepper |
| 13 | 14 | Late Blight | Pepper |
| 14 | 15 | Early Blight | Pepper |
| 15 | 16 | Late Blight | Tomato |
| 16 | 17 | Early Blight | Tomato |
| 17 | 18 | Bacterial Spot | Tomato |
| 18 | 19 | Septoria | Tomato |
| 19 | 20 | Fusarium | Tomato |
| 20 | 21 | Leaf Curl | Tomato |
| 21 | 22 | Healthy | Tomato |
| 22 | 23 | Mosaic | Tomato |

> **Label indexing note:** YOLO uses 0-indexed labels; Faster RCNN reserves 0 for
> background and uses 1-indexed labels. The `label_map.json` in `dataset/` uses
> 1-indexed values and matches the Faster RCNN convention directly.

---

## Hardware Notes

### Apple M4 Pro (MPS)

All scripts auto-detect MPS and apply the appropriate settings.

**Crop-type classifier:**

```text
Device    : mps
Batch     : 64
Workers   : 4
pin_memory: False   (unified memory — pinning has no benefit)
AMP       : fp16 autocast enabled (EfficientNet-B2 is stable at fp16 on MPS)
GradScaler: not used (MPS does not need loss scaling)
```

**YOLO26:**

```text
Device  : mps
Batch   : 16
Workers : 0     (Metal requires this for detection models)
Cache   : disk
AMP     : True  (fp16 ~1.4× speedup)
```

**Faster RCNN v2 / SE-FPN:**

```text
Device    : mps
Batch     : 4   (Faster RCNN is memory-heavier than YOLO)
Workers   : 0   (Metal requires this for detection models)
pin_memory: False
AMP       : not used (detection head index ops have fp16 precision issues on MPS)
EMA       : shadow weights stored on CPU (SE-FPN only)
```

### CUDA (GPU server)

When a CUDA device is detected, all scripts automatically apply CUDA-specific optimisations:

**Crop-type classifier:**

```text
Device    : cuda
Batch     : 64
Workers   : 8
pin_memory: True
AMP       : fp16 autocast + GradScaler (full mixed-precision training)
```

**Faster RCNN v2 / SE-FPN:**

```text
Device    : cuda
Batch     : 4 (effective 8 via gradient accumulation for SE-FPN)
Workers   : 8
pin_memory: True
AMP       : fp16 autocast + GradScaler
cudnn.benchmark: True
```

**YOLO26:** switches to DDP automatically when multiple GPUs are detected.

### CPU fallback

All scripts fall back to CPU if neither MPS nor CUDA is available.
Use `--dry-run` first to estimate time before committing to a full run.

---

## OOD (Out-of-Distribution) Guard

### Primary guard — two-stage pipeline

The crop-type classifier (Stage 1) is the strongest OOD safeguard. A mango leaf or a
completely unrelated image produces low softmax probabilities across all three known classes
(Corn, Pepper, Tomato) because EfficientNet-B2 was trained only on those three crops.
The maximum probability is typically below the 0.55 threshold, so the image is rejected
before any disease detector runs.

This solves the failure mode where YOLO and Faster RCNN — trained only on crop images —
hallucinate disease labels on visually similar non-crop leaves (e.g., mango → Tomato).

### In-detector hard negatives (backup guard)

All detector training pipelines also include hard-negative images to suppress false
positives inside the detector itself:

**YOLO26:**

1. **Hard negatives** — 300 random images with empty `.txt` labels
2. **Label smoothing = 0.10** — prevents overconfident closed-set predictions
3. **Confidence threshold = 0.50** — applied at inference in both demo apps

**Faster RCNN v2:**

1. **Hard negatives** — 200 random images with empty annotation targets (boxes shape `[0, 4]`);
   the RPN assigns all proposals to background, explicitly training the model to suppress detections
2. **Confidence threshold** — apply `score_thresh ≥ 0.50` at inference time

**SE-FPN Final model:**

1. **Categorised hard negatives** — 300 images across 7 semantic categories (animals 50,
   people 50, cityscape 40, landscape 40, transport 35, objects 45, indoor 40); richer
   OOD diversity than generic random images; empty annotation targets
2. **Confidence threshold** — apply `score_thresh ≥ 0.50` at inference time
3. **TTA** — horizontal-flip at evaluation further reduces spurious detections on near-miss frames

All pipelines share the same `data/negatives/` folder. If all three have been run,
the folder will contain up to 300 images (the final model and YOLO both use 300;
the baseline Faster RCNN uses up to 200).

---

## Notebooks

| Notebook | Purpose | Pipeline |
| -------- | ------- | -------- |
| `main.ipynb` | Full YOLO training pipeline | YOLO26 |
| `export.ipynb` | ONNX + ExecuTorch export + Android guide | YOLO26 |
| `detector-torch.ipynb` | Original Faster RCNN notebook (reference) | Faster RCNN |

> `detector-torch.ipynb` is kept as a reference. All improvements it required
> are implemented in `train_fasterrcnn.py`. Run `train_fasterrcnn.py` instead.

---

## Typical Workflow

### Step 0 — Train the crop-type classifier (run once, required for two-stage pipeline)

```bash
python generate_classifier_csv.py   # build CSVs (~1 min, run once)
python train_classifier.py --dry-run  # estimate time
python train_classifier.py            # train EfficientNet-B2 (40 epochs)
```

The classifier output is saved to `outputs/classifier_output/best.pth` and is automatically
loaded by `CropClassifier()` at inference time.

### First time — YOLO26

```bash
python train.py --dry-run       # estimate time
python train.py                 # full run
open outputs/yolo_output/               # inspect figures
python app_gradio.py            # demo (uses two-stage pipeline if classifier is trained)
jupyter notebook export.ipynb   # export to mobile
```

### First time — Faster RCNN v2

```bash
python train_fasterrcnn.py --dry-run        # estimate time
python train_fasterrcnn.py                  # full run
open outputs/fasterrcnn_output/                     # inspect figures
python train_fasterrcnn.py --export-only    # re-export if needed
```

### Ablation study (for paper)

```bash
# Step 1: generate architecture diagrams immediately (no training needed)
python train_alt_fasterrcnn.py --arch-figures

# Step 2: run full ablation (trains all 7 configs sequentially, ~7× longer)
python train_alt_fasterrcnn.py --dry-run    # estimate time first
python train_alt_fasterrcnn.py              # full run

# Step 3: inspect results and regenerate figures
open outputs/alt_fasterrcnn_output/figures/
python train_alt_fasterrcnn.py --figures-only   # if you want to tweak figures
```

### First time — SE-FPN final model

```bash
python train_final.py --dry-run         # estimate time
python train_final.py                   # full run (50 epochs, EMA, SGDR)
open outputs/final_output/                      # inspect all 15 figures
python train_final.py --export-only     # re-export mobile models if needed
```

### After interruption

```bash
python train_classifier.py        # Classifier — auto-resumes from last.pth
python train.py                   # YOLO — auto-resumes from last.pt
python train_fasterrcnn.py        # Faster RCNN baseline — auto-resumes from last.pth
python train_final.py             # SE-FPN final — auto-resumes (EMA + scheduler state saved)
```

---

## Dataset

**YOLO pipeline** (`data/main/`):

- Format: YOLO `.txt` labels (one file per image, normalised XYWH)
- Source: [Ghana Crop Disease Challenge v2](https://universe.roboflow.com/ghanacropdiseasechallenge/ghana-crop-disease-challenge/dataset/2) — Roboflow, CC BY 4.0
- Splits: train / valid / test (Roboflow auto-split)

**Faster RCNN pipeline** (`dataset/`):

- Format: CSV annotations with absolute pixel XYXY bounding boxes
- `integer_label` values match `label_map.json` (1-indexed, 1 = Corn Cercospora … 23 = Tomato Mosaic)
- ~40,850 training images, ~5,837 validation images, ~11,672 test images
