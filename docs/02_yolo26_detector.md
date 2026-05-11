# YOLO26 Crop Disease Detector

**Script:** `train.py`  
**Output directory:** `yolo_output/` (figures) · `runs/crop_disease_yolo26/` (weights)  
**Role:** Primary one-stage detection baseline

---

## 1. Overview

YOLO26 is a one-stage, anchor-free object detector from the Ultralytics family. It predicts bounding boxes and class probabilities in a single forward pass, making it substantially faster than two-stage detectors (Faster RCNN) at the cost of slightly lower mAP on small or densely-packed objects.

In this project, YOLO26n (nano) is used as the primary YOLO model because:
- It fits comfortably on a 24 GB M4 Pro with batch size 32.
- Nano is fast enough for real-time field inference on a mobile device.
- For a 23-class problem where classes are visually distinct at the crop level, the capacity of a nano model is sufficient.

The script can be switched to larger variants (`yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`) by changing the `MODEL_SIZE` constant if validation mAP plateaus.

---

## 2. Architecture

| Component | Detail |
|---|---|
| Model family | YOLO26 (Ultralytics) |
| Variant used | YOLO26n (nano) |
| Detection head | Anchor-free, decoupled classification + regression heads |
| Input resolution | 640 × 640 px |
| Output classes | 23 (0-indexed) |
| Backbone | CSP (Cross Stage Partial) network |
| Neck | PANet feature pyramid |
| Pre-trained weights | `yolo26n.pt` (COCO-pretrained, provided in repo root) |

---

## 3. Pipeline Steps

Running `python train.py` executes four steps automatically:

### Step 1 — `data_fixed.yaml`

Writes an absolute-path YAML config file (`data_fixed.yaml`) from the Roboflow `data.yaml`. This is necessary because Ultralytics resolves paths relative to the working directory, which breaks when running from Kaggle or a different directory. The file is safe to regenerate and is never committed.

### Step 2 — Hard-negative staging

Downloads 300 diverse non-crop images from `picsum.photos` with a fixed random seed (42) and stages them into the training split with **empty label files**. YOLO interprets an empty label as a background-only image and learns to suppress detections on it.

This is the primary OOD guard mechanism: the model is explicitly trained to output zero detections on non-crop imagery.

- Images are downloaded concurrently using `ThreadPoolExecutor`.
- Already-downloaded files are skipped (resumable).
- The 300 images are seeded deterministically so results are reproducible.

### Step 3 — YOLO training

Trains via the Ultralytics Python API with the following configuration:

| Parameter | Value | Notes |
|---|---|---|
| Model | `yolo26n.pt` | COCO-pretrained starting point |
| Data | `data_fixed.yaml` | Absolute-path config |
| Image size | 640 | Matches dataset images |
| Batch size | 32 | Per-GPU (auto-scaled for multi-GPU) |
| Epochs | 200 | With early stopping |
| Patience | 25 | Stops after 25 non-improving epochs |
| Optimizer | Auto (Ultralytics default — AdamW → SGD) | |
| LR schedule | Cosine decay | Built-in Ultralytics schedule |
| Augmentation | Mosaic, MixUp, HSV, flips, scale | Ultralytics built-in suite |
| Label smoothing | 0.1 | Reduces overconfidence on hard classes |
| Confidence threshold | 0.50 | Applied at inference time |
| IoU threshold | 0.45 | NMS IoU threshold |
| Hard negatives | 300 images | Staged with empty labels |
| Cache | disk | Avoids repeated JPEG decoding |
| Workers | 0 on MPS, 8 on CUDA | Metal incompatibility workaround |
| AMP | True on CUDA | Disabled on MPS (fp16 index bug in TAL assigner) |

### Step 4 — Publication figures

Generates up to 10 PNG figures to `yolo_output/` (see Section 7).

---

## 4. MPS-Specific Patch

The Ultralytics TAL (Task-Aligned Learning) assigner performs tensor indexing operations that are incompatible with MPS fp16. The script patches this automatically:

```python
def _patch_tal_for_mps():
    # Moves the TAL assigner forward pass to CPU to avoid the MPS fp16 index bug.
    # The patch wraps the existing forward method and moves all input tensors to
    # CPU before the call, then moves outputs back to the original device.
```

This means that on Apple Silicon, the TAL assignment step runs on CPU while all other operations (backbone, head, loss) run on MPS. The performance penalty is minimal because TAL accounts for a small fraction of total compute time.

---

## 5. Resume Behaviour

YOLO uses Ultralytics' native resume mechanism. The experiment is identified by name (`crop_disease_yolo26`). If `runs/crop_disease_yolo26/weights/last.pt` exists, training resumes automatically on the next run.

```bash
python train.py                    # resumes if last.pt exists
python train.py --skip-negatives   # skip download if negatives already staged
```

To force a fresh start:

```bash
rm -rf runs/crop_disease_yolo26
python train.py
```

---

## 6. All Commands

| Command | Purpose |
|---|---|
| `python train.py` | Full pipeline (steps 1–4) |
| `python train.py --dry-run` | 1-epoch timing estimate; reports projected total time |
| `python train.py --skip-negatives` | Skip hard-negative download (already staged) |
| `python train.py --figures-only` | Regenerate all figures without re-training |
| `python train.py --no-figures` | Train only; skip figure generation |
| `python train.py --epochs 100` | Override epoch count |
| `DRY_RUN=1 python train.py` | Dry-run via environment variable |

---

## 7. Output Files

### Weights — `runs/crop_disease_yolo26/weights/`

| File | Contents |
|---|---|
| `best.pt` | Best validation mAP checkpoint |
| `last.pt` | Final epoch checkpoint (used for resume) |
| `epoch0.pt`, `epoch10.pt`, … | Periodic saves every 10 epochs |

### Figures — `yolo_output/`

| File | Generated | Contents |
|---|---|---|
| `fig_01_dataset_overview.png` | Always | Images per split, hard-negative ratio, box counts |
| `fig_02_class_distribution_train.png` | Always | Per-class annotation count (training set) |
| `fig_03_cross_split_distribution.png` | Always | Normalised class distribution across splits |
| `fig_04_annotation_density.png` | Always | Boxes-per-image histogram (train + val) |
| `fig_05_bbox_spatial_heatmap.png` | Always | Bounding-box centre density maps per crop type |
| `fig_06_bbox_size_analysis.png` | Always | Width/height scatter, aspect ratio, area by crop |
| `fig_07_class_imbalance.png` | Always | Val/Test vs Train frequency ratio per class |
| `fig_08_training_config.png` | Always | Hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | Always | LR cosine schedule + augmentation profile |
| `fig_10_training_metrics.png` | After training | Loss and mAP curves from `results.csv` |

### Additional run outputs — `runs/crop_disease_yolo26/`

Ultralytics also saves confusion matrices, P/R curves, validation batch previews, and a `results.csv` with per-epoch metrics.

---

## 8. Inference

Load the best checkpoint with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("runs/crop_disease_yolo26/weights/best.pt")
results = model("leaf_image.jpg", conf=0.50, iou=0.45)

for box in results[0].boxes:
    cls_id = int(box.cls)
    conf   = float(box.conf)
    xyxy   = box.xyxy[0].tolist()
    print(f"Class {cls_id}: {conf:.2f}  {xyxy}")
```

For the two-stage pipeline, use the `CropClassifier` to gate detection first:

```python
from train_classifier import CropClassifier
from ultralytics import YOLO

clf  = CropClassifier()
yolo = YOLO("runs/crop_disease_yolo26/weights/best.pt")

crop, conf, allowed_ids = clf.predict("leaf.jpg")
if crop != "unknown":
    results  = yolo("leaf.jpg")[0]
    filtered = [b for b in results.boxes if int(b.cls) in allowed_ids]
```

See [06_two_stage_pipeline.md](06_two_stage_pipeline.md) for full integration details.

---

## 9. Class Labels

| YOLO ID | Class |
|---|---|
| 0 | Corn_Cercospora_Leaf_Spot |
| 1 | Corn_Common_Rust |
| 2 | Corn_Healthy |
| 3 | Corn_Northern_Leaf_Blight |
| 4 | Corn_Streak |
| 5 | Pepper_Bacterial_Spot |
| 6 | Pepper_Cercospora |
| 7 | Pepper_Early_Blight |
| 8 | Pepper_Fusarium |
| 9 | Pepper_Healthy |
| 10 | Pepper_Late_Blight |
| 11 | Pepper_Leaf_Blight |
| 12 | Pepper_Leaf_Curl |
| 13 | Pepper_Leaf_Mosaic |
| 14 | Pepper_Septoria |
| 15 | Tomato_Bacterial_Spot |
| 16 | Tomato_Early_Blight |
| 17 | Tomato_Fusarium |
| 18 | Tomato_Healthy |
| 19 | Tomato_Late_Blight |
| 20 | Tomato_Leaf_Curl |
| 21 | Tomato_Mosaic |
| 22 | Tomato_Septoria |

---

## 10. References

- Ultralytics YOLO documentation: https://docs.ultralytics.com
- Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767.
- Ghana Crop Disease Challenge dataset: CC BY 4.0, Roboflow Universe.
