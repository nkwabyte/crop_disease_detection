# Two-Stage Inference Pipeline

**Components:** `train_classifier.py` (Stage 1) + any detector (Stage 2)  
**Purpose:** Prevent cross-crop false positives before disease detection

---

## 1. The Problem

The YOLO and Faster RCNN detectors are trained exclusively on Corn, Pepper, and Tomato leaf images (plus OOD hard negatives). When a leaf from an entirely different plant species is presented — mango, banana, cassava — the detectors have no "unknown" class to assign. Instead, they activate on the closest visual match in their training distribution.

In practice, mango and banana leaves produce high-confidence Tomato disease detections because:
- The detectors learned that "leaf-shaped green thing" maps to one of the 23 disease classes.
- Hard-negative OOD training suppresses completely unrelated scenes (people, cars, buildings) but does not suppress other plant leaves, which are visually similar to training images.

The two-stage pipeline solves this by first classifying the crop type and rejecting images that do not confidently match Corn, Pepper, or Tomato before any detection occurs.

---

## 2. Pipeline Architecture

```
Input Image
    │
    ▼
┌───────────────────────────────────────────┐
│  Stage 1: Crop-Type Classifier            │
│  Model: EfficientNet-B2 (3 classes)       │
│  Classes: Corn / Pepper / Tomato          │
│  Threshold: 0.55 (default)                │
└───────────────────────────────────────────┘
    │
    ├── confidence < 0.55 ──────────────────► "Unknown crop — no detection"
    │
    └── confidence ≥ 0.55
            │
            ▼  crop_label + yolo_class_ids
┌───────────────────────────────────────────┐
│  Stage 2: Disease Detector                │
│  Model: YOLO26 or SE-FPN Faster RCNN      │
│  Output filtered to crop-specific classes │
└───────────────────────────────────────────┘
    │
    ▼
Disease bounding boxes + labels
(only for the detected crop type)
```

### Why this works for non-crop leaves

A mango leaf presented to the EfficientNet-B2 classifier will produce low softmax probabilities across all three classes (Corn, Pepper, Tomato) because it was trained only on those three crops. The maximum probability is likely below the 0.55 threshold, so the image is rejected before the detector ever runs.

---

## 3. Crop-to-YOLO-Class Mapping

| Classifier output | YOLO class IDs passed to detector |
|---|---|
| Corn | 0, 1, 2, 3, 4 |
| Pepper | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 |
| Tomato | 15, 16, 17, 18, 19, 20, 21, 22 |
| unknown | (empty — detector not called) |

After the detector runs, bounding boxes whose class IDs are not in the allowed set are discarded. This prevents a rare case where the detector outputs a cross-crop detection despite the classifier's correct crop-type assignment (e.g., the detector finds a patch that looks like "Tomato Leaf Curl" on a Corn leaf image).

---

## 4. Code Integration

### With YOLO26

```python
from train_classifier import CropClassifier
from ultralytics import YOLO

clf  = CropClassifier()   # auto-loads classifier_output/best.pth
yolo = YOLO("runs/crop_disease_yolo26/weights/best.pt")

def detect_disease(image_path: str, conf_thresh: float = 0.50) -> dict:
    # Stage 1: crop identification
    crop, crop_conf, allowed_ids = clf.predict(image_path)

    if crop == "unknown":
        return {
            "status":    "rejected",
            "reason":    f"Not a recognised crop leaf (confidence={crop_conf:.2f})",
            "crop":      None,
            "diseases":  [],
        }

    # Stage 2: disease detection (filtered to crop's classes)
    results  = yolo(image_path, conf=conf_thresh)[0]
    diseases = [
        {
            "class_id":  int(b.cls),
            "class_name": yolo.names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox_xyxy":  b.xyxy[0].tolist(),
        }
        for b in results.boxes
        if int(b.cls) in allowed_ids
    ]

    return {
        "status":     "detected",
        "crop":       crop,
        "crop_conf":  crop_conf,
        "diseases":   diseases,
    }
```

### With SE-FPN Faster RCNN

```python
import torch
from train_classifier import CropClassifier
from torchvision.transforms import functional as F
from PIL import Image

# YOLO IDs → Faster RCNN IDs: Faster RCNN labels are 1-indexed
# Corn IDs (1–5), Pepper (6–15), Tomato (16–23)
YOLO_TO_RCNN = {i: i + 1 for i in range(23)}
CROP_TO_RCNN_IDS = {
    "Corn":   list(range(1, 6)),
    "Pepper": list(range(6, 16)),
    "Tomato": list(range(16, 24)),
}

clf  = CropClassifier()
# (load SE-FPN model here as shown in docs/05_sefpn_final_model.md)

def detect_disease_rcnn(image_path: str, score_thresh: float = 0.50) -> dict:
    crop, crop_conf, _ = clf.predict(image_path)

    if crop == "unknown":
        return {"status": "rejected", "crop": None, "diseases": []}

    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(tensor)[0]

    allowed_ids = CROP_TO_RCNN_IDS[crop]
    keep = [
        i for i, (label, score) in enumerate(
            zip(predictions["labels"], predictions["scores"])
        )
        if label.item() in allowed_ids and score.item() >= score_thresh
    ]

    diseases = [
        {
            "class_id":   predictions["labels"][i].item(),
            "confidence": predictions["scores"][i].item(),
            "bbox_xyxy":  predictions["boxes"][i].tolist(),
        }
        for i in keep
    ]

    return {"status": "detected", "crop": crop, "crop_conf": crop_conf, "diseases": diseases}
```

---

## 5. Confidence Threshold Tuning

The classifier's `confidence_threshold` (default `0.55`) controls the trade-off between:

| Threshold | Effect |
|---|---|
| Too low (< 0.40) | Non-crop leaves (mango, banana) pass through; false detections re-appear |
| Too high (> 0.75) | Valid crop images with partial occlusion or unusual lighting get rejected |

### Recommended tuning procedure

1. Collect a small set of known non-crop leaf images and a set of valid crop leaf images.
2. Run the classifier on both sets and plot the confidence distribution.
3. Choose the threshold at the gap between the two distributions.

```python
from train_classifier import CropClassifier
from pathlib import Path

clf = CropClassifier(confidence_threshold=0.0)  # disable threshold to see raw scores

for path in Path("test_images/").glob("*.jpg"):
    crop, conf, _ = clf.predict(path)
    print(f"{path.name:<40} {crop:<8} {conf:.3f}")
```

### Changing the threshold at runtime

```python
clf = CropClassifier(confidence_threshold=0.65)
# or after construction:
clf.threshold = 0.65
```

---

## 6. Multi-Model Strategy

The pipeline is detector-agnostic — Stage 1 always returns the same `(crop, confidence, yolo_ids)` tuple regardless of which detector is used in Stage 2. This makes it easy to swap the detector:

```python
# Use YOLO for speed
clf = CropClassifier()
yolo = YOLO("runs/crop_disease_yolo26/weights/best.pt")

# Switch to SE-FPN for accuracy
clf = CropClassifier()   # same Stage 1
# load SE-FPN model as Stage 2
```

---

## 7. Training the Classifier

See [01_crop_classifier.md](01_crop_classifier.md) for full training details.

Quick start:

```bash
python generate_classifier_csv.py    # build train/valid/test CSVs
python train_classifier.py           # train EfficientNet-B2 (40 epochs, early stopping)
```

---

## 8. Expected Behaviour by Input Type

| Input | Classifier output | Action |
|---|---|---|
| Healthy corn leaf | Corn, conf ≥ 0.90 | YOLO detects `Corn_Healthy` region |
| Tomato with Late Blight | Tomato, conf ≥ 0.85 | YOLO detects `Tomato_Late_Blight` region |
| Pepper with Leaf Curl | Pepper, conf ≥ 0.80 | YOLO detects `Pepper_Leaf_Curl` region |
| Mango leaf | Unknown, conf < 0.55 | Rejected — no detection |
| Blurry or cropped leaf | May be Uncertain | If conf < threshold → rejected |
| Non-leaf image | Unknown, conf < 0.55 | Rejected — no detection |

---

## 9. Limitations

1. **False rejections on atypical crop images:** Heavily diseased leaves can look unlike training examples. If the disease dramatically changes leaf colour or texture, the classifier may reject a valid image. Lowering the threshold mitigates this.

2. **No "Other crop" class:** The classifier has no explicit class for non-target crops (mango, cassava, etc.). It rejects them via the confidence threshold, not via a dedicated class. This means a visually ambiguous non-crop leaf might occasionally exceed the threshold and be passed to the detector.

3. **Classifier and detector must agree:** If the classifier correctly identifies a Tomato but the detector was trained on a different label indexing convention, the class filter may need adjustment. Always verify `CROP_TO_YOLO_CLASSES` in `train_classifier.py` matches the detector's label map.
