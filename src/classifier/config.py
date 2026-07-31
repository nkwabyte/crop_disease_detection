"""Configuration constants for crop-type classifier (EfficientNet-B2)."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DATASET_DIR   = PROJECT_ROOT / "dataset"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "classifier_output"
FIGURES_DIR   = OUTPUT_DIR / "figures"

# ── Model & Class Settings ────────────────────────────────────────────────────
CROP_CLASSES  = ["Corn", "Pepper", "Tomato"]
NUM_CLASSES   = len(CROP_CLASSES)       # 3
IMG_SIZE      = 260                     # EfficientNet-B2 native resolution

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS_DEFAULT = 40
BATCH_DEFAULT  = 64                     # M4 Pro 24 GB handles batch-64 with headroom
CUDA_BATCH     = 128                    # default when a CUDA GPU is detected (96 GB RTX PRO 6000)
LR_DEFAULT     = 1e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 8                      # early-stopping patience (epochs)
CONF_DEFAULT   = 0.55                   # min softmax confidence to accept a crop
GRAD_CLIP      = 1.0                    # max gradient norm for training stability

# ── Pipeline Mapping ──────────────────────────────────────────────────────────
# Corn classes 0–4, Pepper 5–14, Tomato 15–22 (for pipeline integration)
CROP_TO_YOLO_CLASSES: dict[str, list[int]] = {
    "Corn":   list(range(0, 5)),
    "Pepper": list(range(5, 15)),
    "Tomato": list(range(15, 23)),
}
