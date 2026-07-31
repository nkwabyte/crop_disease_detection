"""Configuration constants for Faster R-CNN models (baseline, ablation, and SE-FPN final model)."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"

OUTPUT_DIR_BASELINE = PROJECT_ROOT / "outputs" / "fasterrcnn_output"
OUTPUT_DIR_ALT      = PROJECT_ROOT / "outputs" / "alt_fasterrcnn_output"
OUTPUT_DIR_FINAL    = PROJECT_ROOT / "outputs" / "final_output"

TRAIN_CSV     = DATASET_DIR / "final_train_labels.csv"
VAL_CSV       = DATASET_DIR / "final_validate_labels.csv"
TEST_CSV      = DATASET_DIR / "final_test_labels.csv"

TRAIN_IMG_DIR = DATASET_DIR / "train"
VAL_IMG_DIR   = DATASET_DIR / "validate"
TEST_IMG_DIR  = DATASET_DIR / "test"

# ── Model Settings ────────────────────────────────────────────────────────────
NUM_CLASSES = 24       # 23 disease classes (1–23) + background (0)
IMG_SIZE    = 640

# ── Hyperparameter Defaults ────────────────────────────────────────────────────
EPOCHS_DEFAULT        = 30
PATIENCE_DEFAULT      = 8
BATCH_SIZE            = 4    # safe on 24 GB unified memory
ACCUM_STEPS           = 2    # gradient accumulation steps for SE-FPN
LR0                   = 5e-3
WEIGHT_DECAY          = 5e-4
MOMENTUM              = 0.9
WARMUP_EPOCHS         = 3
FREEZE_BACKBONE_EPOCHS= 5
GRAD_CLIP             = 10.0
EVAL_EVERY            = 5
NUM_NEGATIVES         = 200

# ── Class Names (1-indexed matching dataset/label_map.json) ───────────────────
CLASS_NAMES = [
    "",                              # 0  background
    "Corn Cercospora Leaf Spot",     # 1
    "Corn Common Rust",              # 2
    "Corn Healthy",                  # 3
    "Corn Northern Leaf Blight",     # 4
    "Corn Streak",                   # 5
    "Pepper Bacterial Spot",         # 6
    "Pepper Cercospora",             # 7
    "Pepper Early Blight",           # 8
    "Pepper Fusarium",               # 9
    "Pepper Healthy",                # 10
    "Pepper Late Blight",            # 11
    "Pepper Leaf Blight",            # 12
    "Pepper Leaf Curl",              # 13
    "Pepper Leaf Mosaic",            # 14
    "Pepper Septoria",               # 15
    "Tomato Bacterial Spot",         # 16
    "Tomato Early Blight",           # 17
    "Tomato Fusarium",               # 18
    "Tomato Healthy",                # 19
    "Tomato Late Blight",            # 20
    "Tomato Leaf Curl",              # 21
    "Tomato Mosaic",                 # 22
    "Tomato Septoria",               # 23
]
