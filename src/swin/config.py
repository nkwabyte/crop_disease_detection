"""Configuration constants for the Swin detector (Swin-V2-T backbone + FPN + Faster R-CNN head).

Self-contained — duplicates the shared dataset / class-name conventions rather than
importing them, so `src/swin/` stays decoupled from the other model packages.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"

OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "swin_output"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
MODELS_DIR   = OUTPUT_DIR / "models"
METRICS_FILE = OUTPUT_DIR / "metrics_history.json"
FINAL_EVAL_FILE = OUTPUT_DIR / "final_eval.json"

TRAIN_CSV = DATASET_DIR / "final_train_labels.csv"
VAL_CSV   = DATASET_DIR / "final_validate_labels.csv"
TEST_CSV  = DATASET_DIR / "final_test_labels.csv"

TRAIN_IMG_DIR = DATASET_DIR / "train"
VAL_IMG_DIR   = DATASET_DIR / "validate"
TEST_IMG_DIR  = DATASET_DIR / "test"

# ── Model Settings ────────────────────────────────────────────────────────────
NUM_CLASSES = 24        # 23 disease classes (1–23) + background (0)
IMG_SIZE    = 640

# Swin backbone (Swin-V2-T, torchvision, ImageNet-pretrained). Unlike ViT, Swin uses
# windowed attention with relative position bias, so it handles variable input sizes
# and produces a genuine multi-scale feature pyramid (/4, /8, /16, /32).
SWIN_VARIANT       = "swin_v2_t"
FPN_OUT_CHANNELS   = 256
SWIN_IN_CHANNELS   = [96, 192, 384, 768]   # stage output channels (swin_v2_t)

# ── Hyperparameter Defaults ────────────────────────────────────────────────────
EPOCHS_DEFAULT         = 40
PATIENCE_DEFAULT       = 10
BATCH_SIZE             = 2     # swin_v2_t (~28M) is lighter than ViT-B; raise on CUDA
ACCUM_STEPS            = 4     # effective batch = BATCH_SIZE × ACCUM_STEPS
LR0                    = 1e-4  # AdamW peak LR
WEIGHT_DECAY           = 1e-4
WARMUP_EPOCHS          = 3
FREEZE_BACKBONE_EPOCHS = 5
GRAD_CLIP              = 5.0
EVAL_EVERY             = 3
NUM_NEGATIVES          = 200

# ── Inference defaults ─────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45

# ── Class Names (1-indexed, matching dataset/label_map.json / integer_label) ────
CLASS_NAMES = [
    "",                              # 0  background
    "Corn Cercospora Leaf Spot",     # 1
    "Corn Common Rust",              # 2
    "Corn Healthy",                  # 3
    "Corn Streak",                   # 4
    "Corn Northern Leaf Blight",     # 5
    "Pepper Leaf Curl",              # 6
    "Pepper Cercospora",             # 7
    "Pepper Leaf Blight",            # 8
    "Pepper Bacterial Spot",         # 9
    "Pepper Leaf Mosaic",            # 10
    "Pepper Healthy",                # 11
    "Pepper Fusarium",               # 12
    "Pepper Septoria",               # 13
    "Pepper Late Blight",            # 14
    "Pepper Early Blight",           # 15
    "Tomato Late Blight",            # 16
    "Tomato Early Blight",           # 17
    "Tomato Bacterial Spot",         # 18
    "Tomato Septoria",               # 19
    "Tomato Fusarium",               # 20
    "Tomato Leaf Curl",              # 21
    "Tomato Healthy",                # 22
    "Tomato Mosaic",                 # 23
]
CLASS_NAMES_DISPLAY = CLASS_NAMES[1:]
