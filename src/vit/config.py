"""Configuration constants for the ViTDet detector (ViT-B/16 backbone + Faster R-CNN head).

This module is intentionally self-contained — it duplicates the shared dataset /
class-name conventions used by the Faster R-CNN pipeline rather than importing them,
so the `src/vit/` package stays decoupled from `src/fasterrcnn/`. Changing or deleting
one model's package never affects the other.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"

OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "vit_output"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
MODELS_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR  = OUTPUT_DIR                      # figures written flat, like the other pipelines
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
IMG_SIZE    = 640       # fixed square input — ViT positional embeddings are size-specific

# ViT backbone (ViT-B/16, torchvision, ImageNet-pretrained)
VIT_VARIANT      = "vit_b_16"
PATCH_SIZE       = 16
VIT_HIDDEN_DIM   = 768
FPN_OUT_CHANNELS = 256   # 1×1 projection of the 768-dim tokens feeding the detection head

# ── Hyperparameter Defaults ────────────────────────────────────────────────────
# ViT backbones fine-tune best with AdamW + a low LR (not the SGD 5e-3 used by the
# ResNet Faster R-CNN). These defaults reflect that.
EPOCHS_DEFAULT         = 40
PATIENCE_DEFAULT       = 10
BATCH_SIZE             = 2     # default on MPS/CPU (M4 Pro 24 GB) — ViT-B at 640px is heavy
CUDA_BATCH_SIZE        = 8     # default when a CUDA GPU is detected (e.g. 96 GB RTX PRO 6000)
ACCUM_STEPS            = 4     # effective batch = BATCH_SIZE × ACCUM_STEPS
LR0                    = 1e-4  # AdamW peak LR
WEIGHT_DECAY           = 1e-4
WARMUP_EPOCHS          = 3
FREEZE_BACKBONE_EPOCHS = 5     # train only the head + projection first, then unfreeze the ViT
GRAD_CLIP              = 5.0
EVAL_EVERY             = 3
NUM_NEGATIVES          = 200

# ── Inference defaults ─────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45

# ── Class Names (1-indexed, matching dataset/label_map.json and the ─────────────
#    integer_label column of the annotation CSVs the model actually trains on) ───
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
CLASS_NAMES_DISPLAY = CLASS_NAMES[1:]   # 23 disease names, 0-indexed
