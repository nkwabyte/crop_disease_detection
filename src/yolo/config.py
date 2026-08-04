"""Configuration constants for YOLO26 crop disease detector."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "yolo"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"
FIXED_YAML   = PROJECT_ROOT / "data_fixed.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "yolo_output"
EXP_NAME     = os.environ.get("YOLO_EXP", "crop_disease_yolo26")
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Model & Hyperparameters ───────────────────────────────────────────────────
# PENDING EXPERIMENT — see docs/08_next_steps.md § "PENDING — run these on the next
# GPU box". Two independent yolo26n runs plateaued at mAP50 ~0.277, and the split
# holds only ~124 training images per disease class, so capacity may not be what is
# limiting. Sweep n -> s -> m on the next full run and judge on mAP per MB: the app
# already carries a 29 MB classifier, and this detector's .pte is currently 9.3 MB.
# YOLO_MODEL / YOLO_EXP let a capacity sweep run without editing this file, and —
# critically — give each variant its own runs/ directory so the sweep does not
# overwrite the previous variant's weights.
MODEL_SIZE     = Path(os.environ.get("YOLO_MODEL", MODELS_DIR / "yolo26n.pt"))
IMG_SIZE       = 640
BASE_BATCH     = 32          # per-GPU default on MPS/CPU; 32 × yolo26n fits in 24 GB
# Measured on an RTX 5090: batch 32 left the GPU at 27 % utilisation and 6.9 GB of
# 31.4 GB. yolo26n is far too small to occupy a modern card at this batch — raise it
# and watch nvidia-smi. Re-measure per box; a 24 GB A5000 has less headroom than the
# 32 GB card these numbers came from.
CUDA_BATCH     = 64          # per-GPU default when a CUDA GPU is detected (96 GB RTX PRO 6000)
EPOCHS_DEFAULT = 200
PATIENCE       = 25
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45
NUM_NEGATIVES  = 300         # hard-negative images to download and stage

# ── Class Map ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Corn_Cercospora_Leaf_Spot", "Corn_Common_Rust", "Corn_Healthy",
    "Corn_Northern_Leaf_Blight", "Corn_Streak",
    "Pepper_Bacterial_Spot", "Pepper_Cercospora", "Pepper_Early_Blight",
    "Pepper_Fusarium", "Pepper_Healthy", "Pepper_Late_Blight",
    "Pepper_Leaf_Blight", "Pepper_Leaf_Curl", "Pepper_Leaf_Mosaic",
    "Pepper_Septoria",
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Fusarium",
    "Tomato_Healthy", "Tomato_Late_Blight", "Tomato_Leaf_Curl",
    "Tomato_Mosaic", "Tomato_Septoria",
]
