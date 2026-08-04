"""Configuration constants for YOLO26 crop disease detector."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "yolo"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"
FIXED_YAML   = PROJECT_ROOT / "data_fixed.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "yolo_output"
EXP_NAME     = "crop_disease_yolo26"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Model & Hyperparameters ───────────────────────────────────────────────────
MODEL_SIZE     = MODELS_DIR / "yolo26n.pt"   # pretrained weights saved in models/
IMG_SIZE       = 640
BASE_BATCH     = 32          # per-GPU default on MPS/CPU; 32 × yolo26n fits in 24 GB
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
