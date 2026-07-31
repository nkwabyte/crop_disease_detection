"""Configuration constants for the RT-DETR detector (Ultralytics, transformer query head).

Self-contained — duplicates the YOLO-format data + class-name conventions rather than
importing from src/yolo, so the package stays decoupled.

RT-DETR is a query-based transformer detector (no anchors, no NMS). It trains on the
same YOLO-format dataset (data/main/) with the same 0-indexed class order as YOLO26.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "main"
NEG_DIR      = PROJECT_ROOT / "data" / "negatives"
RTDETR_YAML  = PROJECT_ROOT / "data_rtdetr.yaml"     # generated at train time
RUNS_DIR     = PROJECT_ROOT / "outputs" / "rtdetr_output"
EXP_NAME     = "crop_disease_rtdetr"
MODELS_DIR   = PROJECT_ROOT / "models"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

# ── Model & Hyperparameters ───────────────────────────────────────────────────
MODEL_SIZE     = "rtdetr-l"   # Ultralytics RT-DETR large (rtdetr-l.pt / rtdetr-l.yaml)
IMG_SIZE       = 640
BASE_BATCH     = 4            # RT-DETR is heavier than yolo26n; raise on the GPU server
EPOCHS_DEFAULT = 120
PATIENCE       = 25
CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45
NUM_NEGATIVES  = 300          # hard-negative images to download and stage

# ── Class Map (0-indexed, same order as data_fixed.yaml / models/yolo_metadata) ─
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
