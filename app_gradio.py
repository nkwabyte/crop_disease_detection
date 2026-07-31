"""
Crop Disease Detection — Gradio Demo (Two-Stage Pipeline)
──────────────────────────────────────────────────────────
Run with:
    python app_gradio.py
  or
    gradio app_gradio.py

Requires:  pip install -r requirements.txt

Stage 1:  EfficientNet-B2 classifier  →  outputs/classifier_output/best.pth
Stage 2:  YOLO26 disease detector     →  runs/crop_disease_yolo26/weights/best.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the classifier — gracefully degrade if train_classifier is unavailable
try:
    from train_classifier import CropClassifier
    _CLF_AVAILABLE = True
except Exception:
    _CLF_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RUNS_DIR     = PROJECT_ROOT / "runs"
EXP_NAME     = "crop_disease_yolo26"
DEFAULT_PT   = RUNS_DIR / EXP_NAME / "weights" / "best.pt"
DEFAULT_CLF  = PROJECT_ROOT / "outputs" / "classifier_output" / "best.pth"

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

# One colour per class (R, G, B)
PALETTE = [
    (231, 76,  60),  (192, 57,  43),  (46,  204, 113), (39,  174, 96),
    (52,  152, 219), (41,  128, 185), (155, 89,  182), (142, 68,  173),
    (243, 156, 18),  (230, 126, 34),  (26,  188, 156), (22,  160, 133),
    (52,  73,  94),  (44,  62,  80),  (127, 140, 141), (149, 165, 166),
    (211, 84,  0),   (192, 57,  43),  (39,  174, 96),  (41,  128, 185),
    (142, 68,  173), (243, 156, 18),  (22,  160, 133),
]

CROP_ICONS = {"Corn": "🌽", "Pepper": "🫑", "Tomato": "🍅"}
CROP_COLORS = {"Corn": "#f0a500", "Pepper": "#27ae60", "Tomato": "#e74c3c"}


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

INFER_DEVICE = _resolve_device()


# ──────────────────────────────────────────────────────────────────────────────
# Model caches
# ──────────────────────────────────────────────────────────────────────────────

_yolo_cache: dict[str, YOLO] = {}
_clf_cache:  dict[str, CropClassifier] = {}  # type: ignore[type-arg]


def get_yolo(model_path: str) -> YOLO:
    if model_path not in _yolo_cache:
        _yolo_cache[model_path] = YOLO(model_path)
    return _yolo_cache[model_path]


def get_classifier(clf_path: str, threshold: float) -> "CropClassifier | None":
    """Return a cached CropClassifier, or None if unavailable."""
    if not _CLF_AVAILABLE:
        return None
    path = Path(clf_path)
    if not path.exists():
        return None
    key = f"{clf_path}:{threshold:.4f}"
    if key not in _clf_cache:
        try:
            _clf_cache[key] = CropClassifier(
                checkpoint=path,
                confidence_threshold=threshold,
            )
        except Exception:
            return None
    else:
        # Update threshold in-place if it changed
        _clf_cache[key].threshold = threshold
    return _clf_cache[key]


def _find_yolo() -> Path | None:
    if DEFAULT_PT.exists():
        return DEFAULT_PT
    candidates = sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_yolo(
    pil_img: Image.Image,
    conf: float,
    iou: float,
    model_path: str,
    allowed_ids: list[int] | None = None,
) -> tuple[list[dict], float]:
    """Run YOLO and return detections filtered to allowed_ids (all if None)."""
    model = get_yolo(model_path)
    arr   = np.array(pil_img.convert("RGB"))
    t0    = time.perf_counter()
    results = model.predict(source=arr, conf=conf, iou=iou, verbose=False,
                            device=INFER_DEVICE)
    elapsed = time.perf_counter() - t0
    r = results[0]
    detections = []
    if r.boxes and len(r.boxes):
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if allowed_ids is not None and cls_id not in allowed_ids:
                continue
            cf     = float(box.conf[0])
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            crop   = CLASS_NAMES[cls_id].split("_")[0]
            detections.append({
                "class_id"  : cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "crop"      : crop,
                "icon"      : CROP_ICONS.get(crop, "🌿"),
                "confidence": cf,
                "bbox"      : (x1, y1, x2, y2),
                "color"     : PALETTE[cls_id],
            })
    return sorted(detections, key=lambda d: -d["confidence"]), elapsed


def get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Locate the best available TrueType font across OS platforms, fallback to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",              # macOS
        "/System/Library/Fonts/SFCompact.ttf",              # macOS alternative
        "/System/Library/Fonts/Cache/Arial.ttf",            # macOS alternative
        "/Library/Fonts/Arial.ttf",                         # macOS alternative
        "C:\\Windows\\Fonts\\arial.ttf",                     # Windows
        "C:\\Windows\\Fonts\\segoeui.ttf",                   # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Ubuntu/Debian)
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",              # Linux (Arch)
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(pil_img: Image.Image, detections: list[dict], thickness: int = 3) -> Image.Image:
    w, h = pil_img.size
    base_dim = min(w, h)
    
    # Scale thickness, font size, and corner accents relative to image resolution
    scaled_thickness = max(2, int(base_dim / 250))
    font_size = max(11, int(base_dim / 45))
    corner_len = max(10, int(base_dim * 0.04))
    corner_thickness = scaled_thickness + 2
    
    font = get_font(font_size)
    out  = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        r, g, b = det["color"]
        
        # 1. Subtle, clean semi-transparent box region fill
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 15))
        
        # 2. Thin bounding box border (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 180), width=scaled_thickness)
        
        # 3. Styled "Camera Focus / Target" Corner Brackets (L-shapes)
        # Top-left corner
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=(r, g, b, 255), width=corner_thickness)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=(r, g, b, 255), width=corner_thickness)
        # Top-right corner
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=(r, g, b, 255), width=corner_thickness)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=(r, g, b, 255), width=corner_thickness)
        # Bottom-left corner
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=(r, g, b, 255), width=corner_thickness)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=(r, g, b, 255), width=corner_thickness)
        # Bottom-right corner
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=(r, g, b, 255), width=corner_thickness)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=(r, g, b, 255), width=corner_thickness)

        # 4. Premium rounded pill text badge
        label = f" {det['icon']} {det['class_name'].replace('_', ' ')}  {det['confidence']:.2f} "
        
        # Measure label dimensions to size the pill perfectly
        text_bb = draw.textbbox((0, 0), label, font=font)
        text_w = text_bb[2] - text_bb[0]
        text_h = text_bb[3] - text_bb[1]
        
        pad_x = max(6, int(font_size * 0.4))
        pad_y = max(4, int(font_size * 0.25))
        
        pill_w = text_w + 2 * pad_x
        pill_h = text_h + 2 * pad_y
        
        # Dynamic label placement (flip inside the box if it clips at the top edge of image)
        if y1 - pill_h - 2 >= 0:
            ty1 = int(y1) - pill_h - 2
            ty2 = int(y1) - 2
        else:
            ty1 = int(y1) + 2
            ty2 = int(y1) + pill_h + 2
            
        tx1 = int(x1)
        tx2 = int(x1) + pill_w
        
        # Keep pill within horizontal limits
        if tx2 > w:
            tx1 = max(0, w - pill_w)
            tx2 = w

        # Draw rounded pill badge with semi-transparent white highlight border
        radius = max(3, int(pill_h / 4))
        draw.rounded_rectangle([tx1, ty1, tx2, ty2], radius=radius, fill=(r, g, b, 230), outline=(255, 255, 255, 100), width=1)
        
        # Render text vertically/horizontally centered
        draw.text((tx1 + pad_x - text_bb[0], ty1 + pad_y - text_bb[1]), label, fill=(255, 255, 255, 255), font=font)

    return out


def draw_rejected(pil_img: Image.Image, crop_conf: float) -> Image.Image:
    """Return the image with a beautiful glassmorphism-style warning card in the center."""
    out  = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    base_dim = min(w, h)
    
    # 1. Subtle overall red alert tint border
    border_w = max(4, int(base_dim / 100))
    draw.rectangle([0, 0, w, h], outline=(231, 76, 60, 200), width=border_w)
    draw.rectangle([0, 0, w, h], fill=(231, 76, 60, 15))
    
    # 2. Dynamic font sizes
    font_size_big = max(16, int(base_dim / 16))
    font_size_small = max(11, int(base_dim / 26))
    
    font_big = get_font(font_size_big)
    font_small = get_font(font_size_small)

    msg1 = "UNKNOWN CROP — REJECTED"
    msg2 = f"Classifier Confidence: {crop_conf:.1%}"
    msg3 = "Does not match Corn, Pepper, or Tomato"
    
    # Measure text blocks
    bb1 = draw.textbbox((0, 0), msg1, font=font_big)
    bb2 = draw.textbbox((0, 0), msg2, font=font_small)
    bb3 = draw.textbbox((0, 0), msg3, font=font_small)
    
    w1, h1 = bb1[2] - bb1[0], bb1[3] - bb1[1]
    w2, h2 = bb2[2] - bb2[0], bb2[3] - bb2[1]
    w3, h3 = bb3[2] - bb3[0], bb3[3] - bb3[1]
    
    # Card padding
    card_padding_x = max(20, int(base_dim * 0.06))
    card_padding_y = max(16, int(base_dim * 0.05))
    
    max_text_w = max(w1, w2, w3)
    card_w = max_text_w + 2 * card_padding_x
    card_h = h1 + h2 + h3 + 2 * card_padding_y + max(12, int(base_dim * 0.03))
    
    # Clamp card size to fit inside image bounds
    card_w = min(card_w, int(w * 0.9))
    card_h = min(card_h, int(h * 0.9))
    
    # Center bounds
    cx1 = (w - card_w) // 2
    cy1 = (h - card_h) // 2
    cx2 = cx1 + card_w
    cy2 = cy1 + card_h
    
    # Draw central obsidian warning card
    radius = max(6, int(card_h / 10))
    draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=radius, fill=(15, 23, 42, 230), outline=(231, 76, 60, 255), width=2)
    
    # Text offsets inside the warning card
    y_cursor = cy1 + card_padding_y
    
    # Msg 1
    tx1 = cx1 + (card_w - w1) // 2
    draw.text((tx1 - bb1[0], y_cursor - bb1[1]), msg1, fill=(231, 76, 60, 255), font=font_big)
    
    y_cursor += h1 + max(8, int(base_dim * 0.015))
    
    # Msg 2
    tx2 = cx1 + (card_w - w2) // 2
    draw.text((tx2 - bb2[0], y_cursor - bb2[1]), msg2, fill=(255, 255, 255, 240), font=font_small)
    
    y_cursor += h2 + max(6, int(base_dim * 0.01))
    
    # Msg 3
    tx3 = cx1 + (card_w - w3) // 2
    draw.text((tx3 - bb3[0], y_cursor - bb3[1]), msg3, fill=(143, 163, 177, 220), font=font_small)
    
    return out


def make_summary_table(detections: list[dict]) -> list[list]:
    return [
        [
            f"{d['icon']} {d['class_name'].replace('_', ' ')}",
            d["crop"],
            f"{d['confidence']:.1%}",
            f"({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}) → ({d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})",
        ]
        for d in detections
    ]


def make_conf_chart(detections: list[dict], conf_threshold: float = 0.5) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(2, len(detections) * 0.5)))
    
    # Use transparent background to integrate seamlessly with the theme CSS
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    if detections:
        names  = [d["class_name"].replace("_", " ") for d in detections]
        confs  = [d["confidence"] for d in detections]
        colors = [tuple(c / 255 for c in d["color"]) for d in detections]
        bars   = ax.barh(names, confs, color=colors, height=0.6, edgecolor="none")
        ax.set_xlim(0, 1)
        ax.axvline(conf_threshold, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.8,
                   label=f"Detection threshold ({conf_threshold:.2f})")
        for bar, conf in zip(bars, confs):
            ax.text(
                min(conf + 0.02, 0.93),
                bar.get_y() + bar.get_height() / 2,
                f"{conf:.0%}", va="center", ha="left",
                color="#eaeaea", fontsize=10, fontweight="bold",
            )
        ax.legend(fontsize=9, facecolor="#0d1b2a", labelcolor="#aaaaaa", framealpha=0.8, edgecolor="#2c2c4e")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                color="#7f8c8d", fontsize=14, transform=ax.transAxes)

    ax.set_xlabel("Confidence Score", color="#8fa3b1", fontsize=10)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.spines[:].set_color("#2c2c4e")
    ax.xaxis.grid(True, color="#2c2c4e", linewidth=0.8)
    plt.tight_layout(pad=1.5)
    return fig


def _stage1_html(crop: str, crop_conf: float) -> str:
    """Render the Stage 1 result banner."""
    color = CROP_COLORS.get(crop, "#e74c3c")
    icon  = CROP_ICONS.get(crop, "❓")
    return f"""
    <div style="
        display:flex; align-items:center; gap:12px;
        padding:10px 18px; border-radius:10px;
        background:rgba(255,255,255,0.04);
        border:1px solid {color}55;
        font-size:0.95rem; color:#dde6f0;">
      <span style="font-size:1.6rem;">{icon}</span>
      <div>
        <span style="color:{color}; font-weight:700;">{crop}</span>
        <span style="color:#8fa3b1; margin-left:6px;">identified by classifier</span>
      </div>
      <div style="margin-left:auto; text-align:right;">
        <span style="
            background:{color}22; color:{color};
            border:1px solid {color}55;
            border-radius:20px; padding:2px 12px;
            font-size:0.85rem; font-weight:600;">
          {crop_conf:.1%} confidence
        </span>
      </div>
    </div>"""


def _rejected_html(crop_conf: float, threshold: float) -> str:
    return f"""
    <div style="
        display:flex; align-items:center; gap:12px;
        padding:10px 18px; border-radius:10px;
        background:rgba(231,76,60,0.07);
        border:1px solid rgba(231,76,60,0.4);
        font-size:0.95rem; color:#dde6f0;">
      <span style="font-size:1.6rem;">🚫</span>
      <div>
        <span style="color:#e74c3c; font-weight:700;">Unknown crop — rejected</span>
        <span style="color:#8fa3b1; margin-left:6px;">
          Classifier max confidence {crop_conf:.1%} is below threshold {threshold:.0%}.
          This does not appear to be a Corn, Pepper, or Tomato leaf.
        </span>
      </div>
    </div>"""


def _no_clf_html() -> str:
    return """
    <div style="
        padding:8px 18px; border-radius:10px;
        background:rgba(243,156,18,0.07);
        border:1px solid rgba(243,156,18,0.35);
        font-size:0.88rem; color:#f0c060;">
      ⚠️  Stage 1 classifier not loaded — running disease detection without crop filter.
      Train the classifier first: <code>python train_classifier.py</code>
    </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main predict function
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    image: Image.Image | None,
    clf_threshold: float,
    conf_threshold: float,
    iou_threshold: float,
    model_path: str,
    clf_path: str,
) -> tuple:
    """
    Returns: annotated_image, stage1_html, summary_table, conf_chart, status_text
    """
    placeholder = Image.new("RGB", (640, 400), (20, 30, 48))

    if image is None:
        draw = ImageDraw.Draw(placeholder)
        draw.text((160, 185), "Upload a crop leaf image to begin  ☝️", fill=(100, 130, 160))
        return placeholder, "", [], make_conf_chart([], conf_threshold), "⏳ Awaiting image…"

    # ── Stage 1: crop-type classifier ─────────────────────────────────────────
    clf = get_classifier(clf_path, clf_threshold)

    if clf is None:
        # Classifier not available — skip filtering, warn user
        allowed_ids = None
        stage1_html = _no_clf_html()
        crop_label  = None
    else:
        crop_label, crop_conf, allowed_ids = clf.predict(image)

        if crop_label == "unknown":
            annotated  = draw_rejected(image, crop_conf)
            stage1_html = _rejected_html(crop_conf, clf_threshold)
            status = (
                f"🚫  Rejected by Stage 1 classifier — "
                f"max confidence {crop_conf:.1%} < threshold {clf_threshold:.0%}.  "
                "Not a recognised crop leaf."
            )
            return annotated, stage1_html, [], make_conf_chart([], conf_threshold), status

        stage1_html = _stage1_html(crop_label, crop_conf)

    # ── Stage 2: disease detector ─────────────────────────────────────────────
    yolo_file = Path(model_path)
    if not yolo_file.exists():
        err_img = Image.new("RGB", (640, 200), (40, 10, 10))
        draw    = ImageDraw.Draw(err_img)
        draw.text((20, 85), f"YOLO model not found: {model_path}", fill=(231, 76, 60))
        return err_img, stage1_html, [], make_conf_chart([], conf_threshold), f"❌ YOLO model not found: {model_path}"

    detections, elapsed = run_yolo(image, conf_threshold, iou_threshold,
                                   model_path, allowed_ids)
    annotated = draw_boxes(image, detections)
    table     = make_summary_table(detections)
    chart     = make_conf_chart(detections, conf_threshold)

    crop_tag = f"  [{CROP_ICONS.get(crop_label, '')} {crop_label}]" if crop_label else ""

    if detections:
        top    = detections[0]
        n      = len(detections)
        status = (
            f"✅  {n} detection{'s' if n > 1 else ''} found{crop_tag}  ·  "
            f"Top: {top['icon']} {top['class_name'].replace('_', ' ')} "
            f"({top['confidence']:.1%})  ·  "
            f"Inference: {elapsed * 1000:.0f} ms  [{INFER_DEVICE}]"
        )
    else:
        status = (
            f"🔍  No disease detected above {conf_threshold:.0%} confidence{crop_tag}.  "
            f"Inference: {elapsed * 1000:.0f} ms  [{INFER_DEVICE}]"
        )

    return annotated, stage1_html, table, chart, status


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #0d1b2a, #1a2840) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto; }

.title-text {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #2ecc71, #3498db, #9b59b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-text { color: #8fa3b1; font-size: 1rem; }

.status-box textarea, .status-box input {
    background: rgba(46, 204, 113, 0.08) !important;
    border: 1px solid rgba(46, 204, 113, 0.2) !important;
    color: #2ecc71 !important;
    font-size: 0.9rem !important;
    font-weight: 500;
}

.tab-nav button {
    background: rgba(255,255,255,0.04) !important;
    color: #8fa3b1 !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    transition: all 0.2s;
}
.tab-nav button.selected {
    background: rgba(46, 204, 113, 0.15) !important;
    color: #2ecc71 !important;
    border-bottom-color: transparent !important;
}

.block { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }
input[type=range]::-webkit-slider-thumb { background: #2ecc71 !important; }
.upload-container {
    border: 2px dashed rgba(46, 204, 113, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(46, 204, 113, 0.03) !important;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

auto_yolo = str(_find_yolo() or DEFAULT_PT)
auto_clf  = str(DEFAULT_CLF)

_THEME = gr.themes.Base(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0d1b2a",
    body_background_fill_dark="#0d1b2a",
    block_background_fill="rgba(255,255,255,0.03)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.07)",
    button_primary_background_fill="#2ecc71",
    button_primary_background_fill_hover="#27ae60",
    button_primary_text_color="#0d1b2a",
)

with gr.Blocks(title="Crop Disease Detector", theme=_THEME, css=CUSTOM_CSS) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 28px 0 14px;">
        <div style="font-size:3rem; margin-bottom:8px;">🌿</div>
        <h1 class="title-text">Crop Disease Detector</h1>
        <p class="sub-text">
            Two-Stage Pipeline &nbsp;·&nbsp;
            Stage 1: EfficientNet-B2 crop classifier &nbsp;·&nbsp;
            Stage 2: YOLO26 disease detector &nbsp;·&nbsp;
            23 disease classes &nbsp;·&nbsp;
            🌽 Corn &nbsp;·&nbsp; 🫑 Pepper &nbsp;·&nbsp; 🍅 Tomato
        </p>
    </div>
    """)

    # ── Status bar ────────────────────────────────────────────────────────────
    status_box = gr.Textbox(
        value="⏳ Upload a crop leaf image to begin…",
        interactive=False,
        elem_classes=["status-box"],
        show_label=False,
    )

    # ── Stage 1 result banner ─────────────────────────────────────────────────
    stage1_output = gr.HTML(value="")

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left panel: inputs ────────────────────────────────────────────────
        with gr.Column(scale=4):
            img_input = gr.Image(
                label="Upload crop leaf image",
                type="pil",
                elem_id="upload",
                height=380,
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️  Pipeline settings", open=True):
                clf_slider = gr.Slider(
                    minimum=0.30, maximum=0.90, value=0.55, step=0.01,
                    label="Stage 1 — Classifier confidence threshold",
                    info="Images below this score are rejected as unknown/non-crop",
                )
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=0.50, step=0.01,
                    label="Stage 2 — Detection confidence threshold",
                    info="Minimum YOLO box confidence to show a detection",
                )
                iou_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.45, step=0.01,
                    label="Stage 2 — IoU / NMS threshold",
                    info="Controls bounding box overlap deduplication",
                )

            with gr.Accordion("🔧  Advanced — Model paths", open=False):
                clf_path_box = gr.Textbox(
                    label="Classifier model path (best.pth)",
                    value=auto_clf,
                    info="Produced by train_classifier.py",
                )
                model_path_box = gr.Textbox(
                    label="YOLO detector model path (best.pt)",
                    value=auto_yolo,
                    info="Produced by train.py",
                )

            run_btn = gr.Button("🔍  Analyse Image", variant="primary", size="lg")

        # ── Right panel: outputs ──────────────────────────────────────────────
        with gr.Column(scale=6):
            with gr.Tabs():

                with gr.Tab("🔍 Annotated Image"):
                    img_output = gr.Image(
                        label="Detection result",
                        type="pil",
                        height=440,
                        show_label=False,
                    )

                with gr.Tab("📋 Detections Table"):
                    table_output = gr.Dataframe(
                        headers=["Disease", "Crop", "Confidence", "Bounding Box"],
                        datatype=["str", "str", "str", "str"],
                        label="Detected conditions",
                        wrap=True,
                        interactive=False,
                        row_count=(1, "dynamic"),
                    )

                with gr.Tab("📊 Confidence Chart"):
                    chart_output = gr.Plot(label="Detection confidence breakdown")

    # ── Example images ────────────────────────────────────────────────────────
    sample_dir    = PROJECT_ROOT / "data" / "main" / "test" / "images"
    sample_images = sorted(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []

    if sample_images:
        gr.HTML('<p style="color:#8fa3b1; font-size:0.85rem; margin:16px 0 4px;">📁 Sample images from test set:</p>')
        gr.Examples(
            examples=[[str(p)] for p in sample_images],
            inputs=img_input,
            label="",
            examples_per_page=6,
        )

    # ── About accordion ───────────────────────────────────────────────────────
    with gr.Accordion("ℹ️  About the two-stage pipeline", open=False):
        gr.Markdown("""
        ## How it works

        This demo runs a **two-stage pipeline** on every image:

        ### Stage 1 — Crop-Type Classifier (EfficientNet-B2)
        The classifier was trained on 2,861 crop images to identify whether a leaf
        belongs to **Corn**, **Pepper**, or **Tomato**. If the maximum class probability
        falls below the *classifier threshold* (default **0.55**), the image is
        **rejected** before the detector ever runs. This prevents the disease detector
        from hallucinating disease labels on mango leaves, banana leaves, or unrelated images.

        | Input | Stage 1 output | Action |
        |-------|---------------|--------|
        | Corn leaf | Corn — 96% | Pass to YOLO, Corn classes only |
        | Tomato leaf | Tomato — 88% | Pass to YOLO, Tomato classes only |
        | Mango leaf | unknown — 23% | **Rejected** — no detection |
        | Blurry photo | unknown — 41% | **Rejected** — below threshold |

        ### Stage 2 — Disease Detector (YOLO26)
        YOLO26 runs only on images that pass Stage 1, and its output is filtered to
        the class IDs that belong to the identified crop. This eliminates cross-crop
        false positives (e.g., the detector predicting a Tomato disease on a Corn leaf).

        ## Supported disease classes (23)
        | 🌽 Corn (5) | 🫑 Pepper (10) | 🍅 Tomato (8) |
        |-------------|----------------|---------------|
        | Cercospora Leaf Spot | Bacterial Spot | Bacterial Spot |
        | Common Rust | Cercospora | Early Blight |
        | Healthy | Early Blight | Fusarium |
        | Northern Leaf Blight | Fusarium | Healthy |
        | Streak | Healthy | Late Blight |
        | | Late Blight | Leaf Curl |
        | | Leaf Blight | Mosaic |
        | | Leaf Curl | Septoria |
        | | Leaf Mosaic | |
        | | Septoria | |

        ## Dataset
        Trained on the [Ghana Crop Disease Challenge v2](https://universe.roboflow.com/ghanacropdiseasechallenge/ghana-crop-disease-challenge)
        (CC BY 4.0) — ~5,239 images, ~35,775 annotations.
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; color:#4a5568; font-size:0.78rem;
                padding:24px 0 8px; border-top:1px solid rgba(255,255,255,0.05);
                margin-top:24px;">
        🌿 Crop Disease Detector &nbsp;·&nbsp; Two-Stage Pipeline &nbsp;·&nbsp;
        EfficientNet-B2 + YOLO26 &nbsp;·&nbsp; Ghana Crop Disease Challenge Dataset
    </div>
    """)

    # ── Wire up events ────────────────────────────────────────────────────────
    inputs  = [img_input, clf_slider, conf_slider, iou_slider,
               model_path_box, clf_path_box]
    outputs = [img_output, stage1_output, table_output, chart_output, status_box]

    run_btn.click(fn=predict, inputs=inputs, outputs=outputs)
    img_input.change(fn=predict, inputs=inputs, outputs=outputs)
    clf_slider.release(fn=predict, inputs=inputs, outputs=outputs)
    conf_slider.release(fn=predict, inputs=inputs, outputs=outputs)
    iou_slider.release(fn=predict, inputs=inputs, outputs=outputs)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
