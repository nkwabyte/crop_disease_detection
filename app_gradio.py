"""
Crop Disease Detection — Gradio Demo
──────────────────────────────────────
Run with:
    python app_gradio.py
  or
    gradio app_gradio.py

Requires:  pip install -r requirements.txt
Model:     runs/crop_disease_yolo26/weights/best.pt  (produced by train.py)
"""

from __future__ import annotations

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

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
RUNS_DIR     = PROJECT_ROOT / "runs"
EXP_NAME     = "crop_disease_yolo26"
DEFAULT_PT   = RUNS_DIR / EXP_NAME / "weights" / "best.pt"

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

# Visual palette — one colour per class (R, G, B) 0-255
PALETTE = [
    (231, 76,  60),  (192, 57,  43),  (46,  204, 113), (39,  174, 96),
    (52,  152, 219), (41,  128, 185), (155, 89,  182), (142, 68,  173),
    (243, 156, 18),  (230, 126, 34),  (26,  188, 156), (22,  160, 133),
    (52,  73,  94),  (44,  62,  80),  (127, 140, 141), (149, 165, 166),
    (211, 84,  0),   (192, 57,  43),  (39,  174, 96),  (41,  128, 185),
    (142, 68,  173), (243, 156, 18),  (22,  160, 133),
]

CROP_ICONS = {"Corn": "🌽", "Pepper": "🫑", "Tomato": "🍅"}


# ──────────────────────────────────────────────────────────────────────────────
# Device — prefer MPS (Apple Silicon) → CUDA → CPU
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

INFER_DEVICE = _resolve_device()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_model() -> Path | None:
    if DEFAULT_PT.exists():
        return DEFAULT_PT
    candidates = sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


_model_cache: dict[str, YOLO] = {}

def get_model(model_path: str) -> YOLO:
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    pil_img: Image.Image,
    conf: float,
    iou: float,
    model_path: str,
) -> tuple[list[dict], float]:
    model = get_model(model_path)
    arr   = np.array(pil_img.convert("RGB"))
    t0    = time.perf_counter()
    results = model.predict(source=arr, conf=conf, iou=iou, verbose=False, device=INFER_DEVICE)
    elapsed = time.perf_counter() - t0
    r = results[0]
    detections = []
    if r.boxes and len(r.boxes):
        for box in r.boxes:
            cls_id = int(box.cls[0])
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


def draw_boxes(
    pil_img: Image.Image,
    detections: list[dict],
    thickness: int = 3,
) -> Image.Image:
    out  = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        r, g, b = det["color"]
        fill    = (r, g, b, 255)

        # Filled semi-transparent rectangle
        draw.rectangle([x1, y1, x2, y2], outline=fill, width=thickness)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 22))

        # Label
        label    = f"  {det['icon']} {det['class_name'].replace('_',' ')}  {det['confidence']:.2f}  "
        text_bb  = draw.textbbox((0, 0), label, font=font)
        th       = text_bb[3] - text_bb[1]
        ty1      = max(0, int(y1) - th - 6)
        ty2      = int(y1)
        tx2      = int(x1) + (text_bb[2] - text_bb[0]) + 6
        draw.rectangle([int(x1), ty1, tx2, ty2], fill=(r, g, b, 220))
        draw.text((int(x1) + 3, ty1 + 1), label, fill=(255, 255, 255, 255), font=font)

    return out


def make_summary_table(detections: list[dict]) -> list[list]:
    rows = []
    for det in detections:
        rows.append([
            f"{det['icon']} {det['class_name'].replace('_', ' ')}",
            det["crop"],
            f"{det['confidence']:.1%}",
            f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}) → ({det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})",
        ])
    return rows


def make_conf_chart(detections: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, max(2, len(detections) * 0.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if detections:
        names  = [d["class_name"].replace("_", " ") for d in detections]
        confs  = [d["confidence"] for d in detections]
        colors = [tuple(c/255 for c in d["color"]) for d in detections]
        bars   = ax.barh(names, confs, color=colors, height=0.6, edgecolor="none")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.8, label="OOD threshold (0.50)")
        for bar, conf in zip(bars, confs):
            ax.text(
                min(conf + 0.02, 0.93),
                bar.get_y() + bar.get_height() / 2,
                f"{conf:.0%}", va="center", ha="left",
                color="#eaeaea", fontsize=10, fontweight="bold",
            )
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="#aaaaaa", framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                color="#7f8c8d", fontsize=14, transform=ax.transAxes)

    ax.set_xlabel("Confidence Score", color="#8fa3b1", fontsize=10)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.spines[:].set_color("#2c2c4e")
    ax.xaxis.grid(True, color="#2c2c4e", linewidth=0.8)
    plt.tight_layout(pad=1.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main prediction function (called by Gradio)
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    image: Image.Image | None,
    conf_threshold: float,
    iou_threshold: float,
    model_path: str,
) -> tuple:
    """
    Returns: annotated_image, summary_table, conf_chart, status_text
    """
    if image is None:
        placeholder = Image.new("RGB", (640, 400), (20, 30, 48))
        draw = ImageDraw.Draw(placeholder)
        draw.text((200, 185), "Upload an image to begin ☝️", fill=(100, 130, 160))
        return placeholder, [], make_conf_chart([]), "⏳ Awaiting image…"

    model_file = Path(model_path)
    if not model_file.exists():
        err_img = Image.new("RGB", (640, 200), (40, 10, 10))
        draw    = ImageDraw.Draw(err_img)
        draw.text((20, 85), f"Model not found: {model_path}", fill=(231, 76, 60))
        return err_img, [], make_conf_chart([]), f"❌ Model not found: {model_path}"

    detections, elapsed = run_inference(image, conf_threshold, iou_threshold, model_path)
    annotated = draw_boxes(image, detections)
    table     = make_summary_table(detections)
    chart     = make_conf_chart(detections)

    if detections:
        top        = detections[0]
        icon       = top["icon"]
        disease    = top["class_name"].replace("_", " ")
        confidence = top["confidence"]
        n          = len(detections)
        status = (
            f"✅  {n} detection{'s' if n > 1 else ''} found  ·  "
            f"Top: {icon} {disease} ({confidence:.1%})  ·  "
            f"Inference: {elapsed*1000:.0f} ms  [{INFER_DEVICE}]"
        )
    else:
        status = (
            f"🚫  No crop disease detected above {conf_threshold:.0%} confidence.  "
            f"Inference: {elapsed*1000:.0f} ms  [{INFER_DEVICE}]  |  "
            "This image may not contain a recognised crop leaf."
        )

    return annotated, table, chart, status


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #0d1b2a, #1a2840) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto; }

/* Header */
#component-0 { text-align: center; }
.title-text {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #2ecc71, #3498db, #9b59b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-text { color: #8fa3b1; font-size: 1rem; }

/* Status bar */
.status-box textarea, .status-box input {
    background: rgba(46, 204, 113, 0.08) !important;
    border: 1px solid rgba(46, 204, 113, 0.2) !important;
    color: #2ecc71 !important;
    font-size: 0.9rem !important;
    font-weight: 500;
}

/* Tabs */
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

/* Panel / block */
.block { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }

/* Sliders */
input[type=range]::-webkit-slider-thumb { background: #2ecc71 !important; }

/* Upload area */
.upload-container {
    border: 2px dashed rgba(46, 204, 113, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(46, 204, 113, 0.03) !important;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

auto_model = str(_find_model() or DEFAULT_PT)

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

with gr.Blocks(title="Crop Disease Detector") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 28px 0 14px;">
        <div style="font-size:3rem; margin-bottom:8px;">🌿</div>
        <h1 class="title-text">Crop Disease Detector</h1>
        <p class="sub-text">
            Powered by YOLO26 &nbsp;·&nbsp; 23 disease classes &nbsp;·&nbsp;
            🌽 Corn &nbsp;·&nbsp; 🫑 Pepper &nbsp;·&nbsp; 🍅 Tomato
        </p>
    </div>
    """)

    # ── Status bar ────────────────────────────────────────────────────────────
    status_box = gr.Textbox(
        label="Status",
        value="⏳ Upload an image to run detection…",
        interactive=False,
        elem_classes=["status-box"],
        show_label=False,
    )

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left panel: inputs ────────────────────────────────────────────────
        with gr.Column(scale=4):
            img_input = gr.Image(
                label="Upload crop image",
                type="pil",
                elem_id="upload",
                height=380,
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Detection settings", open=True):
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=0.50, step=0.01,
                    label="Confidence threshold",
                    info="Raise to reduce OOD false positives on non-crop images",
                )
                iou_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.45, step=0.01,
                    label="IoU (NMS) threshold",
                    info="Controls bounding box overlap deduplication",
                )

            with gr.Accordion("🔧 Advanced — Model path", open=False):
                model_path_box = gr.Textbox(
                    label="Path to best.pt",
                    value=auto_model,
                    info="Produced by train.py  (runs/crop_disease_yolo26/weights/best.pt)",
                )

            run_btn = gr.Button(
                "🔍  Detect Diseases",
                variant="primary",
                size="lg",
            )

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
                    chart_output = gr.Plot(label="Confidence breakdown")

    # ── Examples ──────────────────────────────────────────────────────────────
    sample_dir = PROJECT_ROOT / "data" / "main" / "test" / "images"
    sample_images = sorted(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []

    if sample_images:
        gr.HTML('<p style="color:#8fa3b1; font-size:0.85rem; margin:16px 0 4px;">📁 Sample images from test set:</p>')
        gr.Examples(
            examples=[[str(p)] for p in sample_images],
            inputs=img_input,
            label="",
            examples_per_page=6,
        )

    # ── Info accordion ────────────────────────────────────────────────────────
    with gr.Accordion("ℹ️  About this model & OOD rejection", open=False):
        gr.Markdown("""
        ## About the Model
        This demo uses **YOLO26** — Ultralytics' January 2026 architecture — fine-tuned on the
        [Ghana Crop Disease Challenge v2](https://universe.roboflow.com/ghanacropdiseasechallenge/ghana-crop-disease-challenge)
        dataset (~40,850 training images, 23 disease classes across Corn, Pepper and Tomato).
        The model is loaded from `runs/crop_disease_yolo26/weights/best.pt` produced by `train.py`.

        ## Why does it sometimes detect crops in non-crop images?
        YOLO is a **closed-set classifier** — it always tries to map any input to one of its
        known classes. To mitigate this:

        | Strategy | Effect |
        |----------|--------|
        | Hard negative mining (300 extra background images) | Model learns to output nothing on non-crop scenes |
        | Label smoothing = 0.1 during training | Model becomes less overconfident |
        | Confidence threshold ≥ 0.50 (default) | Rejects low-confidence spurious detections |

        **If you still see false positives** on human faces or unrelated objects:
        - Raise the confidence threshold to **0.65 – 0.75**
        - The model will then only report detections it is very sure about.

        ## Supported Classes (23)
        | 🌽 Corn | 🫑 Pepper | 🍅 Tomato |
        |---------|----------|----------|
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
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; color:#4a5568; font-size:0.78rem; padding:24px 0 8px; border-top:1px solid rgba(255,255,255,0.05); margin-top:24px;">
        🌿 Crop Disease Detector · YOLO26 · Ghana Crop Disease Challenge Dataset
    </div>
    """)

    # ── Wire up events ────────────────────────────────────────────────────────
    inputs  = [img_input, conf_slider, iou_slider, model_path_box]
    outputs = [img_output, table_output, chart_output, status_box]

    run_btn.click(fn=predict, inputs=inputs, outputs=outputs)

    # Auto-run when image is uploaded
    img_input.change(fn=predict, inputs=inputs, outputs=outputs)

    # Re-run when sliders change (debounced by Gradio)
    conf_slider.release(fn=predict, inputs=inputs, outputs=outputs)
    iou_slider.release(fn=predict, inputs=inputs, outputs=outputs)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,         # set True to get a public Gradio link
        show_error=True,
        theme=_THEME,
        css=CUSTOM_CSS,
    )
