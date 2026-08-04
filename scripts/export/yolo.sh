#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# yolo.sh — export YOLO26n to ONNX + ExecuTorch (.pte).
#
# RUNS ON: your Mac (or any machine with `flatc`) — NOT the GPU server.
# ExecuTorch shells out to `flatc` to serialize the XNNPACK payload, and the
# rented box has no package for it.
#
# Ultralytics exports to ExecuTorch directly (`model.export(format="executorch")`),
# so this wrapper just checks the preconditions and calls the project's script.
#
# Usage:
#   bash scripts/export/yolo.sh
#
# Produces  models/crop_disease_yolo26.pte  +  .onnx  +  yolo_metadata.yaml
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

CKPT="runs/crop_disease_yolo26/weights/best.pt"

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}
[ -f "$CKPT" ] || {
  echo "  ✗ no YOLO checkpoint at $CKPT"
  echo "    Train it first (bash scripts/train_yolo.sh), or pull runs/ from the server."
  exit 1
}

echo "  ▶ exporting YOLO26n   (checkpoint: $CKPT)"
"$PY" -m src.yolo.export_yolo "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh models/crop_disease_yolo26.* 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
