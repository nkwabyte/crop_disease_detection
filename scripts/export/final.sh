#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# final.sh — export the SE-FPN final detector to ExecuTorch (.pte).
#
# RUNS ON: your Mac (or any machine with `flatc`) — NOT the GPU server.
# ExecuTorch shells out to `flatc` to serialize the XNNPACK payload, and the
# rented box has no package for it.
#
# The torchvision detectors have no standalone export module: the exporter lives
# inside their trainer, reached with --export-only, which loads the best
# checkpoint and skips training entirely.
#
# Usage:
#   bash scripts/export/final.sh
#   bash scripts/export/final.sh --no-ema    # export raw weights, not the EMA copy
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

CKPT="outputs/final_output/checkpoints/best.pth"

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}
[ -f "$CKPT" ] || {
  echo "  ✗ no SE-FPN checkpoint at $CKPT"
  echo "    Train it first (bash scripts/train_final.sh), or pull it:"
  echo "      bash scripts/sync_data.sh pull"
  exit 1
}

echo "  ▶ exporting SE-FPN final   (checkpoint: $CKPT)"
"$PY" -m src.fasterrcnn.faster_rcnn_final --export-only "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh outputs/final_output/models/*.pte models/*final* 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
