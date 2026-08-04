#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# vit.sh — export the ViTDet detector (ViT-B/16) to ExecuTorch (.pte).
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
#   bash scripts/export/vit.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

CKPT="outputs/vit_output/checkpoints/best.pth"

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}
[ -f "$CKPT" ] || {
  echo "  ✗ no ViT checkpoint at $CKPT"
  echo "    Train it first (bash scripts/train_vit.sh), or pull it:"
  echo "      bash scripts/sync_data.sh pull"
  exit 1
}

echo "  ▶ exporting ViTDet   (checkpoint: $CKPT)"
"$PY" -m src.vit.train_vit --export-only "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh outputs/vit_output/models/*.pte models/*vit* 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
