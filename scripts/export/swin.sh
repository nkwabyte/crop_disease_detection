#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# swin.sh — export the Swin detector (Swin-V2-T) to ExecuTorch (.pte).
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
#   bash scripts/export/swin.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

CKPT="outputs/swin_output/checkpoints/best.pth"

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}
[ -f "$CKPT" ] || {
  echo "  ✗ no Swin checkpoint at $CKPT"
  echo "    Train it first (bash scripts/train_swin.sh), or pull it:"
  echo "      bash scripts/sync_data.sh pull"
  exit 1
}

echo "  ▶ exporting Swin   (checkpoint: $CKPT)"
"$PY" -m src.swin.train_swin --export-only "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh outputs/swin_output/models/*.pte models/*swin* 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
