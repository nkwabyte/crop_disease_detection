#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rtdetr.sh — export RT-DETR-L to ONNX + ExecuTorch (.pte).
#
# RUNS ON: your Mac (or any machine with `flatc`) — NOT the GPU server.
# ExecuTorch shells out to `flatc` to serialize the XNNPACK payload, and the
# rented box has no package for it.
#
# Like YOLO, RT-DETR goes through Ultralytics' own ExecuTorch exporter.
#
# Usage:
#   bash scripts/export/rtdetr.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}

echo "  ▶ exporting RT-DETR-L"
"$PY" -m src.rtdetr.export_rtdetr "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh models/*rtdetr* 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
