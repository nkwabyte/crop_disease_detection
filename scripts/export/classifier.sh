#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# classifier.sh — export the crop classifier to ExecuTorch (.pte).
#
# RUNS ON: your Mac (or any machine with `flatc`) — NOT the GPU server.
# ExecuTorch shells out to `flatc` to serialize the XNNPACK payload, and the
# rented box has no package for it. Training happens on the server; conversion
# happens here.
#
# Usage:
#   bash scripts/export/classifier.sh                 # 3-class — the model the app loads
#   bash scripts/export/classifier.sh --variant ood   # 4-class with a learned "Other"
#
# Produces  models/crop_classifier.pte  +  models/classifier_metadata.yaml
#      (or  models/crop_classifier_ood.pte  for the ood variant)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

# Pick the checkpoint the requested variant writes to, so the precondition
# check fails with a useful message instead of a Python traceback.
CKPT="outputs/classifier_output/best.pth"
case " $* " in *" ood "*) CKPT="outputs/classifier_ood_output/best.pth" ;; esac

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}
[ -f "$CKPT" ] || {
  echo "  ✗ no checkpoint at $CKPT"
  echo "    Train it first, or pull it from the server:"
  echo "      bash scripts/sync_data.sh pull"
  exit 1
}

echo "  ▶ exporting classifier   (checkpoint: $CKPT)"
"$PY" -m src.classifier.export_classifier "$@"
rc=$?
[ "$rc" -eq 0 ] && ls -lh models/crop_classifier*.pte 2>/dev/null | awk '{print "    "$9"  "$5}'
exit "$rc"
