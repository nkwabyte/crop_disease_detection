#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# all.sh — export every model that has a checkpoint to ExecuTorch (.pte).
#
# RUNS ON: your Mac (or any machine with `flatc`) — NOT the GPU server.
#
# Models without a checkpoint are SKIPPED rather than treated as failures, so
# this is safe to run when only some models have been trained. A model whose
# export genuinely fails is reported and the run continues.
#
# Usage:
#   bash scripts/export/all.sh                    # everything available
#   bash scripts/export/all.sh classifier yolo    # only these
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

command -v flatc >/dev/null 2>&1 || {
  echo "  ✗ flatc not found — ExecuTorch needs it to write the .pte."
  echo "    macOS : brew install flatbuffers"
  echo "    Debian: sudo apt-get install -y flatbuffers-compiler"
  exit 1
}

# model : checkpoint that must exist for the export to be attempted
CKPTS="classifier:outputs/classifier_output/best.pth
yolo:runs/crop_disease_yolo26/weights/best.pt
fasterrcnn:outputs/fasterrcnn_output/checkpoints/best.pth
final:outputs/final_output/checkpoints/best.pth
vit:outputs/vit_output/checkpoints/best.pth
swin:outputs/swin_output/checkpoints/best.pth
rtdetr:runs/rtdetr/weights/best.pt"

ARGS="$*"
want() { [ -z "$ARGS" ] && return 0; for a in $ARGS; do [ "$a" = "$1" ] && return 0; done; return 1; }

ok=""; failed=""; skipped=""
while IFS=: read -r model ckpt; do
  [ -n "$model" ] || continue
  want "$model" || continue
  if [ ! -f "$ckpt" ]; then
    skipped="$skipped $model"
    continue
  fi
  echo
  echo "════════════════════════════════════════════════════════════════"
  bash "scripts/export/${model}.sh"
  if [ $? -eq 0 ]; then ok="$ok $model"; else failed="$failed $model"; fi
done <<EOF
$CKPTS
EOF

echo
echo "════════════════════════════════════════════════════════════════"
[ -n "$ok" ]      && echo "  ✓ exported :$ok"
[ -n "$skipped" ] && echo "  – skipped  :$skipped   (no checkpoint yet)"
[ -n "$failed" ]  && echo "  ✗ failed   :$failed"
echo
echo "  Artifacts in models/ :"
ls -lh models/*.pte 2>/dev/null | awk '{print "    "$9"  "$5}'
[ -z "$failed" ]
