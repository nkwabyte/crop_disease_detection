#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# yolo_capacity_sweep.sh — train YOLO26 s and m to answer the capacity question.
#
# RUNS ON: the GPU server, from the repo root. Start it inside tmux.
#
# The question (docs/08_next_steps.md § PENDING): three yolo26n runs have plateaued
# at mAP50 ~0.28 on a split holding only ~124 training images per disease class.
# Is the ceiling model capacity, or data? A sharp gain from a larger model says
# capacity; no gain says the ceiling is the dataset — and that is the more useful
# answer, because it redirects effort to collection.
#
# Ordered s BEFORE m deliberately: s is both the faster run and the realistic
# mobile candidate, so if the box dies partway the more useful result is the one
# already finished.
#
# Each variant gets its own runs/ directory via YOLO_EXP, so no run overwrites
# another's weights. Weights are archived after each variant, so a later failure
# cannot cost you an earlier success.
#
# Usage:
#   tmux new -s sweep 'bash scripts/yolo_capacity_sweep.sh'
#   bash scripts/yolo_capacity_sweep.sh yolo26s        # just one variant
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# variant : batch. Conservative on purpose — this runs unattended, and an OOM
# costs the whole run. yolo26n used 8.5 GB at batch 32 on a 31.4 GB card; these
# scale that with parameter count and leave headroom.
VARIANTS="yolo26s:48
yolo26m:32"

WANT="$*"
want() { [ -z "$WANT" ] && return 0; for a in $WANT; do [ "$a" = "$1" ] && return 0; done; return 1; }

echo "════════════════════════════════════════════════════════════════"
echo "  YOLO26 capacity sweep   ($(date '+%F %H:%M:%S'))"
"$PY" -c "import torch;print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null
echo "════════════════════════════════════════════════════════════════"

while IFS=: read -r model batch; do
  [ -n "$model" ] || continue
  want "$model" || continue

  weights="models/${model}.pt"
  if [ ! -f "$weights" ]; then
    echo "  ✗ ${model}: ${weights} missing — skipping"
    continue
  fi

  exp="crop_disease_${model}"
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "  ▶ ${model}   batch ${batch}   ->  runs/${exp}   ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────────────────────────"

  start=$(date +%s)
  YOLO_MODEL="$weights" YOLO_EXP="$exp" \
    "$PY" -m src.yolo.train --batch-size "$batch" 2>&1 | tee "logs/${model}.log"
  rc=${PIPESTATUS[0]}

  # An OOM should not end the sweep: retry once at half batch, then move on.
  if [ "$rc" -ne 0 ]; then
    half=$(( batch / 2 ))
    echo "  ! ${model} failed (rc=$rc) — retrying once at batch ${half}"
    YOLO_MODEL="$weights" YOLO_EXP="$exp" \
      "$PY" -m src.yolo.train --batch-size "$half" 2>&1 | tee -a "logs/${model}.log"
    rc=${PIPESTATUS[0]}
  fi

  mins=$(( ($(date +%s) - start) / 60 ))
  if [ "$rc" -eq 0 ]; then
    echo "  ✓ ${model} done in ${mins}m"
    # Archive immediately — a later variant crashing must not cost this result.
    "$PY" scripts/archive_weights.py --model "$model" \
      --run "runs/${exp}" --pte "models/nonexistent.pte" 2>&1 | sed 's/^/    /'
  else
    echo "  ✗ ${model} FAILED after ${mins}m — see logs/${model}.log"
  fi
done <<EOF
$VARIANTS
EOF

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Sweep finished ($(date '+%F %H:%M:%S'))"
"$PY" scripts/archive_weights.py --list 2>/dev/null
echo
echo "  Compare against yolo26n (mAP50 0.2904) — and weigh mAP against .pte size:"
echo "    yolo26n .pt 5.3 MB   yolo26s .pt 20.4 MB   yolo26m .pt 44.3 MB"
echo "════════════════════════════════════════════════════════════════"
