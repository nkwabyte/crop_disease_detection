#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# bisect_frcnn_divergence.sh — find why Faster R-CNN goes NaN at epoch 6.
#
# RUNS ON: the GPU server, from the repo root.
#
# Observed: losses fall smoothly for 5 epochs (0.185 -> 0.166) then go nan at
# epoch 6 — exactly FREEZE_BACKBONE_EPOCHS + 1. WARMUP_EPOCHS is 3, so the LR has
# already reached its 5e-3 peak by the time ~23M ResNet-50 parameters start
# receiving gradients, with no re-warm. AMP ordering and grad clipping are both
# correct, so they are not the suspects.
#
# Three hypotheses, one variable each. 7 epochs is enough: the failure is at 6.
#
#   A  freeze=2   unfreeze DURING warmup, while LR is still ramping
#   B  no AMP     rules fp16 overflow in the freshly-unfrozen backbone in or out
#   C  lr=1e-3    tests "peak LR is simply too high once unfrozen"
#
# Whichever arm survives epoch 6 identifies the cause. Arms write to separate
# output directories so none overwrites another, or the real run.
#
# Usage:
#   tmux new -s bisect 'bash scripts/bisect_frcnn_divergence.sh'
#   bash scripts/bisect_frcnn_divergence.sh A C      # only these arms
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs
EPOCHS="${EPOCHS:-7}"

WANT="$*"
want() { [ -z "$WANT" ] && return 0; for a in $WANT; do [ "$a" = "$1" ] && return 0; done; return 1; }

run_arm() {
  local arm="$1" desc="$2"; shift 2
  want "$arm" || return 0
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "  ARM ${arm}: ${desc}   ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────────────────────────"
  local start; start=$(date +%s)
  FRCNN_ARM="$arm" "$@" 2>&1 | tee "logs/bisect_${arm}.log"
  local rc=${PIPESTATUS[0]}
  local mins=$(( ($(date +%s) - start) / 60 ))
  # The guard raises RuntimeError on a non-finite loss, so a non-zero exit with
  # that message means the arm diverged; clean exit means it survived.
  if grep -qa "non-finite loss" "logs/bisect_${arm}.log"; then
    echo "  ✗ ARM ${arm}: DIVERGED after ${mins}m — $(grep -oam1 'non-finite loss at epoch [0-9]*' "logs/bisect_${arm}.log")"
  elif [ "$rc" -eq 0 ]; then
    echo "  ✓ ARM ${arm}: survived ${EPOCHS} epochs in ${mins}m — hypothesis CONFIRMED"
  else
    echo "  ? ARM ${arm}: exited rc=$rc after ${mins}m for another reason — inspect logs/bisect_${arm}.log"
  fi
}

echo "════════════════════════════════════════════════════════════════"
echo "  Faster R-CNN divergence bisect — ${EPOCHS} epochs per arm"
"$PY" -c "import torch;print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null
echo "════════════════════════════════════════════════════════════════"

run_arm A "unfreeze during warmup (FRCNN_FREEZE=2)" \
  env FRCNN_FREEZE=2 FRCNN_OUT=outputs/bisect_a_output \
  "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline \
        --batch-size 8 --epochs "$EPOCHS" --skip-negatives --no-figures

run_arm B "AMP disabled (FRCNN_NO_AMP=1)" \
  env FRCNN_NO_AMP=1 FRCNN_OUT=outputs/bisect_b_output \
  "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline \
        --batch-size 8 --epochs "$EPOCHS" --skip-negatives --no-figures

run_arm C "lower peak LR (FRCNN_LR=1e-3)" \
  env FRCNN_LR=1e-3 FRCNN_OUT=outputs/bisect_c_output \
  "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline \
        --batch-size 8 --epochs "$EPOCHS" --skip-negatives --no-figures

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Summary"
for a in A B C; do
  [ -f "logs/bisect_${a}.log" ] || continue
  if grep -qa "non-finite loss" "logs/bisect_${a}.log"; then
    echo "    ARM ${a}: diverged"
  else
    echo "    ARM ${a}: survived"
  fi
done
echo "  An arm that survives identifies the cause; more than one means they compound."
echo "════════════════════════════════════════════════════════════════"
