#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_final.sh — SE-FPN final detector (the tuned Faster R-CNN variant:
#                  squeeze-excite FPN, EMA weights, TTA).
#
# RUNS ON: the GPU server, from the repo root.
#
# Usage:
#   bash scripts/train_final.sh                 # full run
#   bash scripts/train_final.sh --dry-run       # 2-epoch timing estimate
#   bash scripts/train_final.sh --epochs 10
#   BATCH=12 bash scripts/train_final.sh        # override the auto-picked batch
#   bash scripts/train_final.sh --no-ema --no-tta
#   bash scripts/train_final.sh --export-only   # re-export the best checkpoint
#
# Extra args pass through to `python -m src.fasterrcnn.train_final`.
# Logs to logs/final.log. Resume-aware — re-run to continue from the last checkpoint.
# Exports at the end of training on its own; no separate export step.
#
# Unattended:  tmux new -s final 'bash scripts/train_final.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="final"
MODULE="src.fasterrcnn.train_final"

# Unbuffered so `tee` streams progress live instead of block-buffering it.
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Batch size ────────────────────────────────────────────────────────────────
# src/fasterrcnn/config.py sets CUDA_BATCH_SIZE=16 assuming a 96 GB RTX PRO 6000.
# Same ResNet-50 FPN cost profile as the baseline, plus an EMA copy of the
# weights held in memory — so keep the 32 GB value conservative.
pick_batch() {
  local mib gb
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${mib:-}" ] || { echo ""; return; }        # no GPU — let config.py decide
  gb=$(( mib / 1024 ))
  if   [ "$gb" -ge 80 ]; then echo 16     # RTX PRO 6000 96 GB (config default)
  elif [ "$gb" -ge 30 ]; then echo 8      # RTX 5090 / 4090 32 GB
  elif [ "$gb" -ge 20 ]; then echo 4
  else                        echo 2
  fi
}

BATCH_ARG=()
case " $* " in
  *" --batch-size "*) ;;                            # explicit flag always wins
  *) B="${BATCH:-$(pick_batch)}"; [ -n "$B" ] && BATCH_ARG=(--batch-size "$B") ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo "  ▶  ${MODEL}   ($(date '+%F %H:%M:%S'))"
"$PY" -c "import torch; print('     GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>/dev/null
echo "     $PY -m $MODULE ${BATCH_ARG[*]} $*"
echo "════════════════════════════════════════════════════════════════"

start=$(date +%s)
"$PY" -m "$MODULE" "${BATCH_ARG[@]}" "$@" 2>&1 | tee "logs/${MODEL}.log"
rc=${PIPESTATUS[0]}
mins=$(( ($(date +%s) - start) / 60 ))

if [ "$rc" -eq 0 ]; then
  echo "  ✓ ${MODEL} finished in ${mins}m — log: logs/${MODEL}.log"
else
  echo "  ✗ ${MODEL} FAILED (rc=$rc) after ${mins}m — see logs/${MODEL}.log"
  echo "    OOM? Retry with a smaller batch:  BATCH=4 bash scripts/train_final.sh"
fi
exit "$rc"
