#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_vit.sh — ViTDet detector (ViT-B/16 backbone + Faster R-CNN head @ 640px).
#
# RUNS ON: the GPU server, from the repo root.
# This is the heaviest model in the project — its epoch time drives the total
# cost of a full sweep, so it is the one worth timing before committing.
#
# Usage:
#   bash scripts/train_vit.sh                 # full run (40 epochs)
#   bash scripts/train_vit.sh --dry-run       # 2-epoch timing estimate
#   bash scripts/train_vit.sh --epochs 10
#   BATCH=6 bash scripts/train_vit.sh         # override the auto-picked batch
#   bash scripts/train_vit.sh --export-only   # re-export the best checkpoint
#
# Extra args pass through to `python -m src.vit.train_vit`.
# Logs to logs/vit.log. Resume-aware — re-run to continue from the last checkpoint.
# Exports at the end of training on its own; no separate export step.
#
# Unattended:  tmux new -s vit 'bash scripts/train_vit.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="vit"
MODULE="src.vit.train_vit"

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Batch size ────────────────────────────────────────────────────────────────
# src/vit/config.py sets CUDA_BATCH_SIZE=8 assuming a 96 GB RTX PRO 6000.
# ViT-B/16 at 640px is 1600 tokens per image — the most memory-hungry model here,
# so a 32 GB card needs this roughly halved. ACCUM_STEPS=4 keeps the effective
# batch reasonable even when the physical batch is small.
pick_batch() {
  local mib gb
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${mib:-}" ] || { echo ""; return; }        # no GPU — let config.py decide
  gb=$(( mib / 1024 ))
  if   [ "$gb" -ge 80 ]; then echo 8      # RTX PRO 6000 96 GB (config default)
  elif [ "$gb" -ge 30 ]; then echo 4      # RTX 5090 / 4090 32 GB
  elif [ "$gb" -ge 20 ]; then echo 2
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
  echo "    OOM? Retry with a smaller batch:  BATCH=2 bash scripts/train_vit.sh"
fi
exit "$rc"
