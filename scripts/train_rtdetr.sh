#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_rtdetr.sh — RT-DETR-L detector (Ultralytics transformer detector, 640px).
#
# RUNS ON: the GPU server, from the repo root.
#
# Usage:
#   bash scripts/train_rtdetr.sh                 # full run
#   bash scripts/train_rtdetr.sh --dry-run       # 1-epoch timing estimate
#   bash scripts/train_rtdetr.sh --epochs 50
#   BATCH=12 bash scripts/train_rtdetr.sh        # override the auto-picked batch
#   SKIP_EXPORT=1 bash scripts/train_rtdetr.sh   # train only, no export
#
# Extra args pass through to `python -m src.rtdetr.train_rtdetr`.
# Logs to logs/rtdetr.log. Resume-aware — re-run to continue from the last checkpoint.
#
# Unattended:  tmux new -s rtdetr 'bash scripts/train_rtdetr.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="rtdetr"
MODULE="src.rtdetr.train_rtdetr"

# Unbuffered so `tee` streams progress live instead of block-buffering it.
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Batch size ────────────────────────────────────────────────────────────────
# src/rtdetr/config.py sets CUDA_BATCH=16 assuming a 96 GB RTX PRO 6000.
# RT-DETR-L is the largest of the Ultralytics models here (~32M params) and its
# decoder attention adds memory on top of the backbone — halve it for 32 GB.
pick_batch() {
  local mib gb
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${mib:-}" ] || { echo ""; return; }        # no GPU — let config.py decide
  gb=$(( mib / 1024 ))
  if   [ "$gb" -ge 80 ]; then echo 16     # RTX PRO 6000 96 GB (config default)
  elif [ "$gb" -ge 30 ]; then echo 8      # RTX 5090 / 4090 32 GB
  elif [ "$gb" -ge 20 ]; then echo 4
  else                        echo 4
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

if [ "$rc" -ne 0 ]; then
  echo "  ✗ ${MODEL} FAILED (rc=$rc) after ${mins}m — see logs/${MODEL}.log"
  echo "    OOM? Retry with a smaller batch:  BATCH=4 bash scripts/train_rtdetr.sh"
  exit "$rc"
fi
echo "  ✓ ${MODEL} finished in ${mins}m — log: logs/${MODEL}.log"

# ── Export (skipped for dry-runs) ─────────────────────────────────────────────
case " $* " in *" --dry-run "*) exit 0 ;; esac
[ "${SKIP_EXPORT:-0}" = "1" ] && exit 0

echo "  Exporting RT-DETR ..."
"$PY" -m src.rtdetr.export_rtdetr 2>&1 | tee "logs/${MODEL}_export.log"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo "  ✓ export done" || echo "  ! export failed — training results are still valid"
exit 0
