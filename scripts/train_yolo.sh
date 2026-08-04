#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_yolo.sh — YOLO26n detector (Ultralytics, 640px).
#
# RUNS ON: the GPU server, from the repo root.
#
# Usage:
#   bash scripts/train_yolo.sh                 # full run (200 epochs)
#   bash scripts/train_yolo.sh --dry-run       # 1-epoch timing estimate
#   bash scripts/train_yolo.sh --epochs 50
#   BATCH=48 bash scripts/train_yolo.sh        # override the auto-picked batch
#   SKIP_EXPORT=1 bash scripts/train_yolo.sh   # train only, no export
#
# Extra args pass through to `python -m src.yolo.train`.
# Logs to logs/yolo.log. Resume-aware — re-run to continue from the last checkpoint.
#
# Note: --batch-size on this trainer is PER-GPU; with >1 GPU Ultralytics runs DDP
# and the effective batch is per-GPU × GPU count.
#
# Unattended:  tmux new -s yolo 'bash scripts/train_yolo.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="yolo"
MODULE="src.yolo.train"

# Unbuffered so `tee` streams progress live instead of block-buffering it.
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Batch size ────────────────────────────────────────────────────────────────
# src/yolo/config.py sets CUDA_BATCH=64 per GPU assuming a 96 GB RTX PRO 6000.
# YOLO26n is a nano model (~2.4M params), so it is the cheapest detector here —
# but 64 × 640px with Ultralytics' mosaic augmentation still crowds a 32 GB card.
pick_batch() {
  local mib gb
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${mib:-}" ] || { echo ""; return; }        # no GPU — let config.py decide
  gb=$(( mib / 1024 ))
  if   [ "$gb" -ge 80 ]; then echo 64     # RTX PRO 6000 96 GB (config default)
  elif [ "$gb" -ge 30 ]; then echo 32     # RTX 5090 / 4090 32 GB
  elif [ "$gb" -ge 20 ]; then echo 24
  else                        echo 16
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
  echo "    OOM? Retry with a smaller batch:  BATCH=16 bash scripts/train_yolo.sh"
  exit "$rc"
fi
echo "  ✓ ${MODEL} finished in ${mins}m — log: logs/${MODEL}.log"

# ── Export (skipped for dry-runs and figure-only passes) ──────────────────────
case " $* " in *" --dry-run "*|*" --figures-only "*) exit 0 ;; esac
[ "${SKIP_EXPORT:-0}" = "1" ] && exit 0

echo "  Exporting YOLO ..."
"$PY" -m src.yolo.export_yolo 2>&1 | tee "logs/${MODEL}_export.log"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo "  ✓ export done" || echo "  ! export failed — training results are still valid"
exit 0
