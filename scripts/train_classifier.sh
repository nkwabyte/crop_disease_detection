#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_classifier.sh — Stage-1 crop classifier (EfficientNet-B2 @ 260px).
#
# RUNS ON: the GPU server, from the repo root.
#
# Usage:
#   bash scripts/train_classifier.sh                 # full run (40 epochs)
#   bash scripts/train_classifier.sh --dry-run       # 2-epoch timing estimate
#   bash scripts/train_classifier.sh --epochs 10
#   BATCH=256 bash scripts/train_classifier.sh       # override the auto-picked batch
#   SKIP_EXPORT=1 bash scripts/train_classifier.sh   # train only, no .pte export
#
# Extra args pass through to `python -m src.classifier.train_classifier`.
# Logs to logs/classifier.log. Resume-aware — re-run to continue from the last checkpoint.
#
# Unattended:  tmux new -s clf 'bash scripts/train_classifier.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="classifier"
MODULE="src.classifier.train_classifier"

# Unbuffered so `tee` streams progress live instead of block-buffering it.
export PYTHONUNBUFFERED=1

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Batch size ────────────────────────────────────────────────────────────────
# src/classifier/config.py sets CUDA_BATCH=128 assuming a 96 GB RTX PRO 6000.
# EfficientNet-B2 at 260px is light, so 128 still fits a 32 GB card comfortably.
pick_batch() {
  local mib gb
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${mib:-}" ] || { echo ""; return; }        # no GPU — let config.py decide
  gb=$(( mib / 1024 ))
  if   [ "$gb" -ge 80 ]; then echo 128    # RTX PRO 6000 96 GB (config default)
  elif [ "$gb" -ge 30 ]; then echo 128    # RTX 5090 / 4090 32 GB
  elif [ "$gb" -ge 20 ]; then echo 64
  else                        echo 32
  fi
}

BATCH_ARG=()
case " $* " in
  *" --batch-size "*) ;;                            # explicit flag always wins
  *) B="${BATCH:-$(pick_batch)}"; [ -n "$B" ] && BATCH_ARG=(--batch-size "$B") ;;
esac

# ── Prerequisite: the per-crop CSVs the classifier trains from ────────────────
if [ ! -f dataset/classifier_train.csv ]; then
  echo "  classifier CSVs missing — generating them first ..."
  "$PY" -m src.classifier.generate_classifier_csv 2>&1 | tee "logs/${MODEL}_csv.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "  ✗ CSV generation failed — aborting."; exit 1; }
fi

# ── Train ─────────────────────────────────────────────────────────────────────
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
  echo "    OOM? Retry with a smaller batch:  BATCH=64 bash scripts/train_classifier.sh"
  exit "$rc"
fi
echo "  ✓ ${MODEL} finished in ${mins}m — log: logs/${MODEL}.log"

# ── Export (skipped for dry-runs and figure-only passes) ──────────────────────
case " $* " in *" --dry-run "*|*" --figures-only "*) exit 0 ;; esac
[ "${SKIP_EXPORT:-0}" = "1" ] && exit 0

echo "  Exporting classifier ..."
"$PY" -m src.classifier.export_classifier 2>&1 | tee "logs/${MODEL}_export.log"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo "  ✓ export done" || echo "  ! export failed — training results are still valid"
exit 0
