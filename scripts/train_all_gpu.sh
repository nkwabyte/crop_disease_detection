#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_all_gpu.sh — full training + benchmark + export sweep for the GPU server.
#
# Runs every model sequentially (Stage-1 classifier, then all Stage-2 detectors),
# then the cross-model benchmark, latency, quantization and export steps. Each step
# logs to logs/<step>.log and continues on failure so one bad run doesn't abort the
# sweep. Batch sizes auto-scale to CUDA (see each src/*/config.py CUDA_BATCH*).
#
# Usage:
#   bash scripts/train_all_gpu.sh                 # everything
#   bash scripts/train_all_gpu.sh classifier vit  # only these steps
#   EPOCHS_VIT=60 bash scripts/train_all_gpu.sh vit   # override via env (see below)
#
# Run it inside tmux/screen so it survives disconnects:
#   tmux new -s train 'bash scripts/train_all_gpu.sh 2>&1 | tee logs/sweep.log'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python"
mkdir -p logs

# Optional per-model batch overrides (empty = use the CUDA auto-default from config).
BATCH_CLF="${BATCH_CLF:-}"        # e.g. 256
BATCH_FRCNN="${BATCH_FRCNN:-}"    # e.g. 24
BATCH_FINAL="${BATCH_FINAL:-}"    # e.g. 24
BATCH_VIT="${BATCH_VIT:-}"        # e.g. 12
BATCH_SWIN="${BATCH_SWIN:-}"      # e.g. 24
BATCH_RTDETR="${BATCH_RTDETR:-}"  # e.g. 24

_bs() { [ -n "$1" ] && echo "--batch-size $1" || echo ""; }

run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "════════════════════════════════════════════════════════════════"
  echo "  ▶  ${label}   ($(date '+%H:%M:%S'))"
  echo "     $*"
  echo "════════════════════════════════════════════════════════════════"
  "$@" 2>&1 | tee "logs/${label}.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && echo "  ✓ ${label} done" || echo "  ✗ ${label} FAILED (rc=$rc) — continuing"
}

# has <name> : true if no step args were given, or <name> is among them
ARGS="$*"
has() { [ -z "$ARGS" ] && return 0; for a in $ARGS; do [ "$a" = "$1" ] && return 0; done; return 1; }

echo "Python: $($PY --version 2>&1)"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')" 2>/dev/null || true

# ── Stage 1: crop classifier ────────────────────────────────────────────────────
if has classifier; then
  run classifier_csv "$PY" -m src.classifier.generate_classifier_csv
  run classifier     "$PY" -m src.classifier.train_classifier $(_bs "$BATCH_CLF")
  run classifier_export "$PY" -m src.classifier.export_classifier
fi

# ── Stage 2: detectors ──────────────────────────────────────────────────────────
if has yolo; then
  run yolo        "$PY" -m src.yolo.train
  run yolo_export "$PY" -m src.yolo.export_yolo
fi
if has fasterrcnn;  then run fasterrcnn "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline $(_bs "$BATCH_FRCNN"); fi
if has final;       then run final      "$PY" -m src.fasterrcnn.faster_rcnn_final       $(_bs "$BATCH_FINAL"); fi
if has vit;         then run vit        "$PY" -m src.vit.train_vit                 $(_bs "$BATCH_VIT"); fi
if has swin;        then run swin       "$PY" -m src.swin.train_swin               $(_bs "$BATCH_SWIN"); fi
if has rtdetr;      then
  run rtdetr        "$PY" -m src.rtdetr.train_rtdetr $(_bs "$BATCH_RTDETR")
  run rtdetr_export "$PY" -m src.rtdetr.export_rtdetr
fi

# ── Aggregate + deployment artifacts ────────────────────────────────────────────
if has benchmark || [ -z "$ARGS" ]; then
  run benchmark "$PY" -m src.benchmark.compare_models
  run quantize  "$PY" -m src.benchmark.quantize
  run latency   "$PY" -m src.benchmark.latency
fi

echo "All done. Per-step logs in ./logs/ ; benchmark outputs in outputs/benchmarks/."
