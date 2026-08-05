#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_detectors_sweep.sh — train the four torchvision detectors back to back.
#
# RUNS ON: the GPU server, from the repo root. Start it inside tmux.
#
# These four all read data/detector/, so none can start until that upload is
# complete — the script refuses to run on a partial dataset rather than training
# on whatever happens to be present and producing numbers that mean nothing.
#
# Ordered lightest first. On an unattended overnight run the box may expire or a
# later model may fail, so the cheapest results should be banked earliest:
#
#   fasterrcnn  ResNet-50 FPN v2      lightest
#   final       SE-FPN + EMA + TTA    baseline plus attention and EMA overhead
#   swin        Swin-V2-T             windowed attention
#   vit         ViT-B/16              heaviest — 1600 tokens/image at 640px
#
# Each step continues past a failure so one bad model cannot end the sweep, and
# every model checkpoints per epoch, so a box that dies mid-run leaves resumable
# state rather than nothing.
#
# Usage:
#   tmux new -s detectors 'bash scripts/train_detectors_sweep.sh'
#   bash scripts/train_detectors_sweep.sh swin vit      # only these
#   SKIP_DATA_CHECK=1 bash scripts/train_detectors_sweep.sh   # train on partial data
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# ── Refuse to train on a partial dataset ─────────────────────────────────────
# The CSVs reference every image; a missing file is a crash mid-epoch at best and
# a silently truncated training set at worst.
if [ "${SKIP_DATA_CHECK:-0}" != "1" ]; then
  echo "  Checking data/detector is complete ..."
  "$PY" - <<'PY'
import csv, os, sys
missing = total = 0
for split, name in [("train", "final_train_labels.csv"),
                    ("validate", "final_validate_labels.csv"),
                    ("test", "final_test_labels.csv")]:
    path = os.path.join("data/detector", name)
    if not os.path.exists(path):
        sys.exit(f"  x {path} missing — upload has not finished")
    rows = list(csv.DictReader(open(path)))
    absent = [r for r in rows if not os.path.exists(
        os.path.join("data/detector", split, r["crop"], r["fname"]))]
    total += len(rows)
    missing += len(absent)
    print(f"    {split:9} {len(rows) - len(absent):6,} / {len(rows):6,} images present")
if missing:
    sys.exit(f"  x {missing:,} of {total:,} images still missing — wait for the upload, "
             f"or set SKIP_DATA_CHECK=1 to train on a partial set anyway")
print(f"  OK - all {total:,} images present")
PY
  [ $? -eq 0 ] || exit 1
fi

# model : module : batch   (batch sized for a ~32 GB card; see scripts/README.md)
MODELS="fasterrcnn:src.fasterrcnn.train_alt_faster_rcnn --mode baseline:8
final:src.fasterrcnn.faster_rcnn_final:8
swin:src.swin.train_swin:16
vit:src.vit.train_vit:16"

WANT="$*"
want() { [ -z "$WANT" ] && return 0; for a in $WANT; do [ "$a" = "$1" ] && return 0; done; return 1; }

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Detector sweep   ($(date '+%F %H:%M:%S'))"
"$PY" -c "import torch;print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null
echo "════════════════════════════════════════════════════════════════"

done_ok=""; failed=""
while IFS=: read -r model module batch; do
  [ -n "$model" ] || continue
  want "$model" || continue

  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "  ▶ ${model}   batch ${batch}   ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────────────────────────"
  start=$(date +%s)
  # shellcheck disable=SC2086  # $module may carry a --mode flag
  "$PY" -m $module --batch-size "$batch" 2>&1 | tee "logs/${model}_full.log"
  rc=${PIPESTATUS[0]}
  mins=$(( ($(date +%s) - start) / 60 ))

  if [ "$rc" -eq 0 ]; then
    echo "  ✓ ${model} done in ${mins}m"
    done_ok="$done_ok $model"
  else
    echo "  ✗ ${model} FAILED (rc=$rc) after ${mins}m — continuing"
    echo "    OOM? re-run just this one with a smaller batch:"
    echo "      bash scripts/train_detectors_sweep.sh ${model}   # after lowering it above"
    failed="$failed $model"
  fi
done <<EOF
$MODELS
EOF

echo
echo "════════════════════════════════════════════════════════════════"
[ -n "$done_ok" ] && echo "  ✓ completed:$done_ok"
[ -n "$failed" ]  && echo "  ✗ failed   :$failed"
echo "  Pull results:  bash scripts/sync_data.sh pull   (from the laptop)"
echo "════════════════════════════════════════════════════════════════"
