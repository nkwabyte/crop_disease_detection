#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# preflight.sh — prove the pipelines run BEFORE paying for a GPU.
#
# RUNS ON: your laptop (CPU/MPS). No CUDA required.
#
# Every failure this project hit on a rented box was a bug that a tiny local run
# would have caught, and each cost real money and hours:
#
#   models/yolo26n.pt.pt        wrong path, only on the FRESH-START branch — every
#                               local run had resumed, so it was never exercised
#   boxplot(labels=)            matplotlib 3.11 removed the kwarg; only reached in
#                               figure generation AFTER a full 179-epoch run
#   NameError: main             a truncated file that still compiled and was synced
#   data/detector_yolo/data/…   a relative YOLO_DATA joined onto itself
#
# So this does not just import modules — it runs each pipeline end to end on a
# tiny subset: fresh start (no checkpoint), training, figures, export. That is the
# combination that exercises the code paths the failures lived in.
#
# Usage:
#   bash scripts/preflight.sh              # everything
#   bash scripts/preflight.sh yolo vit     # named stages only
#   PCT=1 EPOCHS=1 bash scripts/preflight.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
PCT="${PCT:-1}"           # percent of the detector set to use
EPOCHS="${EPOCHS:-1}"
WORK="outputs/_preflight"
mkdir -p logs "$WORK"

WANT="$*"
want() { [ -z "$WANT" ] && return 0; for a in $WANT; do [ "$a" = "$1" ] && return 0; done; return 1; }

pass=0; fail=0; skipped=""
check() {                       # check <name> <command...>
  local name="$1"; shift
  want "$name" || { skipped="$skipped $name"; return 0; }
  printf "  %-34s " "$name"
  local out; out=$("$@" 2>&1)
  if [ $? -eq 0 ]; then
    echo "ok"; pass=$((pass+1))
  else
    echo "FAIL"
    echo "$out" | tail -6 | sed 's/^/      /'
    fail=$((fail+1))
  fi
}

echo "════════════════════════════════════════════════════════════════"
echo "  Preflight — ${PCT}% of data, ${EPOCHS} epoch(s), no GPU needed"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Static: everything compiles and imports ───────────────────────────────
echo
echo "  Static checks"
check "all modules compile" bash -c \
  'for f in $(git ls-files "*.py"); do python3 -m py_compile "$f" || exit 1; done'
check "all scripts parse" bash -c \
  'for f in $(git ls-files "*.sh"); do bash -n "$f" || exit 1; done'
check "no file shorter than HEAD" bash -c \
  'for f in $(git ls-files "src/**/*.py"); do
     a=$(wc -l < "$f"); b=$(git show "HEAD:$f" 2>/dev/null | wc -l)
     [ "$a" -ge $((b - 5)) ] || { echo "$f shrank: $a vs $b"; exit 1; }
   done'
for m in src.classifier.train_classifier src.yolo.train src.rtdetr.train_rtdetr \
         src.vit.train_vit src.swin.train_swin src.fasterrcnn.faster_rcnn_final; do
  check "import ${m##*.}" "$PY" -c "import importlib;importlib.import_module('$m')"
done
check "import train_alt_faster_rcnn" "$PY" -c \
  "import importlib;m=importlib.import_module('src.fasterrcnn.train_alt_faster_rcnn');
assert hasattr(m,'main_baseline') and hasattr(m,'main_ablation'), 'missing entry point'"

# ── 2. Data resolves ─────────────────────────────────────────────────────────
echo
echo "  Data"
check "detector CSV rows resolve" "$PY" - <<'PY'
import csv, os, sys
bad = 0
for split, name in [("train","final_train_labels.csv"),("validate","final_validate_labels.csv"),
                    ("test","final_test_labels.csv")]:
    p = os.path.join("data/detector", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    for r in list(csv.DictReader(open(p)))[:500]:
        if not os.path.exists(os.path.join("data/detector", split, r["crop"], r["fname"])):
            bad += 1
sys.exit(f"{bad} unresolved paths" if bad else 0)
PY
check "yolo dataset present" bash -c \
  '[ -d data/yolo/train/images ] && [ -d data/yolo/train/labels ]'

# ── 3. Fresh-start training — the branch the .pt.pt bug lived on ─────────────
echo
echo "  Fresh-start training (${EPOCHS} epoch, ${PCT}% data)"
check "fasterrcnn fresh start" env FRCNN_SUBSET_PCT="$PCT" FRCNN_OUT="$WORK/frcnn" \
  "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline \
        --epochs "$EPOCHS" --skip-negatives --no-figures
check "vit fresh start" env FRCNN_SUBSET_PCT="$PCT" \
  "$PY" -m src.vit.train_vit --epochs "$EPOCHS" --skip-negatives --no-figures --batch-size 1
check "yolo fresh start" bash -c \
  "rm -rf runs/_preflight_yolo && YOLO_EXP=_preflight_yolo $PY -m src.yolo.train \
   --epochs $EPOCHS --skip-negatives --no-figures --batch-size 4"

# ── 4. The post-training steps that only run at the very end ────────────────
echo
echo "  Figures and export"
check "yolo figures" bash -c \
  "YOLO_EXP=_preflight_yolo $PY -m src.yolo.train --figures-only"
check "classifier export" bash -c \
  "command -v flatc >/dev/null || exit 0; $PY -m src.classifier.export_classifier"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  passed: $pass   failed: $fail${skipped:+   skipped:$skipped}"
[ "$fail" -eq 0 ] && echo "  Safe to spend money on a GPU." \
                  || echo "  Fix these before renting — each one would cost GPU hours."
echo "════════════════════════════════════════════════════════════════"
[ "$fail" -eq 0 ]
