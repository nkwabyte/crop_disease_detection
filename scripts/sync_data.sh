#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_data.sh — push code + datasets to the GPU server, and pull results back.
#
# RUNS ON: your laptop.
#
# Usage:
#   bash scripts/sync_data.sh              # code + datasets (default)
#   bash scripts/sync_data.sh code         # source only — fast, after every edit
#   bash scripts/sync_data.sh data         # datasets only (~4.1 GB the first time)
#   bash scripts/sync_data.sh subset       # code + a 10% slice, for a quick throughput test
#   bash scripts/sync_data.sh pull         # bring outputs/ + logs/ back to the laptop
#   bash scripts/sync_data.sh check        # compare file counts on both ends
#
# SUBSET_PCT=5 bash scripts/sync_data.sh subset   # smaller slice
#
# Transfers are resumable at file level: re-run after an interruption and rsync
# skips everything already copied. Safe to run repeatedly.
#
# macOS ships openrsync (rsync 2.6.9-compatible), so this sticks to flags that
# both it and the server's rsync 3.x understand — no --info/--partial.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

GPU_HOST="${GPU_HOST:-gpumart}"
REMOTE_DIR="${REMOTE_DIR:-crop_disease_detection}"
MODE="${1:-all}"

die() { echo "  ✗ $*" >&2; exit 1; }
step() { echo; echo "════ $* ════"; }

ssh -o BatchMode=yes -o ConnectTimeout=10 "$GPU_HOST" true 2>/dev/null \
  || die "Cannot reach ${GPU_HOST}. Run: bash scripts/connect.sh --setup"

ssh "$GPU_HOST" "mkdir -p ~/${REMOTE_DIR}"

# Source tree: everything except the venv, the datasets, and generated artifacts.
sync_code() {
  step "Code → ${GPU_HOST}:~/${REMOTE_DIR}"
  rsync -az \
    --exclude '.venv' --exclude 'outputs' --exclude 'runs' \
    --exclude 'dataset' --exclude 'data' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    ./ "${GPU_HOST}:${REMOTE_DIR}/" || die "code sync failed"
  echo "  ✓ code synced ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git') on this end)"
}

# Datasets. dataset/ ≈ 3.8 GB of JPEGs (classifier / RCNN / ViT / Swin),
#           data/    ≈ 0.3 GB once caches are excluded (YOLO + RT-DETR).
#
# data/ measures 3.9 GB on disk, but 3.6 GB of that is *.npy — Ultralytics'
# `cache="disk"` sidecars (see src/yolo/train.py). They are regenerated on the
# first epoch, and *.cache label indexes embed absolute paths, so shipping either
# wastes hours of upload for files the server would rewrite anyway.
sync_data() {
  step "Datasets → ${GPU_HOST} (~4.1 GB first time, resumable)"
  echo "  dataset/ (~3.8 GB) ..."
  rsync -az --exclude '.DS_Store' \
    dataset "${GPU_HOST}:${REMOTE_DIR}/" || die "dataset sync failed"
  echo "  data/ (~0.3 GB — skipping regenerable .npy/.cache) ..."
  rsync -az --exclude '.DS_Store' --exclude '*.npy' --exclude '*.cache' \
    data "${GPU_HOST}:${REMOTE_DIR}/" || die "data sync failed"
  echo "  ✓ datasets synced"
}

# A stratified slice of dataset/ — enough to measure real throughput (sec/iter)
# without waiting out the full upload on a slow link. Epoch time scales linearly
# with image count, so full-run estimates extrapolate from this exactly.
sync_subset() {
  local pct="${SUBSET_PCT:-10}"
  step "Dataset subset (${pct}%) → ${GPU_HOST}"
  python3 - "$pct" <<'PY' > /tmp/_subset_files.txt
import csv, sys, os
pct = int(sys.argv[1])
for split, csvf in [("train","dataset/final_train_labels.csv"),
                    ("validate","dataset/final_validate_labels.csv")]:
    rows = list(csv.DictReader(open(csvf)))
    # Round-robin by class so every label stays represented in the slice.
    by = {}
    for r in rows:
        by.setdefault(r.get("integer_label", "0"), []).append(r["fname"])
    for cls, names in by.items():
        keep = max(1, len(names) * pct // 100)
        for n in sorted(set(names))[:keep]:
            print(f"{split}/{n}")
PY
  echo "  $(wc -l < /tmp/_subset_files.txt | tr -d ' ') images selected"
  # NOTE: macOS openrsync accepts --files-from but silently transfers nothing, so
  # this streams a tar instead. No -z: JPEGs are already compressed.
  ssh "$GPU_HOST" "mkdir -p ~/${REMOTE_DIR}/dataset"
  # --no-xattrs + COPYFILE_DISABLE stop macOS from shipping quarantine attributes
  # and ._AppleDouble sidecars, which GNU tar on the server only warns about.
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C dataset -T /tmp/_subset_files.txt \
    | ssh "$GPU_HOST" "tar -xf - -C ~/${REMOTE_DIR}/dataset" \
    || die "subset sync failed"
  # The CSVs are small and the trainers need all of them present.
  rsync -az dataset/*.csv dataset/label_map.json "${GPU_HOST}:${REMOTE_DIR}/dataset/" \
    || die "label sync failed"
  rm -f /tmp/_subset_files.txt
  echo "  ✓ subset synced — later a full 'sync_data.sh data' run tops up the rest"
}

# Trained artifacts + logs back to the laptop for inspection.
pull_results() {
  step "Results ← ${GPU_HOST}"
  mkdir -p outputs logs
  rsync -az "${GPU_HOST}:${REMOTE_DIR}/outputs/" outputs/ || echo "  ! outputs/ pull failed (nothing there yet?)"
  rsync -az "${GPU_HOST}:${REMOTE_DIR}/logs/"    logs/    || echo "  ! logs/ pull failed (no runs yet?)"
  echo "  ✓ pulled into ./outputs and ./logs"
}

check_counts() {
  step "File counts"
  local_train=$(ls dataset/train 2>/dev/null | wc -l | tr -d ' ')
  local_val=$(ls dataset/validate 2>/dev/null | wc -l | tr -d ' ')
  local_test=$(ls dataset/test 2>/dev/null | wc -l | tr -d ' ')
  echo "  laptop : train=${local_train} validate=${local_val} test=${local_test}"
  ssh "$GPU_HOST" "cd ~/${REMOTE_DIR} 2>/dev/null && printf '  server : train=%s validate=%s test=%s\n' \
    \$(ls dataset/train 2>/dev/null | wc -l) \
    \$(ls dataset/validate 2>/dev/null | wc -l) \
    \$(ls dataset/test 2>/dev/null | wc -l)"
  ssh "$GPU_HOST" "cd ~/${REMOTE_DIR} 2>/dev/null && echo '  server disk:' && du -sh dataset data 2>/dev/null"
}

case "$MODE" in
  all)    sync_code; sync_data; check_counts ;;
  code)   sync_code ;;
  data)   sync_data; check_counts ;;
  subset) sync_code; sync_subset ;;
  pull)   pull_results ;;
  check)  check_counts ;;
  *)      die "unknown mode '$MODE' — use: all | code | data | subset | pull | check" ;;
esac

echo
echo "Done."
