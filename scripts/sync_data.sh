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
    --exclude 'data' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    ./ "${GPU_HOST}:${REMOTE_DIR}/" || die "code sync failed"
  echo "  ✓ code synced ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git') on this end)"
}

# All datasets now live under data/ :
#   data/detector/  ≈ 3.8 GB — detector images, grouped by crop (RCNN / ViT / Swin)
#   data/yolo/      ≈ 0.3 GB — YOLO / RT-DETR format, also the classifier's source
#   data/negatives/ ≈ 0.5 GB — hard negatives + non-target foliage
#
# data/ measures ~7.7 GB on disk, but 3.6 GB of that is *.npy — Ultralytics'
# `cache="disk"` sidecars (see src/yolo/train.py). They are regenerated on the
# first epoch, and *.cache label indexes embed absolute paths, so shipping either
# wastes hours of upload for files the server would rewrite anyway.
sync_data() {
  step "Datasets → ${GPU_HOST} (~4.1 GB first time, resumable)"
  rsync -az --exclude '.DS_Store' --exclude '*.npy' --exclude '*.cache' \
    data "${GPU_HOST}:${REMOTE_DIR}/" || die "data sync failed"
  echo "  ✓ datasets synced"
}

# A stratified slice of data/detector/ — enough to measure real throughput
# without waiting out the full upload on a slow link. Epoch time scales linearly
# with image count, so full-run estimates extrapolate from this exactly.
sync_subset() {
  local pct="${SUBSET_PCT:-10}"
  step "Dataset subset (${pct}%) → ${GPU_HOST}"
  python3 - "$pct" <<'PY' > /tmp/_subset_files.txt
import csv, sys
pct = int(sys.argv[1])
for split, csvf in [("train",    "data/detector/final_train_labels.csv"),
                    ("validate", "data/detector/final_validate_labels.csv"),
                    ("test",     "data/detector/final_test_labels.csv")]:
    rows = list(csv.DictReader(open(csvf)))
    # Round-robin by class so every label stays represented in the slice.
    by = {}
    for r in rows:
        by.setdefault(r.get("integer_label", "0"), []).append((r["crop"], r["fname"]))
    for cls, items in by.items():
        keep = max(1, len(items) * pct // 100)
        for crop, n in sorted(set(items))[:keep]:
            print(f"{split}/{crop}/{n}")
PY
  echo "  $(wc -l < /tmp/_subset_files.txt | tr -d ' ') images selected"
  # NOTE: macOS openrsync accepts --files-from but silently transfers nothing, so
  # this streams a tar instead. No -z: JPEGs are already compressed.
  ssh "$GPU_HOST" "mkdir -p ~/${REMOTE_DIR}/data/detector"
  # --no-xattrs + COPYFILE_DISABLE stop macOS from shipping quarantine attributes
  # and ._AppleDouble sidecars, which GNU tar on the server only warns about.
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C data/detector -T /tmp/_subset_files.txt \
    | ssh "$GPU_HOST" "tar -xf - -C ~/${REMOTE_DIR}/data/detector" \
    || die "subset sync failed"
  # The CSVs are small and the trainers need all of them present.
  rsync -az data/detector/*.csv data/detector/label_map.json \
    "${GPU_HOST}:${REMOTE_DIR}/data/detector/" || die "label sync failed"
  rm -f /tmp/_subset_files.txt

  # The CSVs still reference every image, but only the slice is on the server —
  # the loaders would fail on the missing files. Rewrite each CSV to the rows
  # whose image actually exists, keeping the original as *.full.csv. A later
  # full `sync_data.sh data` re-copies the unfiltered CSVs and undoes this.
  step "Filtering label CSVs to the images present on the server"
  ssh "$GPU_HOST" "cd ~/${REMOTE_DIR} && python3 - <<'PY'
import csv, os, shutil
for split, name in [('train','final_train_labels.csv'),
                    ('validate','final_validate_labels.csv'),
                    ('test','final_test_labels.csv')]:
    path = os.path.join('data/detector', name)
    if not os.path.exists(path):
        continue
    full = path.replace('.csv', '.full.csv')
    if not os.path.exists(full):
        shutil.copy2(path, full)
    with open(full, newline='') as fh:
        rdr = csv.DictReader(fh)
        fields, rows = rdr.fieldnames, list(rdr)
    keep = [r for r in rows
            if os.path.exists(os.path.join('data/detector', split, r['crop'], r['fname']))]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    print(f'  {split:9} {len(keep):6,} / {len(rows):6,} rows kept')
PY" || die "CSV filtering failed"

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
  local_train=$(find data/detector/train -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  local_val=$(find data/detector/validate -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  local_test=$(find data/detector/test -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  echo "  laptop : train=${local_train} validate=${local_val} test=${local_test}"
  ssh "$GPU_HOST" "cd ~/${REMOTE_DIR} 2>/dev/null && printf '  server : train=%s validate=%s test=%s\n' \
    \$(find data/detector/train -name '*.jpg' 2>/dev/null | wc -l) \
    \$(find data/detector/validate -name '*.jpg' 2>/dev/null | wc -l) \
    \$(find data/detector/test -name '*.jpg' 2>/dev/null | wc -l)"
  ssh "$GPU_HOST" "cd ~/${REMOTE_DIR} 2>/dev/null && echo '  server disk:' && du -sh data/* 2>/dev/null"
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
