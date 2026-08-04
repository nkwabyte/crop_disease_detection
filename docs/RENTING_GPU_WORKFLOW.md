# Renting a Monthly GPU for Competition Training

A practical workflow for training the crop-disease and biohub (ZebraTrack) repos on a
rented monthly GPU server (GPU Mart), plus the checkpoint push-and-pull loop so nothing
is lost when you are done. Written for a Kaggle and Zindi competition cadence.

Why monthly (not hourly): a flat monthly box stays provisioned between sessions, so your
data and environment persist and you never re-stream 80 GB or reinstall between runs. That
convenience is worth it when you are training most days across several competitions.

---

## 1. Which plan to rent

The binding constraint is VRAM. The crop project's ViTDet wants roughly 30 to 40 GB at its
default batch, so a 48 GB card runs everything with no tuning. Biohub only needs about
16.5 GB, so it fits anything from 24 GB up.

Recommended primary: the 48 GB tier.

- RTX A6000 dedicated, 48 GB, about 329.40/mo. Bare metal, runs both projects at
  documented defaults. Best value with no compromises.
- RTX Pro 5000 VPS, 48 GB Blackwell, about 349/mo. Newer generation, a bit faster, same
  48 GB. Either is a fine primary.

Step up only if you want speed for a tight deadline:

- RTX Pro 6000, 96 GB, about 599/mo, roughly 1.7x faster.
- A100 40 GB, about 399 to 639/mo depending on the listing, HBM memory.

Step down only to save money if you accept batch tuning:

- RTX A5000 or RTX Pro 4000, 24 GB, about 175 to 199/mo. Both projects still run, but you
  lower the crop ViT and Faster R-CNN batch sizes (see section 5).

Skip 16 GB cards (A4000, V100) for the heavy crop detectors, and skip multi-GPU servers:
most training scripts here are single-GPU, so extra cards sit idle.

---

## 2. One-time box setup

Provision the server (Ubuntu), then connect and start a persistent session so long runs
survive a disconnect.

```bash
ssh user@<server-ip>
sudo apt-get update && sudo apt-get install -y git tmux rsync
tmux new -s work            # detach with Ctrl-b then d; reattach with: tmux attach -t work

# confirm the GPU is visible
nvidia-smi
```

### Crop-disease repo

```bash
git clone <crop-repo-url> crop_disease_detection && cd crop_disease_detection
git checkout dev
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-server.txt     # torch pulls the CUDA build on Linux
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If executorch fails to install, ignore it: it is only needed for the export and quantize
steps, not training.

### Biohub (ZebraTrack) repo

```bash
git clone <biohub-repo-url> biohub && cd biohub
python3.12 -m venv .venv && source .venv/bin/activate     # biohub is pinned to Python 3.12
pip install -e .
# the host scorer (dev/eval only) has a git-URL dep that needs --no-deps:
pip install --no-deps "tracking-cellmot @ git+https://github.com/royerlab/kaggle-cell-tracking-competition@main"
pip install -e ".[dev]" ".[tooling]"
```

Biohub also ships a one-command bring-up that installs CUDA torch, deps, the scorer, and
streams the data:

```bash
scripts/bring_up_cloud.sh --with-zebrahub     # bare box to training-ready, incl. external data
```

---

## 3. Get the data onto the box

The datasets are git-ignored (large). On a monthly box you do this once and it persists.

Crop-disease (about 7.4 GB total). Either rsync from your Mac, or use the Roboflow
download documented in docs/07_dataset.md.

```bash
# from your Mac, into the server:
rsync -avP dataset/train dataset/validate dataset/test  user@<server-ip>:~/crop_disease_detection/dataset/
rsync -avP data/main                                    user@<server-ip>:~/crop_disease_detection/data/
```

Biohub (about 82 GB). The bring-up script streams it; otherwise fetch the external crops:

```bash
python scripts/fetch_zebrahub.py --output-dir data/zebrahub
```

---

## 4. Run training

### Crop-disease: the whole sweep, unattended

```bash
tmux new -s train
bash scripts/train_all_gpu.sh 2>&1 | tee logs/sweep.log
# Ctrl-b d to detach; tmux attach -t train to return
```

Run a subset by naming steps, and override batch sizes per model via env vars:

```bash
bash scripts/train_all_gpu.sh classifier yolo          # only these
BATCH_VIT=2 BATCH_FRCNN=8 bash scripts/train_all_gpu.sh vit fasterrcnn
```

Every script is resume-aware: re-running the same command continues from the last
checkpoint. Each also takes --dry-run for a quick timing estimate before committing.

### Biohub: per fold (train both folds for a real LOEO comparison)

```bash
# self-supervised pretraining, then the fine-tuned detector, fold A
python scripts/train.py --config configs/pretrain_ext_a.yaml --run-name pretrain_ext_a --pretrain
python scripts/train.py --config configs/detector_ext_long_a.yaml --run-name detector_ext_long_a
# repeat with the _b configs for fold B
```

Score a checkpoint on its held-out fold through the real harness:

```bash
python scripts/run_local_eval.py --checkpoint checkpoints/detector_ext_long_a/ckpt_XXXX \
    --data-root data --device cuda --tracker lap --limit 3
```

---

## 5. Batch-size cheatsheet by card VRAM

Crop defaults are tuned for a large card. On smaller VRAM, lower these two memory-heavy
runs; everything else fits. Biohub fits at defaults on any 24 GB+ card.

| Card VRAM | Crop ViT (BATCH_VIT) | Crop Faster R-CNN / final / Swin | Notes |
| --------- | -------------------- | -------------------------------- | ----- |
| 48 GB (A6000, Pro 5000, A40) | 8 (default) | 16 (default) | No tuning needed |
| 40 GB (A100) | 8 | 16 | ViT tight but fits |
| 32 GB (RTX 5090) | 6 to 8 | 16 | Near-default |
| 24 GB (A5000, Pro 4000, 4090) | 2 to 4 | 8 | Set BATCH_VIT / BATCH_FRCNN / BATCH_FINAL / BATCH_SWIN |
| 96 GB (Pro 6000) | 12 to 16 | 24 to 32 | Push higher for speed |

Rule of thumb: watch `nvidia-smi` during the first epoch. If it OOMs, halve the batch; if
the GPU sits below 60 percent utilised, raise it.

---

## 6. Monitor

```bash
watch -n2 nvidia-smi        # GPU utilisation and memory
tail -f logs/vit.log        # live crop training log (per step in scripts/train_all_gpu.sh)
tail -f checkpoints/detector_ext_long_a/train.log   # biohub
```

---

## 7. Checkpoints in and out (do not lose work)

Outputs and checkpoints are git-ignored in both repos. On a monthly box they persist
between sessions, but still copy anything you care about off the server.

Crop-disease: rsync the results back to your Mac.

```bash
# from your Mac:
rsync -avP user@<server-ip>:~/crop_disease_detection/outputs ./outputs
rsync -avP user@<server-ip>:~/crop_disease_detection/runs    ./runs
```

Biohub: use the built-in Kaggle dataset push and pull, which also stages weights for the
offline submission notebook.

```bash
python scripts/push_checkpoint_to_kaggle.py --checkpoint checkpoints/detector_ext_long_a/ckpt_XXXX \
    --kaggle-dataset-slug <user>/zebratrack-detectors-v2
# on another machine (or Kaggle):
python scripts/pull_checkpoint.py --kaggle-dataset-slug <user>/zebratrack-detectors-v2 \
    --output-dir checkpoints/from_kaggle
```

---

## 8. Submit (biohub Kaggle code competition)

Biohub is a code competition: the submission is an offline, internet-off notebook driven
by the package.

```bash
scripts/submit_kaggle.sh "message"        # push notebook, run on Kaggle, submit
SUBMIT=0 scripts/submit_kaggle.sh "msg"   # push and run only, to validate first
```

Force a T4 accelerator on Kaggle (the notebook's default P100 is unsupported by its torch
build). Budget is about 161 s per video, roughly 9 hours for 199 videos, inside the
12-hour limit.

For Zindi, there is no hosted compute: train on the rented box, then submit the prediction
file the competition asks for.

---

## 9. Cost discipline on a monthly box

- A monthly plan is flat, so utilisation is free once paid. Keep the box busy: queue the
  next competition's training rather than letting it idle.
- Snapshot or rsync your data and checkpoints before the billing cycle ends, so you can
  cancel and re-provision later without re-streaming 82 GB.
- If a competition ends and the next is weeks away, cancel the monthly plan and switch to
  hourly rental for the gap, then come back to monthly when training ramps up again.
- One 48 GB box serves both repos and any typical Zindi vision task, so you do not need a
  second server.

---

## Quick reference

| Task | Command |
| ---- | ------- |
| Start persistent session | `tmux new -s train` |
| Crop full sweep | `bash scripts/train_all_gpu.sh 2>&1 \| tee logs/sweep.log` |
| Crop subset + batch override | `BATCH_VIT=4 bash scripts/train_all_gpu.sh vit` |
| Biohub pretrain then detector | `python scripts/train.py --config configs/pretrain_ext_a.yaml --pretrain` |
| Biohub bring-up + data | `scripts/bring_up_cloud.sh --with-zebrahub` |
| Watch GPU | `watch -n2 nvidia-smi` |
| Pull crop results to Mac | `rsync -avP user@ip:~/crop_disease_detection/outputs ./outputs` |
| Push biohub checkpoint | `python scripts/push_checkpoint_to_kaggle.py --checkpoint <dir> --kaggle-dataset-slug <slug>` |
| Biohub submit | `scripts/submit_kaggle.sh "message"` |
