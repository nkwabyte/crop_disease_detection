# Renting a Monthly GPU for Competition Training

A practical workflow for training the crop-disease and biohub (ZebraTrack) repos on a
rented monthly GPU server (GPU Mart), plus the checkpoint push-and-pull loop so nothing
is lost when you are done. Written for a Kaggle and Zindi competition cadence.

Why monthly (not hourly): a flat monthly box stays provisioned between sessions, so your
data and environment persist and you never re-stream 80 GB or reinstall between runs. That
convenience is worth it when you are training most days across several competitions.

---

## 1. Which plan to rent

> **Revised 2026-08-04 from measurement.** An earlier version of this section said the
> crop ViTDet "wants roughly 30 to 40 GB" and recommended the 48 GB tier. That estimate
> was never measured. Benchmarked on a rented RTX 5090, **ViTDet peaks at 12.2 GB** at
> batch 16 — a third of the estimate. The recommendation below changes accordingly.
> Evidence is in [section 5a](#5a-measured-performance-rtx-5090) and
> [`scripts/README.md`](../scripts/README.md#measured-performance).

The binding constraint is still VRAM, but it is **biohub (~16.5 GB), not crop (12.2 GB)**.
That flip is what moves the recommendation down a tier.

**Recommended primary: the 24 GB tier** — RTX A5000 or RTX Pro 4000, about 175 to 199/mo.

- Covers biohub's ~16.5 GB with headroom, and crop's 12.2 GB with room to spare.
- Roughly **45 % cheaper** than the 48 GB tier this doc previously recommended, for no
  measured loss on either project.
- Crop needs no batch tuning at all on 24 GB; see the revised cheatsheet in section 5.

Why not step up to 48 GB or 96 GB: crop's throughput **plateaus above batch 8** — 61.7,
63.2 and 64.1 img/s at batches 8, 16 and 24 respectively, while VRAM climbs 5.9 → 14.7 GB.
Tripling the batch bought 4 % throughput. Extra VRAM past ~24 GB therefore buys almost
nothing on these models; you would be paying for memory that sits idle. The 5090 tested
here ran at **61 % of its VRAM unused**.

Step up only if:

- You are adding a workload that genuinely needs the memory — LLM fine-tuning, large
  diffusion training, or video models. None of the current repos do.
- You need ECC. Consumer cards (GeForce 5090/4090) have **no ECC**; a silent bit flip nine
  hours into an eleven-hour run is undetectable and unrecoverable. The A6000 and RTX Pro
  cards have it. This is the strongest argument for the 48 GB tier, and it is a
  reliability argument, not a capacity one.

Skip 16 GB cards: crop would fit, but biohub's 16.5 GB would not. Skip multi-GPU servers
for now — the training scripts are single-GPU. (Note though that because throughput
plateaus, two cheaper GPUs in data-parallel would scale these detectors better than one
faster GPU, if the scripts were ever adapted for it.)

### Check the machine class, not just the card

The listing advertises the GPU; it does not always say what the host is. The 5090 box
tested here turned out to be a **KVM virtual machine**, not bare metal:

| | Found |
| --- | --- |
| Virtualization | KVM, full virtualization |
| CPU | "Intel Core Processor (Broadwell)" — generic QEMU model, 32 vCPU (1 core/socket each) |
| SIMD | AVX2 yes, **AVX-512 no** — CPU image augmentation runs ~2× slower than on modern silicon |
| GPU link | **PCIe Gen 3 x16** (the 5090 supports Gen 5) |
| ECC | Not available (GeForce) |
| Disk | virtio, 475 MB/s write |

None of this blocked the workload — PCIe Gen 3 is using under 2 % of its bandwidth here,
and the slow CPU augmentation is hidden behind dataloader workers. But it is worth asking
the provider directly whether a tier is bare metal or VPS, and whether ECC is available,
before committing monthly.

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
bash scripts/setup_server.sh        # venv + a torch build matching the GPU's architecture
```

Do not just `pip install -r requirements-server.txt` on a recent card. **Blackwell GPUs
(RTX 5090/5080, compute capability sm_120) need CUDA 12.8+ wheels**, and
`torch.cuda.is_available()` returns `True` even when the installed wheel contains no
sm_120 kernels — it only fails at the first real operation, as `CUDA error: no kernel image
is available`. `setup_server.sh` installs from the cu128 index and proves it with an actual
fp16 matmul, then prints VRAM-appropriate batch sizes.

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

Crop-disease: **~4.1 GB needs uploading**, not the 7.7 GB the directories measure. 3.6 GB
of `data/**/*.npy` are Ultralytics `cache="disk"` sidecars that the server regenerates on
its first epoch, and `*.cache` label indexes embed absolute paths. Shipping them nearly
doubles the transfer for files that get overwritten.

```bash
bash scripts/sync_data.sh          # code + data, correct exclusions, resumable
bash scripts/sync_data.sh subset   # 10 % slice (~380 MB) for a quick throughput check
```

Budget for it: on a 1.4 Mbps home uplink, 4.1 GB takes about **6.7 hours**. Measure your
own upload before planning around it — if it is slow, use the Roboflow download in
[`docs/07_dataset.md`](07_dataset.md) from the server's datacenter link instead, or start
with `subset` to get timings the same day and top up the rest overnight (rsync skips what
the subset already placed).

Biohub (about 82 GB). The bring-up script streams it; otherwise fetch the external crops:

```bash
python scripts/fetch_zebrahub.py --output-dir data/zebrahub
```

---

## 4. Run training

### Before the first full run on a new box

1. **Re-run `bash scripts/setup_server.sh`.** It picks a torch build matching that
   GPU's compute capability and prints VRAM-appropriate batch sizes.
2. **Raise the YOLO batch size and re-measure.** On the 5090 these runs sat at
   17–27 % GPU utilisation, so there is free wall-clock in a larger batch. The
   A5000 has 24 GB against that box's 31.4 GB, so measure rather than copy a number.

The YOLO26 capacity question is **settled** — see
[`docs/08_next_steps.md`](08_next_steps.md#ok-done--yolo26-capacity-sweep-rtx-5090-2026-08-04).
8.7x the parameters bought 6.8 % mAP@0.5, so the ceiling is the dataset (~124 training
images per disease class), not the model. yolo26n stays the deployed choice. **The
highest-value work on the detector is now collecting more images, not architecture.**

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

Revised from measurement — the previous table assumed ViT needed far more memory than it
does. Biohub fits at defaults on any 24 GB+ card.

| Card VRAM | Crop ViT (BATCH_VIT) | Crop Faster R-CNN / final / Swin | Notes |
| --------- | -------------------- | -------------------------------- | ----- |
| 24 GB (A5000, Pro 4000, 4090) | 16 | 16 | **Recommended tier.** ViT peaks ~12.2 GB |
| 32 GB (RTX 5090) | 16 | 16 | Measured: 12.2 GB used of 31.4 |
| 40 GB (A100) | 16 | 16 | Headroom unused |
| 48 GB (A6000, Pro 5000, A40) | 16 | 16 | Headroom unused |
| 96 GB (Pro 6000) | 16 | 24 to 32 | Past batch 8 the gain is ~4 % |
| 16 GB (A4000, V100) | 8 | 8 | Crop fits; biohub does not |

Note the column is now nearly constant. That is the point: **throughput plateaus above
batch 8**, so there is no reason to push the batch higher on a bigger card. Batch 8 already
delivers 96 % of peak throughput at roughly half the memory.

The per-model scripts (`scripts/train_<model>.sh`) read the GPU's actual VRAM and apply
these automatically — you only need this table when using `train_all_gpu.sh`, which does
not auto-scale.

Rule of thumb: if it OOMs, halve the batch. **Do not** raise the batch merely because the
GPU sits below 60 % utilised — on these detectors that is normal and is not a sign the
batch is too small; see section 5a.

---

## 5a. Measured performance (RTX 5090)

Benchmarked 2026-08-04 on a rented 5090 (31.4 GB, 32 vCPU, 82 GB RAM), crop repo.
Detector figures use a 10 % subset (4,076 train images); the classifier used its full data.

| Model | Batch | Peak VRAM | Throughput | Full 40-epoch estimate |
| ----- | ----- | --------- | ---------- | ---------------------- |
| ViTDet (ViT-B/16 @640) | 16 | 12.2 GB | ~44 img/s | **~11 h** |
| Swin-V2-T @640 | 16 | 4.5 GB | — | ~10 h |
| Classifier (EfficientNet-B2) | 128 | — | 7 s/epoch | **~7 min** |

Classifier test accuracy after only 2 epochs was 83.76 % (Corn f1 0.98, Pepper 0.85,
**Tomato 0.58 — recall just 0.47**, worth watching in a full run).

**The workload is GPU-bound, not input-bound.** Pure GPU throughput with no dataloader is
63.2 img/s at batch 16; end-to-end training reaches ~44 img/s, about 70 % of that ceiling.
Every dataloader change tested — pre-grouping annotations, dropping the expensive `hue`
augmentation, more workers, thread pinning, deeper prefetch — moved end-to-end throughput
by 1.04× or less, or made it worse. With 16 workers the CPU pipeline already outruns the
GPU.

Two practical consequences:

- Low GPU utilisation (30–60 %) is expected here and is **not** a tuning failure.
- A faster single GPU buys little, because throughput plateaus above batch 8.

Beware when benchmarking this yourself: repeated runs of an *identical* configuration
varied 28–46 img/s on this box. Interleave the variants across several repetitions and
compare medians — a single before/after pair will show speedups that do not exist.

Also note `--dry-run` is 2 epochs while `FREEZE_BACKBONE_EPOCHS = 5`, so a plain dry-run
never unfreezes the ViT/Swin backbone and understates the phase that dominates a real run.

---

## 6. Monitor

```bash
watch -n2 nvidia-smi        # GPU utilisation and memory
tail -f logs/vit.log        # live crop training log (per step in scripts/train_all_gpu.sh)
tail -f checkpoints/detector_ext_long_a/train.log   # biohub
```

From your Mac, without opening a shell on the box:

```bash
bash scripts/connect.sh --gpu          # one-shot nvidia-smi
bash scripts/connect.sh --watch        # live GPU status
bash scripts/connect.sh --logs vit     # tail a training log
```

On a new box, check for **thermal throttling** before trusting a long run — a card that
holds clocks for five minutes may still decay over eleven hours:

```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap \
  --format=csv,noheader -l 15
```

`hw_thermal_slowdown` or `sw_thermal_slowdown` going `Active`, or the SM clock decaying
steadily, means sustained throughput will be below what a short benchmark suggests.

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
- One 24 GB box serves both repos and any typical Zindi vision task, so you do not need a
  second server. (This previously said 48 GB — measurement showed 24 GB is enough; see
  section 1.)
- Right-size from measurement, not from estimates. Following the old 48 GB recommendation
  costs roughly **150/mo more** than the 24 GB tier for capacity that measurement shows
  goes unused. Before committing to any tier, run `bash scripts/train_vit.sh --dry-run` and
  read the peak VRAM off `nvidia-smi` — it takes minutes and it is the number that decides
  the plan.

---

## Quick reference

| Task | Command |
| ---- | ------- |
| Start persistent session | `tmux new -s train` |
| Bootstrap a fresh box | `bash scripts/setup_server.sh` |
| Push code + data | `bash scripts/sync_data.sh` |
| Pull results back | `bash scripts/sync_data.sh pull` |
| Connect / watch GPU / tail log | `bash scripts/connect.sh` · `--watch` · `--logs vit` |
| Train one model | `bash scripts/train_vit.sh` (or `--dry-run`) |
| Crop full sweep | `bash scripts/train_all_gpu.sh 2>&1 \| tee logs/sweep.log` |
| Crop subset + batch override | `BATCH_VIT=16 bash scripts/train_all_gpu.sh vit` |
| Biohub pretrain then detector | `python scripts/train.py --config configs/pretrain_ext_a.yaml --pretrain` |
| Biohub bring-up + data | `scripts/bring_up_cloud.sh --with-zebrahub` |
| Watch GPU | `watch -n2 nvidia-smi` |
| Pull crop results to Mac | `rsync -avP user@ip:~/crop_disease_detection/outputs ./outputs` |
| Push biohub checkpoint | `python scripts/push_checkpoint_to_kaggle.py --checkpoint <dir> --kaggle-dataset-slug <slug>` |
| Biohub submit | `scripts/submit_kaggle.sh "message"` |
