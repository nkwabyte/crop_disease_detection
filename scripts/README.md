# `scripts/` — GPU server setup, sync, and training

Shell wrappers for training this project on a rented CUDA box. They exist so a full
training run never depends on remembering module paths, batch sizes, or SSH flags.

**Where each script runs matters.** Two live on your laptop and talk to the server over
SSH; the rest run *on the server*, from the repo root.

| Script | Runs on | Purpose |
| ------ | ------- | ------- |
| [`connect.sh`](#connectsh) | laptop | Open a shell on the server, or run one command on it |
| [`sync_data.sh`](#sync_datash) | laptop | Push code + datasets up, pull results back |
| [`setup_server.sh`](#setup_serversh) | server | One-time bootstrap: venv + CUDA-matched torch |
| [`train_classifier.sh`](#per-model-training) | server | Stage-1 crop classifier (EfficientNet-B2) |
| [`train_yolo.sh`](#per-model-training) | server | YOLO26n detector |
| [`train_fasterrcnn.sh`](#per-model-training) | server | Faster R-CNN v2 baseline |
| [`train_final.sh`](#per-model-training) | server | SE-FPN final detector (EMA + TTA) |
| [`train_vit.sh`](#per-model-training) | server | ViTDet (ViT-B/16) — the heaviest model |
| [`train_swin.sh`](#per-model-training) | server | Swin-V2-T detector |
| [`train_rtdetr.sh`](#per-model-training) | server | RT-DETR-L detector |
| [`train_all_gpu.sh`](#train_all_gpush) | server | Every model in sequence, then benchmark + export |

---

## Typical session

From a cold rented box to a timing estimate:

```bash
# ── on the laptop ──
bash scripts/connect.sh --setup     # install your SSH key (asks for the password once)
bash scripts/sync_data.sh           # push code + the ~7.3 GB of datasets

# ── on the server ──
bash scripts/connect.sh             # drops you into ~/crop_disease_detection
bash scripts/setup_server.sh        # venv + torch (~10 min)
bash scripts/train_classifier.sh --dry-run     # quick end-to-end smoke test
bash scripts/train_vit.sh --dry-run            # epoch-time estimate for the heaviest model

# ── back on the laptop ──
bash scripts/sync_data.sh pull      # bring outputs/ + logs/ home
```

---

## `connect.sh`

Runs on your **laptop**. Wraps SSH so you never type the host or key path.

```bash
bash scripts/connect.sh                 # interactive shell, already cd'd into the repo
bash scripts/connect.sh nvidia-smi      # run one command on the server, print output, exit
bash scripts/connect.sh --gpu           # one-shot GPU status
bash scripts/connect.sh --watch         # live nvidia-smi (Ctrl-C to stop)
bash scripts/connect.sh --tmux          # attach to the 'train' tmux session (creates it if absent)
bash scripts/connect.sh --logs vit      # tail logs/vit.log live
bash scripts/connect.sh --setup         # install your SSH key — run this first, once
```

`--setup` generates `~/.ssh/id_ed25519_gpumart` if needed, copies it to the server, and
verifies keyless login. Every later call is password-free.

Host settings come from `~/.ssh/config` under `Host gpumart`. **When the trial box expires
and you rent a new one**, either update that config block's `HostName`, or override per run:

```bash
GPU_HOST=<new-ip> bash scripts/connect.sh --setup
```

## `sync_data.sh`

Runs on your **laptop**. Moves code and data in both directions.

```bash
bash scripts/sync_data.sh          # code + datasets (default)
bash scripts/sync_data.sh code     # source only — fast; run after every local edit
bash scripts/sync_data.sh data     # datasets only (~7.3 GB the first time)
bash scripts/sync_data.sh pull     # bring outputs/ + logs/ back to the laptop
bash scripts/sync_data.sh check    # compare file counts on both ends
```

Transfers are **resumable at file level** — if the upload drops, re-run it and rsync skips
everything already copied. Safe to run repeatedly.

What gets pushed: everything except `.venv/`, `outputs/`, `runs/`, `__pycache__/`. The
datasets (`dataset/` ≈ 3.6 GB, `data/` ≈ 3.7 GB) are git-ignored, so they only reach the
server through this script.

> macOS ships **openrsync**, not GNU rsync, so these commands stick to flags both it and
> the server's rsync 3.x understand. That is why there is no `--progress` / `--partial`.

## `setup_server.sh`

Runs on the **server**, once per box. Creates `.venv` and installs a torch build that
matches the actual GPU.

```bash
bash scripts/setup_server.sh
TORCH_INDEX=https://download.pytorch.org/whl/cu129 bash scripts/setup_server.sh
PYBIN=python3.12 bash scripts/setup_server.sh
```

It handles two things a plain `pip install -r requirements-server.txt` gets wrong:

1. **Blackwell GPUs (RTX 5090/5080, compute capability `sm_120`) need CUDA 12.8+.**
   `torch.cuda.is_available()` returns `True` even when the wheel has no `sm_120` kernels —
   it only fails on the first real operation. The script installs from the cu128 index and
   then runs an actual fp16 matmul to prove the kernels exist.
2. **Batch defaults in `src/*/config.py` assume a 96 GB card.** The script reads the GPU's
   real VRAM and prints the `BATCH_*` overrides to use on smaller hardware.

Idempotent — re-running reuses the existing venv and just re-verifies.

## Per-model training

One script per model, all following the same contract. They are deliberately
self-contained: each carries its own batch-sizing table rather than sourcing a shared
helper, so editing or deleting one never affects the others.

```bash
bash scripts/train_vit.sh                 # full run
bash scripts/train_vit.sh --dry-run       # short timing estimate (2 epochs; 1 for YOLO/RT-DETR)
bash scripts/train_vit.sh --epochs 10     # any trainer flag passes straight through
BATCH=6 bash scripts/train_vit.sh         # override the auto-picked batch size
bash scripts/train_vit.sh --batch-size 6  # equivalent; an explicit flag always wins
```

Every script:

- **Picks a batch size from the GPU's actual VRAM.** The values in `src/*/config.py` were
  tuned for a 96 GB RTX PRO 6000 and will OOM on a 32 GB card, so each script maps VRAM to
  a safe batch (see the table below).
- **Logs to `logs/<model>.log`** via `tee`, so you see output live *and* keep a record.
- **Is resume-aware** — re-running continues from the last checkpoint rather than restarting.
- **Reports wall-clock** on completion, and suggests a smaller batch on failure.
- **Exits with the trainer's exit code**, so it composes in a larger pipeline.

| Script | Module | 96 GB | **32 GB (5090)** | 24 GB | Source |
| ------ | ------ | ----- | ---------------- | ----- | ------ |
| `train_classifier.sh` | `src.classifier.train_classifier` | 128 | **128** | 64 | estimated |
| `train_yolo.sh` | `src.yolo.train` | 64 | **32** | 24 | estimated |
| `train_fasterrcnn.sh` | `src.fasterrcnn.train_fasterrcnn` | 16 | **8** | 4 | estimated |
| `train_final.sh` | `src.fasterrcnn.train_final` | 16 | **8** | 4 | estimated |
| `train_vit.sh` | `src.vit.train_vit` | 8 | **16** | 2 | **measured** |
| `train_swin.sh` | `src.swin.train_swin` | 16 | **16** | 4 | **measured** |
| `train_rtdetr.sh` | `src.rtdetr.train_rtdetr` | 16 | **8** | 4 | estimated |

Only the two rows marked *measured* have been verified on real hardware (see
[Measured performance](#measured-performance)). The rest are deliberately conservative
starting points. Watch `nvidia-smi` during the first epoch: if VRAM sits below ~70 % used,
raise the batch with `BATCH=<n>` for faster epochs.

## Measured performance

RTX 5090 (31.4 GB), 32 cores, on a 10 % subset (4,076 train / 573 val), 2 epochs each:

| Model | Batch | Backbone | 2 epochs | min/epoch | Peak VRAM | GPU util |
| ----- | ----- | -------- | -------- | --------- | --------- | -------- |
| ViT-B/16 @640 | 4 | frozen | 5.0 min | 2.50 | 1.7 GB | — |
| ViT-B/16 @640 | 4 | unfrozen | 4.2 min | 2.10 | 4.9 GB | 37 % avg |
| ViT-B/16 @640 | 16 | unfrozen | 3.3 min | 1.65 | 11.9 GB | 59 % avg, 99 % peak |
| Swin-V2-T @640 | 16 | frozen | 3.1 min | 1.55 | 4.5 GB | 29 % avg |

Extrapolated to the full 40,852-image train split (×10.02), ViT at batch 16 runs about
**16.5 min/epoch, ~11 h for 40 epochs**; at batch 4 it is ~21 min/epoch, ~14 h.

### Training is input-bound, not compute-bound

The clearest evidence: the *unfrozen* ViT run was **faster** than the frozen one (4.2 vs
5.0 min). Backpropagating through the backbone cannot be cheaper than skipping it — the
difference is page cache, since the first run read every image cold from disk. GPU
utilisation swinging 11 %→99 % and power drawing ~270 W of a 575 W TDP say the same thing.

So the GPU idles a large fraction of the time, and buying a faster one buys little. The
levers that would actually help, in order:

1. **Dataloader workers.** Every trainer caps them at `min(16, cpu_count)` — on this
   32-core box, half the cores sit idle. Raising the cap is a one-line change.
2. **Prefetch depth.** The loaders set `pin_memory` and `persistent_workers` but leave
   `prefetch_factor` at the default 2.
3. **Decode cost.** Images are already 640×640, so workers spend their time on JPEG decode.
   Decoding on the GPU (DALI / `nvjpeg`) or caching decoded tensors would remove it.

Batch size is *not* the main lever: 4×-ing it only bought 1.27× throughput, exactly what
you would expect when the GPU is waiting on input.

### Re-measuring

`--dry-run` is 2 epochs, but `FREEZE_BACKBONE_EPOCHS=5` in the ViT and Swin configs — so a
plain dry-run **never unfreezes the backbone** and understates both VRAM and epoch time for
the phase that dominates a real run. To measure the unfrozen phase, temporarily set
`FREEZE_BACKBONE_EPOCHS = 0` in the model's `config.py`, run the dry-run, then restore it.

Two scripts do extra work around training:

- `train_classifier.sh` generates `dataset/classifier_*.csv` first if they are missing, and
  runs the ExecuTorch export afterwards.
- `train_yolo.sh` and `train_rtdetr.sh` run their export step after a successful full run.

Set `SKIP_EXPORT=1` to train without exporting. Dry-runs never export. The remaining
detectors (Faster R-CNN, SE-FPN, ViT, Swin) export at the end of their own training.

### Running unattended

Training runs for hours, so start them under `tmux` — the run then survives a dropped SSH
connection:

```bash
tmux new -s vit 'bash scripts/train_vit.sh'
#   Ctrl-b then d   to detach
#   tmux attach -t vit   to return
```

From the laptop you can watch without attaching:

```bash
bash scripts/connect.sh --logs vit
```

## `train_all_gpu.sh`

Runs on the **server**. The full unattended sweep: classifier → all detectors → benchmark →
quantize → latency. Each step logs to `logs/<step>.log` and a failure in one step does not
abort the rest.

```bash
bash scripts/train_all_gpu.sh                     # everything
bash scripts/train_all_gpu.sh vit swin            # only these steps
BATCH_VIT=4 BATCH_SWIN=8 bash scripts/train_all_gpu.sh vit swin
```

Note it takes `BATCH_<MODEL>` env vars, whereas the per-model scripts take plain `BATCH`.
Unlike the per-model scripts, `train_all_gpu.sh` does **not** auto-scale to VRAM — on a card
smaller than 96 GB, pass the `BATCH_*` values that `setup_server.sh` printed.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `CUDA out of memory` | Re-run with a smaller `BATCH=<n>`. The script prints a suggestion on failure. |
| `CUDA error: no kernel image is available` | Wrong torch build for the GPU. Re-run `setup_server.sh`, or pin a newer `TORCH_INDEX`. |
| `Cannot reach gpumart` | Trial box expired or the IP changed — `GPU_HOST=<new-ip> bash scripts/connect.sh --setup`. |
| GPU sits below 60 % utilised | Batch is too small; raise `BATCH`. Check the dataloader worker count too. |
| Training died when SSH dropped | Use `tmux` (see [Running unattended](#running-unattended)). |
| `executorch` failed to install | Export-only dependency — training is unaffected. Export on a machine that has it. |

## Related docs

- [`docs/09_gpu_server.md`](../docs/09_gpu_server.md) — server guide: hardware, data placement, monitoring
- [`docs/07_dataset.md`](../docs/07_dataset.md) — obtaining the datasets
