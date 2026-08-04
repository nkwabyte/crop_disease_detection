# Training on the GPU Server

Guide for running the full training + benchmark + export sweep on the rented CUDA box.

For the scripts that automate all of this, see [`scripts/README.md`](../scripts/README.md).
The short version:

```bash
bash scripts/connect.sh --setup    # laptop: install your SSH key, once
bash scripts/sync_data.sh          # laptop: push code + data
bash scripts/setup_server.sh       # server: venv + a CUDA-matched torch
bash scripts/train_vit.sh --dry-run  # server: timing estimate
```

## Reference hardware

The batch defaults in `src/*/config.py` were tuned for the first column. **Anything
smaller needs the batch sizes lowered** — `setup_server.sh` reads the GPU's real VRAM
and prints the values to use, and the per-model `scripts/train_*.sh` apply them
automatically.

| Component | RTX PRO 6000 (config default) | RTX 5090 (verified 2026-08-04) |
| --------- | ----------------------------- | ------------------------------ |
| GPU | 96 GB VRAM, ~119 TFLOPS | **31.4 GB usable**, Blackwell |
| Compute capability | sm_90 | **sm_120** |
| CPU | AMD EPYC 9654 (48 cores) | 32 cores |
| RAM | ~129 GB | 82 GB |
| Disk | Micron 7450 NVMe (~740 GB) | 398 GB (355 GB free) |
| OS / driver | Linux, up to CUDA 13.3 | Ubuntu 24.04, driver 575.51.02 |

The code auto-detects CUDA and scales to it — no code edits needed on the server.

## Blackwell (RTX 5090 / 5080) — read before installing torch

Blackwell cards report **compute capability sm_120** and need a torch built against
CUDA 12.8 or newer. The trap: `torch.cuda.is_available()` returns `True` even when the
installed wheel contains no sm_120 kernels — the failure only appears at the first real
operation, as `CUDA error: no kernel image is available for execution on the device`.

`scripts/setup_server.sh` handles this: it installs from the cu128 index and then runs an
actual fp16 matmul to prove the kernels exist. A correct install reports:

```text
torch 2.11.0+cu128  (CUDA 12.8)
Device      : NVIDIA GeForce RTX 5090
Capability  : sm_120
Wheel archs : sm_75 sm_80 sm_86 sm_90 sm_100 sm_120   ← sm_120 must appear here
OK - fp16 matmul on GPU succeeded.
```

If `sm_120` is missing from the arch list, reinstall with a newer index:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu129 bash scripts/setup_server.sh
```

## What auto-scales on CUDA

All training scripts detect the device and switch defaults (the M4 Pro / MPS values in
parentheses stay in effect on the laptop):

| Model | 96 GB default (MPS) | **32 GB (5090)** | Also enabled on CUDA |
| ----- | ------------------- | ---------------- | -------------------- |
| Classifier (EfficientNet-B2) | **128** (64) | 128 | AMP + GradScaler, cudnn.benchmark, pin_memory, 16 workers |
| YOLO26n | **64/GPU** (32) | 32 | AMP, 16 workers, DDP if >1 GPU |
| Faster RCNN v2 | **16** (4) | 8 | AMP + GradScaler, cudnn.benchmark, pin_memory, 16 workers |
| SE-FPN final | **16** (4, eff. 32 w/ accum) | 8 | same as above |
| ViTDet (ViT-B/16) | **8** (2) | 4 | AMP autocast, pin_memory, 16 workers |
| Swin (Swin-V2-T) | **16** (2) | 8 | same |
| RT-DETR-L | **16** (4) | 8 | Ultralytics AMP, 16 workers |

Every script takes `--batch-size N` to override, and the `scripts/train_*.sh` wrappers pick
the right column automatically from the GPU's VRAM. On a 96 GB card these are conservative
starting points — once a run is stable, watch `nvidia-smi` and push higher (Swin/RCNN/RT-DETR
to 24–32, ViT to 12–16, classifier to 256). On 32 GB, treat the middle column as the ceiling
until you have seen a full epoch fit.

## 1. Setup

Use the bootstrap script — it picks a torch build matching the GPU's compute capability,
which a plain `pip install` does not do (see the Blackwell note above):

```bash
bash scripts/setup_server.sh
```

<details>
<summary>Manual equivalent</summary>

```bash
git clone <repo-url> crop_disease_detection && cd crop_disease_detection
git checkout dev

python3 -m venv .venv                       # Python 3.11–3.13 (torch 2.11 has no 3.14 wheels)
./.venv/bin/pip install --upgrade pip
# On Blackwell (sm_120) the index URL is required, not optional:
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0
./.venv/bin/pip install -r requirements-server.txt

# verify the GPU actually computes — is_available() alone is not proof
./.venv/bin/python -c "import torch; x=torch.randn(64,64,device='cuda'); print((x@x).sum())"
```

</details>

If `executorch` fails to install (export-only dependency), training still runs — install
it later on whatever machine builds the mobile `.pte` artifacts.

## 2. Get the data onto the server

**The whole `data/` tree is git-ignored** — images *and* annotations. Nothing here comes
with a clone, so it must be synced or downloaded onto the server, and you should keep your
own backup of the label CSVs.

```text
data/detector/{train,validate,test}/{Corn,Pepper,Tomato}/*.jpg   # Faster RCNN / ViT / Swin
data/detector/final_{train,validate,test}_labels.csv             # annotations — not in git
data/detector/label_map.json
data/yolo/{train,valid,test}/{images,labels}/                    # YOLO / RT-DETR, and the
data/yolo/classifier_{train,valid,test}.csv                      #   classifier's source
data/negatives/{images,eggplant,millet,potato,sorghum,tobacco,ood}/
```

Detector images are grouped into a per-crop subdirectory; the loaders read the crop from
the CSV's `crop` column and build the path from it.

Get them via the Roboflow download in [`docs/07_dataset.md`](07_dataset.md), or push from
the laptop:

```bash
bash scripts/sync_data.sh data      # ~4.1 GB, resumable
```

### How much actually needs to move

`data/` measures ~7.7 GB on disk, but only **~4.1 GB needs uploading**:

| Path | On disk | Upload | Why |
| ---- | ------- | ------ | --- |
| `data/detector/` (58,361 JPEGs) | 3.79 GB | 3.79 GB | All referenced by the label CSVs — nothing to prune |
| `data/yolo/` JPEG + `.txt` labels | 0.29 GB | 0.29 GB | The actual YOLO / RT-DETR training data |
| `data/negatives/` | 0.5 GB | 0.5 GB | Hard negatives + non-target foliage |
| `data/**/*.npy` | 3.60 GB | — | Ultralytics `cache="disk"` sidecars; regenerated on epoch 1 |
| `data/**/*.cache` | 1.6 MB | — | Label index embedding absolute paths; must be rebuilt anyway |

`sync_data.sh` already excludes both. Shipping the `.npy` caches would nearly double the
transfer for files the server overwrites on its first epoch.

### When the upload link is slow

A home connection can make this the longest step by far — at 1.4 Mbps, 4.1 GB takes about
6.7 hours. To get real throughput numbers without waiting for the whole thing:

```bash
bash scripts/sync_data.sh subset    # ~10% stratified slice, ~380 MB
```

Epoch time scales linearly with image count, so `sec/iter` measured on the slice
extrapolates to full-dataset epoch estimates exactly. Run the full `sync_data.sh data`
afterwards — rsync skips the files the subset already placed, so nothing is re-sent.

Hard-negative images download automatically on first run (`--skip-negatives` to reuse).

## 3. Train

**Everything, in order, unattended** (run inside tmux so it survives disconnects):

```bash
tmux new -s train
bash scripts/train_all_gpu.sh 2>&1 | tee logs/sweep.log
#   Ctrl-b d to detach;  tmux attach -t train to return
```

`train_all_gpu.sh` runs the classifier → all detectors → benchmark → quantize → latency,
logging each step to `logs/<step>.log` and continuing past any single failure. Run a
subset by naming steps, and override batches via env vars:

```bash
bash scripts/train_all_gpu.sh vit swin              # only these
BATCH_VIT=12 BATCH_SWIN=24 bash scripts/train_all_gpu.sh vit swin
```

**Or per model** (each has `--dry-run` for a quick timing estimate first):

```bash
./.venv/bin/python -m src.classifier.generate_classifier_csv
./.venv/bin/python -m src.classifier.train_classifier
./.venv/bin/python -m src.yolo.train
./.venv/bin/python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline
./.venv/bin/python -m src.fasterrcnn.faster_rcnn_final
# the 7-config ablation is not part of the sweep — run it explicitly when needed:
./.venv/bin/python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation
./.venv/bin/python -m src.vit.train_vit --batch-size 12
./.venv/bin/python -m src.swin.train_swin --batch-size 24
./.venv/bin/python -m src.rtdetr.train_rtdetr --batch-size 24
```

All are resume-aware — re-running the same command continues from the last checkpoint.

## 4. Monitor

```bash
watch -n2 nvidia-smi          # GPU utilisation + memory (aim to fill VRAM without OOM)
tail -f logs/vit.log          # live training log
```

If a run OOMs, lower its `--batch-size`; if the GPU sits <60 % utilised, raise it.

## 5. Benchmark, quantize, export

Run automatically at the end of the sweep, or individually:

```bash
./.venv/bin/python -m src.benchmark.compare_models   # accuracy table + figures
./.venv/bin/python -m src.benchmark.quantize         # INT8 .pte (backbones now have checkpoints)
./.venv/bin/python -m src.benchmark.latency          # size + latency table
```

Detectors export at the end of their own training; re-export any time with
`--export-only` (Faster RCNN / ViT / Swin) or the dedicated `export_*` scripts (YOLO,
RT-DETR, classifier). Quantization needs `flatc` — it's auto-located from the bundled
ExecuTorch copy, or `apt-get install -y flatbuffers-compiler` / `pip install flatbuffers`.

## Notes

- **CUDA 13.x driver + torch 2.11 cu12 wheels**: works via forward compatibility; no
  special index URL needed. Only pin a CUDA index if you deliberately want a cu13 build.
- **Single GPU**: YOLO uses DDP only when it sees >1 GPU, so nothing special here.
- **Long runs**: max rental duration is generous; still checkpoint-resume so a preempt or
  disconnect never loses more than one epoch.
- **Outputs** (`outputs/`, `runs/`) and the trained checkpoints are git-ignored — pull the
  ones you want back to the laptop with `rsync`, or run export on the server and copy the
  `models/*.pte` / `*.onnx` artifacts to the app.
