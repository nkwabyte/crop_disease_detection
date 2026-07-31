# Training on the GPU Server

Guide for running the full training + benchmark + export sweep on the rented CUDA box.

## Target hardware

| Component | Spec |
| --------- | ---- |
| GPU | 1× NVIDIA RTX PRO 6000 — **96 GB VRAM**, ~119 TFLOPS, 1401.9 GB/s |
| CPU | AMD EPYC 9654 (48 cores allocated) |
| RAM | ~129 GB |
| Disk | Micron 7450 NVMe (~740 GB, ~16 GB/s) |
| Driver / CUDA | up to CUDA 13.3 |
| OS | Linux |

The code auto-detects CUDA and scales to it — no code edits needed on the server.

## What auto-scales on CUDA

All training scripts detect the device and switch defaults (the M4 Pro / MPS values in
parentheses stay in effect on the laptop):

| Model | CUDA batch (was MPS) | Also enabled on CUDA |
| ----- | -------------------- | -------------------- |
| Classifier (EfficientNet-B2) | **128** (64) | AMP + GradScaler, cudnn.benchmark, pin_memory, 16 workers |
| YOLO26n | **64/GPU** (32) | AMP, 16 workers, DDP if >1 GPU |
| Faster RCNN v2 | **16** (4) | AMP + GradScaler, cudnn.benchmark, pin_memory, 16 workers |
| SE-FPN final | **16** (4, eff. 32 w/ accum) | same as above |
| ViTDet (ViT-B/16) | **8** (2) | AMP autocast, pin_memory, 16 workers |
| Swin (Swin-V2-T) | **16** (2) | same |
| RT-DETR-L | **16** (4) | Ultralytics AMP, 16 workers |

Every script takes `--batch-size N` to override. **96 GB is large** — these defaults are
conservative starting points; once a run is stable, watch `nvidia-smi` and push higher
(e.g. Swin/RCNN/RT-DETR to 24–32, ViT to 12–16, classifier to 256) for faster epochs.

## 1. Setup

```bash
git clone <repo-url> crop_disease_detection && cd crop_disease_detection
git checkout dev

python3 -m venv .venv                       # Python 3.11–3.13 (torch 2.11 has no 3.14 wheels)
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-server.txt   # torch pulls the CUDA build on Linux

# verify the GPU is visible
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `executorch` fails to install (export-only dependency), training still runs — install
it later on whatever machine builds the mobile `.pte` artifacts.

## 2. Get the data onto the server

The datasets are git-ignored (large). Place them exactly where the configs expect:

```text
dataset/train/  dataset/validate/  dataset/test/      # Faster RCNN / ViT / Swin / classifier
dataset/final_{train,validate,test}_labels.csv        # (already tracked in the repo)
data/main/{train,valid,test}/{images,labels}/         # YOLO / RT-DETR (YOLO format)
```

Get them via the Roboflow download in [`docs/07_dataset.md`](07_dataset.md), or copy from
the laptop, e.g.:

```bash
rsync -avP dataset/train dataset/validate dataset/test  user@server:~/crop_disease_detection/dataset/
rsync -avP data/main                                    user@server:~/crop_disease_detection/data/
```

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
./.venv/bin/python -m src.fasterrcnn.train_fasterrcnn
./.venv/bin/python -m src.fasterrcnn.train_final
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
