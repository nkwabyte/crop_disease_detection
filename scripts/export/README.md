# `scripts/export/` — ExecuTorch (.pte) conversion

Converts trained checkpoints into the `.pte` artifacts the Android app loads.

## Run these on your Mac, not the GPU server

ExecuTorch shells out to **`flatc`** (the FlatBuffers compiler) when it serializes
the XNNPACK payload. Without it, export dies with:

```
FileNotFoundError: [Errno 2] No such file or directory: 'flatc'
```

The rented GPU box has no package for it, so the split is:

| Step | Where |
| ---- | ----- |
| Train | GPU server (`scripts/train_*.sh`) |
| Pull checkpoints | `bash scripts/sync_data.sh pull` |
| **Convert to .pte** | **your Mac** |

Verified working on macOS with `flatc` 25.12.19 (`brew install flatbuffers`),
torch 2.11.0 + executorch. Every script checks for `flatc` and for its
checkpoint before doing anything, so a missing prerequisite fails immediately
with the fix rather than a traceback.

## Usage

```bash
bash scripts/export/all.sh                    # everything with a checkpoint
bash scripts/export/all.sh classifier yolo    # only these

bash scripts/export/classifier.sh             # 3-class — the model the app loads
bash scripts/export/classifier.sh --variant ood   # 4-class with a learned "Other"
bash scripts/export/yolo.sh
bash scripts/export/rtdetr.sh
bash scripts/export/fasterrcnn.sh
bash scripts/export/final.sh
bash scripts/export/vit.sh
bash scripts/export/swin.sh
```

`all.sh` **skips** models with no checkpoint rather than failing, so it is safe to
run when only some models have been trained.

## The three export routes

Not every model exports the same way — that is why there is a script per model
rather than one generic one.

| Model | Route | Checkpoint |
| ----- | ----- | ---------- |
| Classifier | dedicated module — torch.export → edge dialect → XNNPACK → `.pte` | `outputs/classifier_output/best.pth` |
| YOLO26n | Ultralytics `model.export(format="executorch")` | `runs/crop_disease_yolo26/weights/best.pt` |
| RT-DETR-L | Ultralytics `model.export(format="executorch")` | `runs/rtdetr/weights/best.pt` |
| Faster R-CNN | trainer's `--export-only` | `outputs/fasterrcnn_output/checkpoints/best.pth` |
| SE-FPN final | trainer's `--export-only` | `outputs/final_output/checkpoints/best.pth` |
| ViTDet | trainer's `--export-only` | `outputs/vit_output/checkpoints/best.pth` |
| Swin | trainer's `--export-only` | `outputs/swin_output/checkpoints/best.pth` |

The torchvision detectors have no standalone export module — the exporter lives
inside each trainer, and `--export-only` loads the best checkpoint and skips
training entirely.

## Classifier variants

Two classifiers now exist (see [the comparison](../../outputs/benchmarks/classifier_variant_comparison.md)):

| Variant | Classes | Rejection | Artifact |
| ------- | ------- | --------- | -------- |
| `base` | Corn / Pepper / Tomato | softmax confidence < `CONF_DEFAULT` → unknown | `models/crop_classifier.pte` |
| `ood` | + `Other` | `argmax == "Other"` → unknown | `models/crop_classifier_ood.pte` |

They export side by side under different filenames, so the app keeps loading
`crop_classifier.pte` until you deliberately migrate it. **The two are not
drop-in compatible**: the OOD model outputs 4 logits instead of 3, and rejects by
argmax rather than by threshold, so the app's inference code has to change with
it. Each `.pte` ships a metadata YAML recording its class list and rejection
rule.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `FileNotFoundError: 'flatc'` | `brew install flatbuffers` (macOS) / `apt-get install -y flatbuffers-compiler` |
| `no checkpoint at …` | Train the model, or `bash scripts/sync_data.sh pull` to fetch it from the server |
| `Torch version 2.11.0 has not been tested with coremltools` | Harmless — coremltools is imported but unused on the ExecuTorch path |
| Export traces but produces a huge `.pte` | Expected: EfficientNet-B2 is ~29 MB fp32. Run `src/benchmark/quantize.py` for the INT8 build |

## Related

- [`scripts/README.md`](../README.md) — training, sync and server tooling
- [`src/benchmark/quantize.py`](../../src/benchmark/quantize.py) — INT8 quantization of the exported models
