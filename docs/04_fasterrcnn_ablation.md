# Faster RCNN Ablation Study

**Script:** `train_alt_fasterrcnn.py`  
**Output directory:** `alt_fasterrcnn_output/`  
**Role:** Systematic comparison of backbone depth, proposal count, NMS policy, and anchor scale to justify the final model architecture

---

## 1. Purpose

Before committing to the SE-FPN final model, a structured ablation study was conducted to identify which architectural choices most affect detection performance on the Ghana Crop Disease dataset. The study mirrors the comparative analysis style of the original Faster RCNN paper (Ren et al., 2015) and produces paper-ready figures and a LaTeX table.

Seven configurations are trained and evaluated under identical conditions (same optimizer, LR schedule, epoch budget, augmentation, and dataset). The results directly inform the architecture decisions made in `train_final.py`.

---

## 2. Configurations

| # | Config ID | Backbone | Proposals | NMS Thresh | Anchor Sizes | Role |
|---|---|---|---|---|---|---|
| 1 | `mobilenet_300` | MobileNetV3-Large-FPN (~19 M params) | 300 | 0.7 | Default | Lightweight baseline |
| 2 | `resnet50_100` | ResNet50-FPN-v2 (~43 M params) | 100 | 0.7 | Default | Low-proposal ablation |
| 3 | `resnet50_300` | ResNet50-FPN-v2 (~43 M params) | 300 | 0.7 | Default | **Selected baseline ★** |
| 4 | `resnet50_1000` | ResNet50-FPN-v2 (~43 M params) | 1000 | 0.7 | Default | High-proposal ablation |
| 5 | `resnet50_no_nms` | ResNet50-FPN-v2 (~43 M params) | 300 | 1.0 | Default | NMS disabled |
| 6 | `resnet50_small_anchors` | ResNet50-FPN-v2 (~43 M params) | 300 | 0.7 | 16–256 px | Disease-lesion optimised |
| 7 | `resnet101_300` | ResNet101-FPN (~60 M params) | 300 | 0.7 | Default | Deeper backbone |

**Config 3 (`resnet50_300`) is the selected model** — it provides the best balance of mAP, inference speed, and parameter count, and serves as the direct baseline for the SE-FPN final model.

---

## 3. What Each Ablation Tests

### Backbone depth (configs 1, 3, 7)

Compares three backbone complexities:
- `mobilenet_300` — tests whether a mobile-weight model is competitive at disease detection, where fine-grained texture discrimination matters.
- `resnet50_300` — the standard Faster RCNN backbone, well-studied and pre-trained on COCO.
- `resnet101_300` — a deeper ResNet with higher representational capacity at the cost of speed and memory.

### Proposal count (configs 2, 3, 4)

The RPN generates region proposals that are passed to the RoI head. Fewer proposals = faster inference but more missed detections; more proposals = higher recall but slower inference and more false positives before NMS.
- Disease lesions can be small and numerous — it is not obvious whether 100 or 1000 proposals optimises mAP@0.5 on this dataset.
- Results from this ablation inform the proposal count used in the final model.

### NMS threshold (configs 3, 5)

Standard NMS with threshold 0.7 is compared against disabling NMS (threshold=1.0). This tests whether overlapping boxes from the RPN are genuinely harmful on crop images or whether keeping all proposals could increase recall.

### Anchor scale (configs 3, 6)

Default COCO anchor sizes (32–512 px) are tuned for general objects. Crop disease lesions are often small (10–80 px). Config 6 uses smaller anchors (16–256 px) that better match the lesion size distribution observed in the training set.

---

## 4. Training Configuration (All Configs)

| Parameter | Value | Notes |
|---|---|---|
| Epochs per config | 15 | Shorter than final pipeline — ablation is comparative, not maximal |
| Early stopping | patience = 5 | Halts quickly on plateaus for efficiency |
| Batch size | 4 | Same as baseline for fair comparison |
| Optimizer | SGD, momentum = 0.9 | Identical across all configs |
| Learning rate | 5e-3 → cosine decay | Linear warmup for 2 epochs |
| Gradient clip | 10.0 | |
| Hard negatives | 100 images | Shared cache with `train_fasterrcnn.py` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 |
| Random seed | 42 | Fixed for reproducibility |
| Speed benchmark | 100 runs, batch = 1 | Full model + backbone-only latency |
| AMP | CUDA only | Same as baseline script |
| Workers | 0 (MPS) / 8 (CUDA) | |

---

## 5. Usage

### Architecture figures only (no training)

```bash
python train_alt_fasterrcnn.py --arch-figures
```

This generates 5 architecture diagrams instantly to `alt_fasterrcnn_output/figures/` — no dataset or GPU required.

### Full ablation (all 7 configs)

```bash
python train_alt_fasterrcnn.py --dry-run   # estimate total time first
python train_alt_fasterrcnn.py             # train all 7 configs sequentially
```

### Single config

```bash
python train_alt_fasterrcnn.py --configs resnet50_300
python train_alt_fasterrcnn.py --configs resnet50_100 resnet50_1000
```

### All commands

| Command | Purpose |
|---|---|
| `python train_alt_fasterrcnn.py` | Train all 7 configs |
| `python train_alt_fasterrcnn.py --configs ID [ID …]` | Train specific config(s) |
| `python train_alt_fasterrcnn.py --epochs 10` | Override epoch count per config |
| `python train_alt_fasterrcnn.py --dry-run` | 2-epoch timing estimate per config |
| `python train_alt_fasterrcnn.py --arch-figures` | Architecture diagrams only (instant) |
| `python train_alt_fasterrcnn.py --figures-only` | All comparison figures from `results.json` |
| `python train_alt_fasterrcnn.py --skip-negatives` | Reuse cached hard-negative images |
| `python train_alt_fasterrcnn.py --no-figures` | Train without generating figures |

---

## 6. Resume Behaviour

Each config maintains its own checkpoint directory:

```text
alt_fasterrcnn_output/checkpoints/
├── mobilenet_300/    last.pth  best.pth  epoch_NNNN.pth
├── resnet50_100/     last.pth  best.pth  …
├── resnet50_300/     last.pth  best.pth  …
└── resnet101_300/    last.pth  best.pth  …
```

Re-running the script resumes each unfinished config automatically. Configs that have already completed are skipped.

To force a fresh run for one config:

```bash
rm -rf alt_fasterrcnn_output/checkpoints/resnet101_300
python train_alt_fasterrcnn.py --configs resnet101_300
```

---

## 7. Output Files

### Architecture figures (no training required)

| File | Contents |
|---|---|
| `fig_arch_01_pipeline.png` | End-to-end detection pipeline: Input → Backbone → RPN → RoI Head → Output |
| `fig_arch_02_backbone_comparison.png` | Parameter count, COCO mAP, and depth for all 3 backbones |
| `fig_arch_03_rpn_detail.png` | RPN architecture: anchor generation, classification/regression heads, NMS |
| `fig_arch_04_anchor_visualization.png` | Default vs small anchor scales at 3 aspect ratios |
| `fig_arch_05_fpn_structure.png` | Feature Pyramid Network: bottom-up and top-down pathways |

### Performance comparison figures (require training)

| File | Contents |
|---|---|
| `fig_cmp_01_map_bar.png` | mAP@0.5 bar chart across all 7 configurations |
| `fig_cmp_02_speed_accuracy.png` | FPS vs mAP scatter plot (speed–accuracy trade-off) |
| `fig_cmp_03_convergence.png` | Training loss and val mAP curves per config |
| `fig_cmp_04_proposal_ablation.png` | mAP and FPS vs proposal count (100 / 300 / 1000) |
| `fig_cmp_05_radar.png` | Normalised multi-metric radar chart |
| `fig_cmp_06_params_vs_map.png` | Model complexity vs detection accuracy |
| `fig_cmp_07_inference_breakdown.png` | Backbone vs RPN+Head latency stacked bar |
| `fig_cmp_08_nms_anchor_ablation.png` | NMS-off vs NMS-on; small anchors vs default anchors |
| `fig_tbl_01_main_results.png` | Paper-style Table 1: all configurations |
| `fig_tbl_02_speed_comparison.png` | Paper-style Table 2: latency breakdown |
| `table_ablation.tex` | LaTeX source for direct inclusion in the paper |

### Per-config model weights

```text
alt_fasterrcnn_output/models/
├── mobilenet_300_best.pth
├── resnet50_100_best.pth
├── resnet50_300_best.pth
└── …
```

### Aggregated results

`alt_fasterrcnn_output/results.json` stores the final metrics for all configs and is used as the sole source for figure generation via `--figures-only`.

---

## 8. Findings Summary

Results from the ablation study (on the Ghana Crop Disease Challenge validation set):

| Finding | Implication for final model |
|---|---|
| `resnet50_300` consistently outperforms `mobilenet_300` | ResNet-50 backbone retained in final model |
| `resnet101_300` improves mAP marginally but is 1.4× slower | ResNet-50 retained for speed–accuracy balance |
| 300 proposals is optimal; 100 reduces recall noticeably | 300 proposals used in final model |
| 1000 proposals adds latency without consistent mAP gain | 300 proposals confirmed |
| Disabling NMS increases FP rate significantly | NMS threshold 0.7 retained |
| Small anchors (16–256 px) improve mAP on small lesions | Informs K-means anchor clustering in final model |

These findings directly motivate the SE-FPN design choices described in [05_sefpn_final_model.md](05_sefpn_final_model.md).

---

## 9. References

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015.
- Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
