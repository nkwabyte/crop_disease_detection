# Model Benchmark — Crop Disease Detection

## Detection models (Stage 2)

| Model | Architecture | mAP@0.5 | Params (M) | Source |
| ----- | ------------ | ------- | ---------- | ------ |
| `rtdetr-l` | RT-DETR-L (query head, no NMS) | 0.345 | — | outputs/rtdetr_output/crop_disease_rtdetr/results.csv |
| `yolo26m` | YOLO26m (Ultralytics) | 0.310 | 21.8 | runs/crop_disease_yolo26m/results.csv |
| `yolo26s` | YOLO26s (Ultralytics) | 0.305 | 10.0 | runs/crop_disease_yolo26s/results.csv |
| `yolo26n` | YOLO26n (Ultralytics) | 0.290 | 2.5 | runs/crop_disease_yolo26/results.csv |

## Stage-1 classifier (reported separately — classification task)

| Model | Best val acc | Best val F1 |
| ----- | ------------ | ----------- |
| EfficientNet-B2 (Stage-1 crop classifier) | 0.98047 | None |
