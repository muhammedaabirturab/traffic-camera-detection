# Model

## Why YOLO

YOLO ("You Only Look Once") is a single-stage detector: one forward pass of a convolutional network
predicts class scores and bounding boxes for the whole image, which makes it fast enough for video on a
laptop. This project uses **Ultralytics YOLO11** (nano variant by default) — a CSP-style backbone, a
feature-pyramid neck and an anchor-free decoupled detection head — plus the original **Darknet YOLOv3**
models from the Kaggle reference for helmet/rider detection.

## Detectors

| Role | Model | Classes used | Trained on |
|---|---|---|---|
| Vehicle / person / signal | `yolo11n.pt` (Ultralytics, 2.6 M parameters) | person, bicycle, car, motorcycle, bus, truck, traffic light | COCO 2017 (official weights) |
| Helmet | YOLOv3 (Darknet) `yolov3-helmet.weights` **or** YOLO11 `helmet.pt` | `Helmet` (+ bare head if your dataset has it) | Kaggle `helmet-detection-yolov3` (pretrained) / your helmet dataset |
| Rider | YOLO11 `rider.pt` **or** YOLOv3 `yolov3-obj_final.weights` | person on two-wheeler | Kaggle `detect-person-on-motorbike-or-scooter` |
| Plate (optional) | YOLO11 `plate.pt` | number plate | not provided |

## Training pipeline

```
Kaggle dataset ──► scripts/prepare_dataset.py ──► datasets/<role>/ (train/val/test + data.yaml)
                                                     │
                                                     ▼
                         scripts/train.py  (transfer learning from yolo11n.pt)
                                                     │
                     best.pt evaluated on held-out TEST split (Ultralytics val)
                                                     │
              ┌──────────────────────────────────────┴───────────────────────────────┐
              ▼                                                                      ▼
     models/<role>.pt                                     models/metrics/<role>.json + plots
     (loaded by the API)                                  (shown on the Model page)
```

* **Split** — 75 / 15 / 10 % by default, grouped so that an image and its `__flip` copy never end up in
  different splits (prevents leakage).
* **Transfer learning** — training starts from COCO weights; Ultralytics' default augmentation
  (mosaic, HSV jitter, flips, scaling) is used; early stopping with `--patience 20`.
* **Metrics** (`scripts/_metrics.py`)
  * Precision = TP / (TP + FP), Recall = TP / (TP + FN)
  * F1 = 2PR / (P + R)
  * mAP@50 — mean average precision with IoU ≥ 0.5 counted as a match
  * mAP@50-95 — mAP averaged over IoU thresholds 0.50 … 0.95 (stricter localisation)
  * per-class values, confusion matrix, PR/F1 curves and loss curves are saved as plots.

## Commands

```bash
python scripts/prepare_dataset.py --source <kaggle folder> --out datasets/rider --names rider
python scripts/train.py    --data datasets/rider/data.yaml --role rider --epochs 60 --device 0
python scripts/validate.py --weights models/rider.pt --data datasets/rider/data.yaml --role rider
python scripts/predict.py  path/to/image_or_video
python scripts/export_model.py --weights models/rider.pt --format onnx
```

## Results

**No metrics are published in this repository.** The models were not trained in the development
environment (the Kaggle data could not be downloaded there), and the COCO and reference YOLOv3 weights
were not re-evaluated on project data. After you train, the Model page shows the measured values read
from `models/metrics/*.json`. Report those numbers — together with the split they were measured on —
in your presentation, not figures from elsewhere.
