# models/

Model weights live here. They are **git-ignored** (large binaries); this file explains what goes where.
TrafficGuard loads whatever is present at startup — click **Reload models** on the Model page (or
`POST /api/model/reload`) after adding files.

| Role | Expected file(s) | Required? | How to get it |
|---|---|---|---|
| **Vehicle / person / traffic-light** | `yolo11n.pt` | Yes | Downloaded automatically by Ultralytics on first run (official COCO-pretrained weights, ~5.6 MB). Any other Ultralytics detection model can be used via `TG_VEHICLE_MODEL=yolo11s.pt`. |
| **Helmet** (option A) | `yolov3-helmet.cfg`, `yolov3-helmet.weights`, `helmet.names` | Strongly recommended | Kaggle dataset [`savanagrawal/helmet-detection-yolov3`](https://www.kaggle.com/datasets/savanagrawal/helmet-detection-yolov3) — the same model the reference notebook uses. Runs through OpenCV DNN (needs `opencv-python<5`). |
| **Helmet** (option B) | `helmet.pt` | — | Train your own on a helmet dataset with images: `scripts/train.py --role helmet`. Takes priority over option A. |
| **Rider** (person on two-wheeler) | `rider.pt` *or* `yolov3_pb.cfg` + `yolov3-obj_final.weights` | Optional | Train on the Kaggle [`detect-person-on-motorbike-or-scooter`](https://www.kaggle.com/datasets/savanagrawal/detect-person-on-motorbike-or-scooter) data (`scripts/train.py --role rider`), or copy that dataset's pretrained YOLOv3 files. Used as supporting evidence for rider association. |
| **Number plate** | `plate.pt` | Optional | No plate data is in the reference datasets. Any Ultralytics model with a "license plate"/"number plate" class works; also `pip install -r requirements-optional.txt` for OCR. |

Without a helmet model the system still runs: riders are found, but helmet checks are reported as
*insufficient evidence* instead of being guessed.

## metrics/

`scripts/train.py` and `scripts/validate.py` write `metrics/<role>.json` (precision, recall, F1,
mAP@50, mAP@50-95, per-class results, training settings, hardware) and copy the training plots to
`metrics/<role>/`. The dashboard's **Model** page reads these files. If they do not exist it shows
*"Model not trained / metrics unavailable"* — numbers are never hard-coded.

The Darknet helmet weights come without test images, so no metrics can be measured for them here.

## Training a new model

```bash
python scripts/prepare_dataset.py --source ~/Downloads/detect-person-on-motorbike-or-scooter \
       --out datasets/rider --names rider
python scripts/train.py --data datasets/rider/data.yaml --role rider --epochs 60 --device 0
```

`train.py` copies `best.pt` to `models/rider.pt` and publishes the metrics automatically. A GPU is
recommended — the notebook in `notebooks/training/` runs on Kaggle's free GPU.
