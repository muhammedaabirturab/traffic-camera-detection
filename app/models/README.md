# Model weights

| File | Purpose | In git? | How you get it |
|---|---|---|---|
| `rider_detector.pt` | YOLOv8n fine-tuned on the `person_bike` dataset (1 class) | yes (~6 MB) | `python scripts/train.py` (or use the committed file) |
| `rider_detector.metrics.json` | Real precision / recall / mAP / training info for the file above | yes | written by `scripts/train.py` |
| `yolov8n.pt` | Official COCO-pretrained YOLOv8 nano: cars, buses, trucks, motorcycles, persons | no | auto-downloaded by Ultralytics on first run, or place it here manually |
| `helmet_detector.pt` | YOLOv8n fine-tuned on EdgeVision: `bike_with_rider`, `no_helmet`, `helmet` | yes (~6 MB) | `python scripts/train.py --data datasets/prepared_helmet/data.yaml --name helmet_detector` |
| `helmet_detector.metrics.json` | Real metrics for the helmet model | yes | written by `scripts/train.py` |

Override any path with `MODEL_PATH`, `COCO_MODEL_PATH`, `HELMET_MODEL_PATH` in `.env`.

## Train / retrain the rider detector

```bash
python scripts/prepare_dataset.py --source <folder with images + YOLO .txt labels>
python scripts/train.py --epochs 50 --batch 8
```

Training copies the best weights here and writes `rider_detector.metrics.json`; the **Model** page reads it.

## Retraining the helmet detector

1. Download the EdgeVision Dataset (CC BY 4.0): <https://data.mendeley.com/datasets/j82bnw7gsr/1> ("Download All", ~1.5 GB zip) and unzip it.
2. Prepare and train (class order must match the dataset: `BikeWithRider`, `NoHelmet`, `Helmet`):
   ```bash
   python scripts/prepare_dataset.py --source "<unzipped>/images" --labels "<unzipped>/labels/yolo"        --names bike_with_rider no_helmet helmet --max-side 800 --out datasets/prepared_helmet
   python scripts/train.py --data datasets/prepared_helmet/data.yaml --name helmet_detector --epochs 25 --patience 10 --batch 8
   ```
3. Class names are matched against `HELMET_CLASS_NAMES` / `NO_HELMET_CLASS_NAMES` in `.env` (defaults cover `helmet`, `no_helmet`, ...).
   To use a different helmet dataset, name its classes accordingly or adjust those variables. Delete `helmet_detector.pt` to switch helmet rules back to *not assessed*.

Only load weights you trust: `.pt` files are Python pickles and can execute code when loaded.

## Licensing

Ultralytics YOLO is released under AGPL-3.0; this matters if you distribute or host a modified version. See <https://www.ultralytics.com/license>.
