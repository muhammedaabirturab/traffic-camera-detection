# Model weights

| File | Purpose | In git? | How you get it |
|---|---|---|---|
| `rider_detector.pt` | YOLOv8n fine-tuned on the `person_bike` dataset (1 class) | yes (~6 MB) | `python scripts/train.py` (or use the committed file) |
| `rider_detector.metrics.json` | Real precision / recall / mAP / training info for the file above | yes | written by `scripts/train.py` |
| `yolov8n.pt` | Official COCO-pretrained YOLOv8 nano: cars, buses, trucks, motorcycles, persons | no | auto-downloaded by Ultralytics on first run, or place it here manually |
| `helmet_detector.pt` | **Optional** helmet / no-helmet detector | no | train your own, see below |

Override any path with `MODEL_PATH`, `COCO_MODEL_PATH`, `HELMET_MODEL_PATH` in `.env`.

## Train / retrain the rider detector

```bash
python scripts/prepare_dataset.py --source <folder with images + YOLO .txt labels>
python scripts/train.py --epochs 50 --batch 8
```

Training copies the best weights here and writes `rider_detector.metrics.json`; the **Model** page reads it.

## Adding helmet detection (optional but needed for helmet rules)

The reference dataset has no helmet labels, so helmet compliance is reported as *not assessed* by default.
To enable it:

1. Obtain a YOLO-format dataset with helmet classes (for example `helmet` and `no_helmet`/`head`) and respect its licence.
2. Prepare and train it under a different name:
   ```bash
   python scripts/prepare_dataset.py --source <helmet dataset> --names helmet no_helmet --out datasets/prepared_helmet
   python scripts/train.py --data datasets/prepared_helmet/data.yaml --name helmet_detector
   ```
3. Make sure the class names match `HELMET_CLASS_NAMES` / `NO_HELMET_CLASS_NAMES` in `.env` (defaults cover `helmet`, `with helmet`, `no_helmet`, `without helmet`, `head`).
4. Restart the backend. The `no_helmet_rider` / `no_helmet_pillion` rules switch from *Needs helmet model* to *Active*.

Only load weights you trust: `.pt` files are Python pickles and can execute code when loaded.

## Licensing

Ultralytics YOLO is released under AGPL-3.0; this matters if you distribute or host a modified version. See <https://www.ultralytics.com/license>.
