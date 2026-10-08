# Model & training

## Models

| Model | Source | Classes | Role |
|---|---|---|---|
| COCO detector | Ultralytics `yolov8n.pt` (official, pretrained, unchanged) | person, bicycle, car, motorcycle, bus, truck (others ignored) | vehicles + persons |
| Rider detector | `yolov8n.pt` **fine-tuned here** on the `person_bike` dataset | `person_bike` | person-on-two-wheeler corroboration |
| Helmet detector | *not shipped* | `helmet` / `no_helmet` (configurable) | enables the helmet rules |

YOLOv8-nano (about 3 M parameters) was chosen so everything runs on a student laptop (a 2 GB MX-class GPU or plain CPU).

## Dataset

See `datasets/README.md`. In short: 795 images, a single class `person_bike`, no helmet labels, about half of the images
are horizontal flips of the others. `scripts/prepare_dataset.py` groups an image with its flip, then splits groups
**train / val / test = 75 % / 15 % / 10 %** (seed 42) so near-duplicates never straddle splits.
The exact counts used for the committed weights are in `app/models/rider_detector.metrics.json` and on the Model page.

## Training

```bash
python scripts/prepare_dataset.py --source <folder with images + .txt labels>
python scripts/train.py --epochs 50 --batch 8 --imgsz 640
```

* Fine-tunes `app/models/yolov8n.pt` with Ultralytics defaults (AdamW chosen by `optimizer=auto`, mosaic/HSV/flip augmentation, cosine LR, early stopping with patience 20).
* Mixed precision is **off by default**: on the GTX16xx/MX-class GPU used during development FP16 produced NaN losses. Pass `--amp` on modern GPUs.
* After training it evaluates the best weights on the held-out **test** split and writes
  `app/models/rider_detector.metrics.json` (precision, recall, F1, mAP@50, mAP@50-95, per-class AP, epochs, image size,
  batch size, hardware, dataset counts) and copies the curves into `docs/figures/rider_detector/`.

## Metrics

Precision, recall, F1, mAP@50 and mAP@50-95 are produced by Ultralytics' validator (`scripts/validate.py` repeats the
test-split evaluation any time). **They are read from the JSON file by the Model page - never typed in.**
If the file does not exist the page says *"Model not trained / metrics unavailable"*.

How to read them honestly: the test split holds only about 80 images from roughly 40 distinct scenes, and the dataset
is dominated by clear, front-on scooter photos. The numbers describe *this* dataset, not real CCTV conditions
(night, rain, occlusion, far-away cameras), where performance will be lower.

## Export

```bash
python scripts/export_model.py --weights app/models/rider_detector.pt --format onnx
```

## Reference Darknet model

The dataset package also contained a Darknet YOLOv3 config and trained weights (`yolov3-obj_final.weights`, 2000 iterations).
They were not used: they require the Darknet/OpenCV-DNN runtime and the Ultralytics workflow offers tracking, export and evaluation out of the box.
