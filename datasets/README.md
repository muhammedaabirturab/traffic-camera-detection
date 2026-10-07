# datasets/

Datasets are **never committed** (everything here except this file is git-ignored).

## Reference project

The project builds on the Kaggle notebook
[Motorbike and Helmet YOLO Detection](https://www.kaggle.com/code/stpeteishii/motorbike-and-helmet-yolo-detection)
(stpeteishii). That notebook does not train a model: it runs two **pretrained Darknet YOLOv3**
models through OpenCV DNN, taken from two datasets by Savan Agrawal. What they actually contain:

| Dataset | Format | Classes | Contents |
|---|---|---|---|
| [`savanagrawal/detect-person-on-motorbike-or-scooter`](https://www.kaggle.com/datasets/savanagrawal/detect-person-on-motorbike-or-scooter) (~370 MB) | Darknet/YOLO: `dataset/obj/<n>.jpg` + `<n>.txt` (`class cx cy w h`, normalised) | **1** — a person on a motorcycle/scooter (`coco.names`) | Images with one box each, horizontally flipped copies named `<n>__flip.jpg`, `yolov3_pb.cfg`, `yolov3-obj_final.weights`, training chart |
| [`savanagrawal/helmet-detection-yolov3`](https://www.kaggle.com/datasets/savanagrawal/helmet-detection-yolov3) (~235 MB) | Darknet model only | **1** — `Helmet` (`helmet.names`) | `yolov3-helmet.cfg`, `yolov3-helmet.weights`, one sample image — **no training images** |

Consequences for the design:

* There is no "no-helmet", number-plate, seat-belt or lane data, so those classes are **not** invented.
  The helmet rule works by looking for a `Helmet` box in each rider's head region; absence → possible violation.
* Vehicles, persons and traffic lights come from the COCO-pretrained YOLO11 model (real COCO classes).
* The rider dataset is used to train a YOLO11 "rider" detector that supports the rider ↔ motorcycle association.

## Setup

1. Download the dataset(s) from Kaggle (website → *Download*, or `kaggle datasets download -d savanagrawal/detect-person-on-motorbike-or-scooter --unzip`).
2. Extract anywhere, e.g. `~/Downloads/detect-person-on-motorbike-or-scooter/`.
3. Convert to a YOLO train/val/test split:

```bash
python scripts/prepare_dataset.py \
  --source ~/Downloads/detect-person-on-motorbike-or-scooter \
  --out datasets/rider --names rider
```

This creates:

```
datasets/rider/
├── data.yaml                 # Ultralytics dataset config
├── dataset_report.json       # counts per split/class + label-validation results
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

The split is **grouped**: `123.jpg` and `123__flip.jpg` always land in the same split, otherwise
near-duplicate images would leak from training into validation and inflate the metrics.
Invalid label lines (out-of-range coordinates, unknown class index) are dropped and counted.

## Other helmet datasets (optional)

To train a YOLO11 helmet model (better than absence-only reasoning because it can learn a bare-head class),
use any dataset that has images, for example a Pascal-VOC helmet dataset with `With Helmet` / `Without Helmet` labels:

```bash
python scripts/prepare_dataset.py --format voc --source ~/Downloads/helmet-voc --out datasets/helmet \
  --class-map "With Helmet=helmet,Without Helmet=no_helmet"
python scripts/train.py --data datasets/helmet/data.yaml --role helmet
```

Class names must come from the dataset; `prepare_dataset.py` refuses to guess them.
