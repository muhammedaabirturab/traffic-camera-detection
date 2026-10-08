# Datasets

Datasets are **never committed** (`datasets/raw/` and `datasets/prepared/` are git-ignored).

## Reference dataset used by this project

A Darknet/YOLO-format "motorbike rider" dataset (the one behind the Kaggle reference notebook
[motorbike-and-helmet-yolo-detection](https://www.kaggle.com/code/stpeteishii/motorbike-and-helmet-yolo-detection)).
What the copy used here actually contains (inspected, not assumed):

| Property | Value |
|---|---|
| Files | 795 `.jpg` + 795 `.txt` in one `obj/` folder |
| Annotation format | YOLO txt: `class x_center y_center width height` (normalised) |
| Classes | **1** - `person_bike` (a person on a two-wheeler, one box covering rider + bike) |
| Helmet labels | **none** |
| Augmentation | about half of the files are horizontally flipped copies named `<name>__flip.jpg` |
| Original split | none (everything in one folder) |
| Also supplied | a Darknet `yolov3_pb.cfg` + trained `yolov3-obj_final.weights` (not used here) |

## Setup

1. Put the image/label pairs in a folder, e.g. `datasets/raw/obj/` (any location works).
2. Run:
   ```bash
   python scripts/prepare_dataset.py --source datasets/raw/obj
   ```
   This creates `datasets/prepared/` with `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`
   and `dataset_stats.json` (real counts, shown on the Model page).
3. Train: `python scripts/train.py` (see the root README).

**No leakage:** an image and its `__flip` copy are always placed in the same split, so the test metrics are not inflated by near-duplicates.

## Using a different dataset

Any YOLO-format folder works: `--source <folder> --names classA classB ...`.
For helmet detection see `app/models/README.md`.

## Helmet dataset (EdgeVision)

Used for the helmet detector. EdgeVision Dataset, Gajjar, Patel, Patel, Patel, Dabhi, Trivedi, Goyani, Bhatt - Mendeley Data, v1, DOI 10.17632/j82bnw7gsr.1, **CC BY 4.0**.
2,392 images (avg ~1154x1411 px), 8,275 boxes, YOLO labels in `labels/yolo/` with classes `BikeWithRider`, `NoHelmet`, `Helmet`.
Prepared with `--max-side 800` (images downscaled, labels unchanged) into `datasets/prepared_helmet/` (git-ignored). Commands: `app/models/README.md`.
