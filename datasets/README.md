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
