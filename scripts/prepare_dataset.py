"""Prepare a Darknet/YOLO-format dataset for Ultralytics training.

Input : a folder holding `image.jpg` + `image.txt` pairs (YOLO txt labels) and a class-name list.
Output: datasets/prepared/{images,labels}/{train,val,test} + datasets/prepared/data.yaml
        + datasets/prepared/dataset_stats.json (real counts, read by the Model page).

Leakage guard: the reference dataset contains horizontally-flipped copies (`name__flip.jpg`).
A flipped copy of a training image must never land in validation/test, so images are grouped by
their un-augmented stem and whole groups are assigned to a split.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}
AUG_SUFFIX = re.compile(r"__(flip|aug\d*|rot\d*)$", re.IGNORECASE)


def group_key(stem: str) -> str:
    return AUG_SUFFIX.sub("", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="folder with images + YOLO .txt labels")
    ap.add_argument("--names", nargs="+", default=["person_bike"], help="class names in index order")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "prepared")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source folder not found: {args.source}\nSee datasets/README.md for the expected layout.")

    pairs: dict[str, list[tuple[Path, Path]]] = {}
    skipped = 0
    for img in sorted(args.source.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            skipped += 1
            continue
        pairs.setdefault(group_key(img.stem), []).append((img, lbl))

    groups = sorted(pairs)
    if not groups:
        raise SystemExit("No image/label pairs found.")
    random.Random(args.seed).shuffle(groups)
    n_val = max(1, round(len(groups) * args.val))
    n_test = max(1, round(len(groups) * args.test))
    split_of = {}
    for i, g in enumerate(groups):
        split_of[g] = "val" if i < n_val else "test" if i < n_val + n_test else "train"

    if args.out.exists():
        shutil.rmtree(args.out)
    counts: Counter = Counter()
    boxes: Counter = Counter()
    class_hist: Counter = Counter()
    for g, items in pairs.items():
        split = split_of[g]
        for img, lbl in items:
            # Normalise file names (spaces/brackets in names confuse some tools).
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", img.stem)
            for sub in ("images", "labels"):
                (args.out / sub / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, args.out / "images" / split / f"{safe}{img.suffix.lower()}")
            shutil.copy2(lbl, args.out / "labels" / split / f"{safe}.txt")
            counts[split] += 1
            for line in lbl.read_text().splitlines():
                if line.strip():
                    boxes[split] += 1
                    class_hist[int(line.split()[0])] += 1

    yaml_text = (
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(args.names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(args.names))
    )
    (args.out / "data.yaml").write_text(yaml_text)
    stats = {
        "source": "local YOLO-format dataset (see datasets/README.md)",
        "classes": args.names,
        "unique_image_groups": len(groups),
        "images": dict(counts),
        "boxes": dict(boxes),
        "boxes_per_class": {args.names[k] if k < len(args.names) else str(k): v for k, v in class_hist.items()},
        "images_without_labels_skipped": skipped,
        "split_seed": args.seed,
        "note": "Flipped copies are kept in the same split as their original (no leakage).",
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
