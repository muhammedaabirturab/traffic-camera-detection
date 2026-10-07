"""Prepare a YOLO training dataset (train / val / test split + data.yaml).

Supported inputs
----------------
--format darknet  (default) Kaggle "detect-person-on-motorbike-or-scooter" layout:
                  <source>/dataset/obj/*.jpg with a YOLO .txt next to each image.
                  Also works for any folder of image + same-name .txt pairs.
--format yolo     An existing YOLO dataset with images/ and labels/ sub-folders
                  (e.g. a helmet dataset exported from Roboflow in YOLO format).
--format voc      Pascal VOC .xml annotations next to / alongside the images
                  (e.g. helmet datasets with "With Helmet" / "Without Helmet" classes).

Key design choices (worth mentioning in a viva)
-----------------------------------------------
* Leakage-free split: the Kaggle data contains horizontally flipped copies
  (``123__flip.jpg``). Original and flipped copies are grouped and always placed in the
  same split, otherwise the validation score would be inflated by near-duplicates.
* Labels are validated (5 fields, coordinates in [0, 1], positive size, known class);
  invalid lines are dropped and counted in dataset_report.json.
* Class names are never invented: they come from the dataset's names file, the VOC
  XML, or are given explicitly with --names / --class-map.

Examples
--------
python scripts/prepare_dataset.py --source ~/Downloads/detect-person-on-motorbike-or-scooter \
    --out datasets/rider --names rider
python scripts/prepare_dataset.py --format voc --source ~/Downloads/helmet-voc --out datasets/helmet \
    --class-map "With Helmet=helmet,Without Helmet=no_helmet"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def group_key(stem: str) -> str:
    """'123__flip' and '123' share a group so they land in the same split."""
    return re.sub(r"(__flip|_flip|-flip)$", "", stem, flags=re.IGNORECASE)


def read_yolo_label(path: Path, n_classes: int | None, stats: Counter) -> list[list[float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
        except (ValueError, TypeError):
            stats["invalid_lines"] += 1
            continue
        if len(parts) < 5 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            stats["invalid_lines"] += 1
            continue
        if cls < 0 or (n_classes is not None and cls >= n_classes):
            stats["unknown_class_lines"] += 1
            continue
        rows.append([cls, x, y, w, h])
    return rows


def collect_darknet(source: Path) -> list[tuple[Path, Path]]:
    root = source / "dataset" / "obj" if (source / "dataset" / "obj").is_dir() else source
    return [(p, p.with_suffix(".txt")) for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXT]


def collect_yolo(source: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(source.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        parts = list(img.parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        pairs.append((img, Path(*parts).with_suffix(".txt")))
    return pairs


def find_names_file(source: Path) -> Path | None:
    for pattern in ("*.names", "classes.txt", "obj.names"):
        hits = sorted(source.rglob(pattern))
        if hits:
            return hits[0]
    return None


def parse_class_map(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def voc_to_yolo(xml_path: Path, class_map: dict[str, str], names: list[str], stats: Counter) -> tuple[Path | None, list]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = float(size.findtext("width", "0")) if size is not None else 0
    h = float(size.findtext("height", "0")) if size is not None else 0
    fname = root.findtext("filename") or xml_path.with_suffix(".jpg").name
    img = next((p for p in [xml_path.parent / fname, xml_path.parent.parent / "images" / fname,
                             *[xml_path.with_suffix(e) for e in IMG_EXT]] if p.exists()), None)
    if img is None or w <= 0 or h <= 0:
        stats["voc_unresolved"] += 1
        return None, []
    rows = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip()
        name = class_map.get(raw, raw)
        if class_map and raw not in class_map:
            stats["unmapped_class_objects"] += 1
            continue
        if name not in names:
            names.append(name)
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            stats["invalid_lines"] += 1
            continue
        rows.append([names.index(name), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw, bh])
    return img, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="extracted dataset folder")
    ap.add_argument("--out", required=True, type=Path, help="output folder, e.g. datasets/rider")
    ap.add_argument("--format", choices=["darknet", "yolo", "voc"], default="darknet")
    ap.add_argument("--names", help="comma-separated class names overriding the dataset names file")
    ap.add_argument("--class-map", help="VOC/YOLO renaming, e.g. 'With Helmet=helmet,Without Helmet=no_helmet'")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--keep-empty", action="store_true", help="keep images without labels as background images")
    args = ap.parse_args()

    src: Path = args.source.expanduser().resolve()
    if not src.exists():
        print(f"Source folder not found: {src}", file=sys.stderr)
        return 1
    stats: Counter = Counter()
    class_map = parse_class_map(args.class_map)

    # ------------------------------------------------------------------ collect
    items: list[tuple[Path, list]] = []
    names: list[str] = []
    names_source = None
    if args.format == "voc":
        for xml in sorted(src.rglob("*.xml")):
            img, rows = voc_to_yolo(xml, class_map, names, stats)
            if img is not None:
                items.append((img, rows))
        names_source = "VOC <name> tags" + (" (renamed with --class-map)" if class_map else "")
    else:
        nf = find_names_file(src)
        if args.names:
            names = [n.strip() for n in args.names.split(",") if n.strip()]
            names_source = "--names argument"
        elif nf:
            names = [ln.strip() for ln in nf.read_text().splitlines() if ln.strip()]
            names_source = f"names file: {nf.relative_to(src)}"
        pairs = collect_darknet(src) if args.format == "darknet" else collect_yolo(src)
        for img, lbl in pairs:
            items.append((img, read_yolo_label(lbl, len(names) or None, stats)))
        if not names:
            max_cls = max((r[0] for _, rows in items for r in rows), default=0)
            print("No names file found and --names not given; refusing to invent class names.\n"
                  f"Labels use class indices 0..{max_cls}. Re-run with --names, e.g. --names rider", file=sys.stderr)
            return 2
        if class_map:
            names = [class_map.get(n, n) for n in names]

    if not items:
        print("No images found — check --source and --format", file=sys.stderr)
        return 1

    labelled = [(i, r) for i, r in items if r]
    stats["images_found"] = len(items)
    stats["images_without_labels"] = len(items) - len(labelled)
    if not args.keep_empty:
        items = labelled

    # ------------------------------------------------------------------ split
    groups: dict[str, list] = defaultdict(list)
    for img, rows in items:
        groups[group_key(img.stem)].append((img, rows))
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    n_test, n_val = int(round(n * args.test)), int(round(n * args.val))
    split_of = {k: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, k in enumerate(keys)}

    out: Path = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    per_split: Counter = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    for key, members in groups.items():
        split = split_of[key]
        for img, rows in members:
            dst_img = out / "images" / split / img.name
            dst_lbl = out / "labels" / split / (img.stem + ".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if args.link:
                dst_img.symlink_to(img)
            else:
                shutil.copy2(img, dst_img)
            dst_lbl.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows))
            per_split[split] += 1
            for r in rows:
                per_class[split][names[int(r[0])]] += 1

    yaml_text = (f"# Generated by scripts/prepare_dataset.py from {src.name}\n"
                 f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                 + ("test: images/test\n" if per_split["test"] else "")
                 + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "data.yaml").write_text(yaml_text)

    report = {
        "source": str(src), "format": args.format, "classes": names, "class_names_from": names_source,
        "split_seed": args.seed, "split_unit": "image group (original + flipped copy kept together)",
        "groups": n, "images_per_split": dict(per_split),
        "boxes_per_split_and_class": {k: dict(v) for k, v in per_class.items()},
        "checks": dict(stats),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
