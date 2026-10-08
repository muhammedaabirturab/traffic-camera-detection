"""Evaluate a trained detector on a dataset split and print Precision / Recall / F1 / mAP.

    python scripts/validate.py --weights app/models/rider_detector.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "app" / "models" / "rider_detector.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "datasets" / "prepared" / "data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    for f in (args.weights, args.data):
        if not f.exists():
            raise SystemExit(f"Missing {f}. Train a model first (python scripts/train.py) - see docs/model.md.")
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    m = YOLO(str(args.weights)).val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=device)
    p, r = float(m.box.mp), float(m.box.mr)
    print(json.dumps({
        "split": args.split, "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
    }, indent=2))


if __name__ == "__main__":
    main()
