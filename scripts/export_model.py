"""Export trained weights to a deployment format (ONNX by default).

    python scripts/export_model.py --weights app/models/rider_detector.pt --format onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "app" / "models" / "rider_detector.pt")
    ap.add_argument("--format", default="onnx", help="onnx, torchscript, openvino, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"{args.weights} not found. Train first: python scripts/train.py")
    from ultralytics import YOLO
    print("Exported to:", YOLO(str(args.weights)).export(format=args.format, imgsz=args.imgsz))


if __name__ == "__main__":
    main()
