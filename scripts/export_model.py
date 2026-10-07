"""Export a trained YOLO model for deployment (ONNX, TorchScript, OpenVINO, ...).

python scripts/export_model.py --weights models/rider.pt --format onnx
python scripts/export_model.py --weights models/helmet.pt --format openvino --half

The exported file is written next to the weights. The API itself uses the .pt files;
exports are for deployment on edge devices / other runtimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _metrics  # noqa: F401  (adds project root to sys.path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--format", default="onnx", help="onnx, torchscript, openvino, engine, coreml, tflite ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (where supported)")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input shapes (ONNX/TensorRT)")
    args = ap.parse_args()

    from ultralytics import YOLO

    path = YOLO(str(args.weights)).export(format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)
    print(f"Exported: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
