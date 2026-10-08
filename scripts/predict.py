"""Run the full TrafficGuard pipeline on one image from the command line (no server needed).

    python scripts/predict.py path/to/image.jpg [--out output_dir]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "predictions")
    args = ap.parse_args()

    from app.analysis.image_analyzer import analyze_image
    from app.config import get_settings
    from app.detection.preprocessing import InvalidInputError
    from app.detection.yolo_detector import ModelUnavailableError, YoloDetector
    from app.rules.rule_engine import RuleEngine

    s = get_settings()
    if not args.image.is_file():
        raise SystemExit(f"File not found: {args.image}")
    try:
        detector = YoloDetector(s)
        result = analyze_image(args.image.read_bytes(), args.image.name, detector, RuleEngine(s), s)
    except (ModelUnavailableError, InvalidInputError) as exc:
        raise SystemExit(str(exc))

    media = s.data_dir / "media" / result["id"]
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(media / "annotated.jpg", args.out / f"{args.image.stem}_annotated.jpg")
    print(json.dumps({k: result[k] for k in ("summary", "violations", "intelligence", "confidence", "processing_time", "notes")}, indent=2))
    print(f"\nAnnotated image: {args.out / (args.image.stem + '_annotated.jpg')}")


if __name__ == "__main__":
    main()
