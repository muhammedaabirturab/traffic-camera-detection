"""Run the full TrafficGuard pipeline from the command line (no server needed).

python scripts/predict.py path/to/image.jpg
python scripts/predict.py path/to/video.mp4 --stop-line 0.62 --direction down
python scripts/predict.py image.jpg --json result.json

Prints a readable summary; annotated outputs are written to data/results/<id>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _metrics import ROOT  # noqa: F401  (adds project root to sys.path)

from app.analysis.image_analyzer import ImageAnalyzer
from app.analysis.video_analyzer import VideoAnalyzer
from app.config import get_settings
from app.detection.yolo_detector import ModelManager
from app.rules.rule_engine import RuleEngine
from app.utils.helpers import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT, extension, new_id, utc_now_iso
from app.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--stop-line", type=float, default=None, help="video only: stop line as fraction of height")
    ap.add_argument("--direction", choices=["any", "down", "up"], default="any")
    ap.add_argument("--json", type=Path, help="save the full JSON result here")
    args = ap.parse_args()
    setup_logging("WARNING")

    s = get_settings()
    models, rules = ModelManager(s), RuleEngine(s)
    models.load()
    aid, now = new_id(), utc_now_iso()
    ext = extension(args.source.name)
    if ext in ALLOWED_IMAGE_EXT:
        result = ImageAnalyzer(s, models, rules).analyze_bytes(args.source.read_bytes(), args.source.name, aid, now)
    elif ext in ALLOWED_VIDEO_EXT:
        def progress(f, m):
            print(f"\r[{f:5.0%}] {m:<60}", end="", flush=True)
        result = VideoAnalyzer(s, models, rules).analyze(args.source, args.source.name, aid, now,
                                                         args.stop_line, args.direction, progress)
        print()
    else:
        ap.error(f"Unsupported file type {ext}")
        return 2

    sm = result["summary"]
    print(f"\nTrafficGuard AI — {args.source.name}  ({result['processing_time']:.2f}s)")
    print(f"Detectors: {', '.join(k for k, v in result['models'].items() if v)}")
    print(f"Vehicles: {sm['vehicles']}  |  classes: {sm.get('by_class') or sm.get('unique_by_class')}")
    print(f"Possible violations: {len(result['violations'])}")
    for v in result["violations"]:
        when = f" @ {v['timestamp']}s" if "timestamp" in v else ""
        print(f"  ⚠ {v['title']} — {v['confidence']:.0%} ({v['confidence_band']}){when}")
        print(f"     {v['explanation']}")
    for o in result["observations"]:
        print(f"  · {o['message']}")
    print(f"\nOutputs: {s.results_dir / aid}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
