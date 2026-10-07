"""Evaluate an existing Ultralytics YOLO model and (optionally) publish its metrics.

python scripts/validate.py --weights models/rider.pt --data datasets/rider/data.yaml --role rider

Writes models/metrics/<role>.json (precision, recall, F1, mAP@50, mAP@50-95, per class)
which the dashboard's Model page displays. Use --no-save to only print the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _metrics import ROOT, has_split, metrics_from_val, write_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--role", choices=["helmet", "rider", "plate"], help="required unless --no-save")
    ap.add_argument("--split", choices=["val", "test"], default=None, help="default: test if present, else val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    if not args.no_save and not args.role:
        ap.error("--role is required unless --no-save is given")

    from ultralytics import YOLO

    data = args.data.resolve()
    split = args.split or ("test" if has_split(data, "test") else "val")
    model = YOLO(str(args.weights))
    out_dir = ROOT / "runs" / "val"
    val = model.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                    plots=True, project=str(out_dir), name=f"{args.role or 'model'}_{split}", exist_ok=True)
    metrics = metrics_from_val(val, model.names)
    print(f"\n=== Metrics on '{split}' split ===")
    for k in ("precision", "recall", "f1", "map50", "map50_95"):
        print(f"{k:>10}: {metrics[k]:.4f}")
    for pc in metrics["per_class"]:
        print(f"  {pc['class']:<16} P={pc['precision']:.3f} R={pc['recall']:.3f} mAP50={pc['map50']:.3f}")
    if not args.no_save:
        path = write_metrics(args.role, args.weights, data, split, metrics, model.names, None, Path(val.save_dir))
        print(f"\nMetrics written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
