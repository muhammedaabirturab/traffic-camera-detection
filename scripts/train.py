"""Fine-tune a YOLO model on a prepared dataset and publish weights + real metrics.

    python scripts/train.py --data datasets/prepared/data.yaml --epochs 60 --imgsz 640 --batch 8

After training this script
  1. copies the best weights to app/models/<name>.pt,
  2. evaluates them on the held-out *test* split,
  3. writes app/models/<name>.metrics.json (read by the Model page - nothing is hard-coded),
  4. copies the training curves to docs/figures/<name>/.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "datasets" / "prepared" / "data.yaml")
    ap.add_argument("--base", default=str(ROOT / "app" / "models" / "yolov8n.pt"), help="pretrained checkpoint to fine-tune (auto-downloaded by Ultralytics if missing)")
    ap.add_argument("--name", default="rider_detector", help="output name -> app/models/<name>.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="mixed precision (off by default: fp16 gives NaN losses on GTX16xx/MX-class GPUs)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run scripts/prepare_dataset.py first (see datasets/README.md).")

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hardware = torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else f"CPU ({platform.processor() or platform.machine()})"

    runs_dir = ROOT / "runs"
    model = YOLO(args.base)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        workers=args.workers, device=device, seed=args.seed, project=str(runs_dir), name=args.name,
        exist_ok=True, plots=True, amp=args.amp, cos_lr=True,
    )
    run = runs_dir / args.name
    best = run / "weights" / "best.pt"

    models_dir = ROOT / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_weights = models_dir / f"{args.name}.pt"
    shutil.copy2(best, out_weights)

    # Honest evaluation on the held-out test split (falls back to val if the yaml has no test key).
    final = YOLO(str(out_weights))
    split = "test" if "test:" in args.data.read_text() else "val"
    m = final.val(data=str(args.data), split=split, imgsz=args.imgsz, device=device, plots=True,
                  project=str(runs_dir), name=f"{args.name}_{split}", exist_ok=True)
    p, r = float(m.box.mp), float(m.box.mr)
    metrics = {
        "model": Path(args.base).stem,
        "weights": out_weights.name,
        "classes": [final.names[i] for i in sorted(final.names)],
        "evaluated_on": split,
        "precision": p,
        "recall": r,
        "f1": (2 * p * r / (p + r)) if (p + r) else 0.0,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "per_class": {final.names[int(c)]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
                      for i, c in enumerate(m.box.ap_class_index)},
        "training": {
            "epochs_requested": args.epochs, "image_size": args.imgsz, "batch_size": args.batch,
            "base_checkpoint": Path(args.base).name, "hardware": hardware, "device": device,
            "torch": torch.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    results_csv = run / "results.csv"
    if results_csv.exists():
        metrics["training"]["epochs_completed"] = max(0, sum(1 for _ in results_csv.open()) - 1)
    stats = args.data.parent / "dataset_stats.json"
    if stats.exists():
        metrics["dataset"] = json.loads(stats.read_text())
    (models_dir / f"{args.name}.metrics.json").write_text(json.dumps(metrics, indent=2))

    figs = ROOT / "docs" / "figures" / args.name
    figs.mkdir(parents=True, exist_ok=True)
    for src in [run / "results.png", run / "confusion_matrix.png", run / "labels.jpg",
                runs_dir / f"{args.name}_{split}" / "BoxPR_curve.png",
                runs_dir / f"{args.name}_{split}" / "BoxF1_curve.png",
                runs_dir / f"{args.name}_{split}" / "val_batch0_pred.jpg"]:
        if src.exists():
            shutil.copy2(src, figs / src.name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
