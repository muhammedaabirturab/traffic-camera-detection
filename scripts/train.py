"""Train a YOLO detector with Ultralytics and register it with the app.

After training, the best checkpoint is evaluated on the held-out *test* split (or
*val* if there is no test split), and:

* the weights are copied to ``models/<role>.pt`` (picked up by the API automatically
  after ``POST /api/model/reload`` or a restart),
* precision, recall, F1, mAP@50 and mAP@50-95 (overall + per class), the training
  configuration and hardware are written to ``models/metrics/<role>.json``,
* training plots (loss curves, PR/F1 curves, confusion matrix) are copied next to it.

Examples
--------
# Rider (person on two-wheeler) model on the Kaggle reference data, prepared with prepare_dataset.py
python scripts/train.py --data datasets/rider/data.yaml --role rider --epochs 50

# Helmet model on a helmet dataset with images (classes helmet / no_helmet)
python scripts/train.py --data datasets/helmet/data.yaml --role helmet --epochs 80 --device 0

# Quick pipeline check (few minutes on CPU, NOT a usable model)
python scripts/train.py --data datasets/rider/data.yaml --role rider --epochs 1 --fraction 0.1 --no-install
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from _metrics import ROOT, has_split, metrics_from_val, write_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="data.yaml produced by prepare_dataset.py")
    ap.add_argument("--role", required=True, choices=["helmet", "rider", "plate"],
                    help="what the model is used for in the app (decides models/<role>.pt)")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (transfer learning), e.g. yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'cpu', '0' (GPU), 'mps'; default: auto")
    ap.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of training images to use")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default=str(ROOT / "runs" / "train"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-install", action="store_true", help="do not copy weights/metrics into models/")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = args.data.resolve()
    start_weights = args.model
    if not Path(start_weights).exists() and (ROOT / "models" / start_weights).exists():
        start_weights = str(ROOT / "models" / start_weights)

    model = YOLO(start_weights)
    name = args.name or f"{args.role}_{time.strftime('%Y%m%d_%H%M%S')}"
    t0 = time.time()
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                patience=args.patience, fraction=args.fraction, workers=args.workers, seed=args.seed,
                project=args.project, name=name, exist_ok=True, plots=True, verbose=True)
    duration = round(time.time() - t0, 1)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    split = "test" if has_split(data, "test") else "val"
    print(f"\nEvaluating {best} on the '{split}' split ...")
    trained = YOLO(str(best))
    val = trained.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                      plots=True, project=str(run_dir), name=f"eval_{split}", exist_ok=True, verbose=False)
    metrics = metrics_from_val(val, trained.names)
    training = {"base_model": Path(args.model).name, "epochs_requested": args.epochs,
                "epochs_completed": int(getattr(model.trainer, "epoch", args.epochs - 1)) + 1,
                "imgsz": args.imgsz, "batch": args.batch, "device": str(args.device or "auto"),
                "patience": args.patience, "fraction": args.fraction, "seed": args.seed,
                "duration_seconds": duration, "run_dir": str(run_dir)}

    print("\n=== Held-out metrics ===")
    for k in ("precision", "recall", "f1", "map50", "map50_95"):
        print(f"{k:>10}: {metrics[k]:.4f}")

    if args.no_install:
        print(f"\nWeights left in {best} (not installed).")
        return 0

    dst = ROOT / "models" / f"{args.role}.pt"
    shutil.copy2(best, dst)
    eval_dir = run_dir / f"eval_{split}"
    for plot in ("results.png", "labels.jpg"):
        if (run_dir / plot).exists():
            shutil.copy2(run_dir / plot, eval_dir / plot)
    out = write_metrics(args.role, dst, data, split, metrics, trained.names, training, eval_dir)
    print(f"\nInstalled weights -> {dst}\nMetrics -> {out}\n"
          "Restart the server or call POST /api/model/reload to use the new model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
