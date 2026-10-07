"""Shared helpers for train.py / validate.py: turn Ultralytics validation output into the
``models/metrics/<role>.json`` file read by the dashboard's Model page."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS_DIR = ROOT / "models" / "metrics"
PLOTS = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
         "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "PR_curve.png", "F1_curve.png",
         "labels.jpg", "val_batch0_pred.jpg"]


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["accelerator"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        info["accelerator"] = "unknown"
    return info


def f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def metrics_from_val(val, names: dict | list) -> dict:
    """Extract precision / recall / mAP / F1 (overall and per class) from a DetMetrics object."""
    box = val.box
    p, r = float(box.mp), float(box.mr)
    out = {
        "precision": round(p, 4), "recall": round(r, 4), "f1": f1(p, r),
        "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
        "per_class": [],
    }
    names = names if isinstance(names, dict) else dict(enumerate(names))
    for i, cls in enumerate(getattr(box, "ap_class_index", [])):
        pc, rc, ap50, ap = box.class_result(i)
        out["per_class"].append({"class": names.get(int(cls), str(cls)), "precision": round(float(pc), 4),
                                 "recall": round(float(rc), 4), "f1": f1(float(pc), float(rc)),
                                 "map50": round(float(ap50), 4), "map50_95": round(float(ap), 4)})
    return out


def write_metrics(role: str, weights: Path, data_yaml: Path, split: str, metrics: dict, names,
                  training: dict | None = None, run_dir: Path | None = None) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plots = []
    if run_dir is not None and Path(run_dir).exists():
        dst = METRICS_DIR / role
        dst.mkdir(exist_ok=True)
        for name in PLOTS:
            src = Path(run_dir) / name
            if src.exists():
                shutil.copy2(src, dst / name)
                plots.append(f"{role}/{name}")
    report_file = Path(data_yaml).parent / "dataset_report.json"
    record = {
        "role": role,
        "weights_name": Path(weights).name,
        "task": "detect",
        "dataset_yaml": str(data_yaml),
        "dataset_report": json.loads(report_file.read_text()) if report_file.exists() else None,
        "evaluated_split": split,
        "classes": list(names.values()) if isinstance(names, dict) else list(names),
        "metrics": metrics,
        "training": training,
        "hardware": hardware(),
        "plots": plots,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = METRICS_DIR / f"{role}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def has_split(data_yaml: Path, split: str) -> bool:
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text())
    return bool(data.get(split))
