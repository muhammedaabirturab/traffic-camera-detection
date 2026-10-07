"""Builds the Model Information page data from the files actually on disk.

Metrics are never hard-coded: they are read from ``models/metrics/<name>.json``,
which ``scripts/train.py`` / ``scripts/validate.py`` write after running Ultralytics
validation on the held-out split. If no such file exists, the page states that the
model is not trained / metrics are unavailable.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from app.config import Settings
from app.detection.yolo_detector import ModelManager

ROLE_DESCRIPTIONS = {
    "vehicle": "General traffic detector (COCO-pretrained): vehicles, persons, traffic lights",
    "helmet": "Helmet / bare-head detector used for the helmet rule",
    "rider": "Person-on-two-wheeler detector (Kaggle reference dataset); supporting evidence for rider association",
    "plate": "Optional number-plate detector for AI-generated plate readings",
}


def load_metrics(metrics_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(metrics_dir).glob("*.json")):
        try:
            data = json.loads(p.read_text())
            data["_file"] = p.name
            out.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def model_info(s: Settings, models: ModelManager) -> dict:
    models.load()
    status = models.status()
    metrics = load_metrics(s.metrics_dir)
    by_weights = {m.get("weights_name"): m for m in metrics}

    detectors = []
    for role, st in status.items():
        entry = {"role": role, "description": ROLE_DESCRIPTIONS[role], **st}
        m = by_weights.get(st.get("weights"))
        if m is None and role != "vehicle":
            m = next((x for x in metrics if x.get("role") == role), None)
        entry["trained_metrics"] = m
        if role == "vehicle":
            entry["provenance"] = ("Official Ultralytics COCO-pretrained weights. Not re-evaluated on project data, "
                                   "so no project-specific metrics are shown for this model.")
        elif not st["available"]:
            entry["provenance"] = "Model not trained / not installed — metrics unavailable."
        elif m is None:
            entry["provenance"] = ("Model file present but no metrics file found in models/metrics — run "
                                   "scripts/validate.py to measure it. Metrics unavailable.")
        else:
            entry["provenance"] = "Metrics measured on the held-out split by scripts/train.py / scripts/validate.py."
        detectors.append(entry)

    try:
        import torch
        cuda = torch.cuda.is_available()
        accel = torch.cuda.get_device_name(0) if cuda else "CPU"
        torch_version = torch.__version__
    except Exception:  # pragma: no cover
        accel, torch_version = "unknown", None
    try:
        import ultralytics
        ul_version = ultralytics.__version__
    except Exception:  # pragma: no cover
        ul_version = None

    return {
        "framework": "Ultralytics YOLO" + (f" {ul_version}" if ul_version else ""),
        "task": "Object detection",
        "inputs": ["Image (JPG/PNG)", "Video (MP4/AVI/MOV)"],
        "inference_imgsz": s.inference_imgsz,
        "device_setting": s.device,
        "runtime": {"python": platform.python_version(), "torch": torch_version, "accelerator": accel},
        "tracker": s.tracker.replace(".yaml", ""),
        "detectors": detectors,
        "training_runs": metrics,
        "trained": any(m for m in metrics),
    }
