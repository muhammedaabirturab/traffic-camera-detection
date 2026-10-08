"""Model page data. Metrics are READ from files written by scripts/train.py - never hard-coded."""
from __future__ import annotations

import json

from app.config import Settings
from app.detection.yolo_detector import YoloDetector


def read_metrics(s: Settings) -> dict | None:
    f = s.model_path.with_suffix(".metrics.json")
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def model_info(detector: YoloDetector | None, s: Settings) -> dict:
    metrics = read_metrics(s)
    status = detector.status() if detector else None
    return {
        "framework": "Ultralytics YOLO",
        "task": "Object detection (+ rule-based traffic inference)",
        "input": "Image (JPG/PNG) / Video (MP4/AVI/MOV)",
        "dataset": "Motorbike / rider traffic dataset (single class: person_bike). No helmet labels.",
        "models": status,
        "metrics_available": metrics is not None,
        "metrics": metrics,
        "metrics_message": None if metrics else "Model not trained / metrics unavailable. Run scripts/train.py to generate real metrics.",
        "helmet_model": bool(detector and detector.helmet_available),
        "config": {
            "confidence_threshold": s.confidence_threshold, "iou_threshold": s.iou_threshold,
            "image_size": s.image_size, "device": status["device"] if status else s.device,
            "video_frame_skip": s.video_frame_skip, "video_min_violation_frames": s.video_min_violation_frames,
            "confidence_bands": {"high": s.high_confidence, "medium": s.medium_confidence},
            "min_violation_confidence": s.min_violation_confidence,
            "min_association_score": s.min_association_score,
        },
    }
