"""Image analysis: bytes in, structured result (and media files) out."""
from __future__ import annotations

import time

import cv2

from app.analysis.evidence_generator import save_evidence
from app.analysis.pipeline import analyze_frame
from app.analysis.summary import intelligence_score, summarize
from app.config import Settings
from app.detection.preprocessing import decode_image, resize_max_side
from app.detection.yolo_detector import YoloDetector
from app.rules.rule_engine import DISCLAIMER, RuleEngine
from app.utils.helpers import new_id, utc_now
from app.utils.visualization import draw_overlay


def analyze_image(data: bytes, filename: str, detector: YoloDetector, rules: RuleEngine, s: Settings) -> dict:
    t0 = time.perf_counter()
    frame = resize_max_side(decode_image(data), s.max_side)  # raises InvalidInputError on bad files
    res = analyze_frame(frame, detector, rules, s)

    aid = new_id()
    media_dir = s.data_dir / "media" / aid
    media_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(media_dir / "original.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated = draw_overlay(frame, res.overlay)
    cv2.imwrite(str(media_dir / "annotated.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    violations = sorted(res.violations, key=lambda v: -v["confidence"])
    for i, v in enumerate(violations, 1):
        save_evidence(annotated, v["box"], media_dir / f"evidence_{i}.jpg")
        v["evidence_image"] = f"/media/{aid}/evidence_{i}.jpg"

    summary = summarize(res.detections, res.units, violations, s)
    intel = intelligence_score(res.overlay, violations, summary)
    possible = [v for v in violations if v["status"] == "possible"]
    confs = [o["confidence"] for o in res.overlay if not o.get("minor")]
    overall = max((v["confidence"] for v in possible), default=(sum(confs) / len(confs) if confs else 0.0))

    notes = []
    if not res.detections:
        notes.append("No traffic objects were detected in this image.")
    if not detector.helmet_available:
        notes.append("No helmet model is installed, so helmet compliance was not assessed.")
    if any(v["status"] != "possible" for v in violations):
        notes.append("Some findings have insufficient visual evidence and are not counted as violations.")

    return {
        "status": "success",
        "id": aid,
        "kind": "image",
        "filename": filename,
        "created_at": utc_now(),
        "image": {"width": frame.shape[1], "height": frame.shape[0]},
        "detections": [d.to_dict(frame.shape[1], frame.shape[0]) for d in res.detections],
        "overlay": res.overlay,
        "violations": violations,
        "summary": summary,
        "intelligence": intel,
        "confidence": round(float(overall), 4),
        "processing_time": round(time.perf_counter() - t0, 3),
        "media": {"original": f"/media/{aid}/original.jpg", "annotated": f"/media/{aid}/annotated.jpg"},
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
