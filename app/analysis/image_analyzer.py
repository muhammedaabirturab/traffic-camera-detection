"""Single-image analysis pipeline.

INPUT -> preprocessing -> YOLO detection -> object identification -> relationship
analysis -> traffic rule engine -> violation classification & confidence evaluation
-> evidence generation -> JSON result for the dashboard.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import numpy as np

from app.analysis import plate_reader
from app.analysis.evidence_generator import EvidenceWriter
from app.config import Settings
from app.detection.preprocessing import decode_image, preprocess
from app.detection.types import VEHICLE_LABELS, Box, Detection, Observation
from app.detection.violation_detector import analyse_frame, rider_summary
from app.detection.yolo_detector import ModelManager
from app.rules.rule_engine import RuleEngine
from app.utils import visualization as viz

DISCLAIMER = ("AI-detected possible violations based on available visual evidence. "
              "Results require human verification and are not legal proof.")


class PipelineLog:
    def __init__(self):
        self.stages: list[dict] = []
        self._t = time.perf_counter()

    def mark(self, stage: str, detail: str = "") -> None:
        now = time.perf_counter()
        self.stages.append({"stage": stage, "detail": detail, "ms": round((now - self._t) * 1000, 1)})
        self._t = now


def count_objects(dets: list[Detection]) -> dict:
    by_class = Counter(d.label for d in dets)
    return {
        "vehicles": sum(v for k, v in by_class.items() if k in VEHICLE_LABELS),
        "by_class": dict(sorted(by_class.items())),
    }


def model_flags(models: ModelManager) -> dict:
    return {
        "vehicle_model": models.vehicle is not None,
        "helmet_model": models.helmet_available,
        "rider_model": models.rider is not None,
        "plate_model": models.plate is not None,
        "plate_ocr": models.plate is not None and plate_reader.ocr_available(),
    }


def capability_observations(models: ModelManager, units) -> list[Observation]:
    obs = []
    if not models.helmet_available and any(u.riders for u in units):
        obs.append(Observation("NO_HELMET", "Helmet checks were not performed because no helmet detection model "
                                            "is loaded (see models/README.md). Riders were detected but cannot be "
                                            "classified as helmeted or not."))
    return obs


def attach_plate(violation: dict, image: np.ndarray, dets: list[Detection], models: ModelManager, s: Settings):
    if models.plate is None or not s.enable_plate_ocr:
        return
    plate = plate_reader.find_plate(Box(*violation["vehicle"]["bbox"]), dets)
    if plate is not None:
        violation["plate_reading"] = plate_reader.read_plate(image, plate) or {
            "text": None, "bbox": plate.box.as_list(),
            "note": "Number plate detected but OCR is unavailable or unreadable."}


class ImageAnalyzer:
    def __init__(self, settings: Settings, models: ModelManager, rules: RuleEngine):
        self.s = settings
        self.models = models
        self.rules = rules

    def analyze_bytes(self, data: bytes, filename: str, analysis_id: str, created_at: str) -> dict:
        return self.analyze(decode_image(data), filename, analysis_id, created_at)

    def analyze(self, raw: np.ndarray, filename: str, analysis_id: str, created_at: str,
                writer: Optional[EvidenceWriter] = None) -> dict:
        t0 = time.perf_counter()
        log = PipelineLog()
        writer = writer or EvidenceWriter(self.s.results_dir, analysis_id)

        img, meta = preprocess(raw)
        log.mark("Preprocessing", f"{meta['width']}x{meta['height']}" +
                 (", low-light enhancement applied" if meta["low_light_enhanced"] else ""))

        dets = self.models.detect_all(img)
        for i, d in enumerate(dets):
            d.det_id = i
        log.mark("YOLO detection", f"{len(dets)} objects after confidence filtering and de-duplication")

        counts = count_objects(dets)
        log.mark("Object identification", ", ".join(f"{v} {k}" for k, v in counts["by_class"].items()) or "none")

        fa = analyse_frame(dets, self.s, self.models.helmet_available, img.shape[0])
        riders = rider_summary(fa.units)
        log.mark("Relationship analysis", f"{riders['riders']} rider(s) on {riders['two_wheelers_with_riders']} two-wheeler(s)")

        violations, low_conf = self.rules.apply(fa.candidates, "image")
        log.mark("Traffic rule engine", f"{len(fa.candidates)} candidate(s) -> {len(violations)} reportable")

        observations = capability_observations(self.models, fa.units) + fa.observations + low_conf
        log.mark("Confidence evaluation", f"{len(observations)} item(s) marked insufficient evidence")

        media = {
            "original": writer.save("original.jpg", img),
            "annotated": writer.save("annotated.jpg", self._annotate(img, dets, violations)),
            "thumbnail": writer.thumbnail(img),
        }
        for i, v in enumerate(violations, start=1):
            v["id"] = i
            v["evidence_images"] = writer.violation_evidence(i, img, dets, Box(*v["region"]), v["title"], v["confidence"])
            attach_plate(v, img, dets, self.models, self.s)
        log.mark("Evidence generation", f"{len(violations)} evidence set(s) saved")

        processing_time = round(time.perf_counter() - t0, 3)
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "kind": "image",
            "filename": filename,
            "created_at": created_at,
            "image": meta,
            "media": media,
            "detections": [d.to_dict() for d in dets],
            "two_wheelers": [self._unit_dict(u) for u in fa.units],
            "summary": {
                **counts,
                "persons": counts["by_class"].get("person", 0),
                **riders,
                "possible_violations": len(violations),
                "insufficient_evidence": len(observations),
            },
            "violations": violations,
            "observations": [o.to_dict() for o in observations],
            "confidence": max((v["confidence"] for v in violations), default=None),
            "models": model_flags(self.models),
            "pipeline": log.stages,
            "processing_time": processing_time,
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _unit_dict(u) -> dict:
        return {
            "vehicle_id": u.vehicle.det_id,
            "vehicle_confidence": round(u.vehicle.confidence, 4),
            "bbox": u.vehicle.box.as_list(),
            "supported_by_rider_model": u.supported_by_rider_model,
            "riders": [{
                "person_id": r.person.det_id,
                "association": r.association,
                "helmet_status": r.status,
                "reason": r.reason,
                "helmet_confidence": round(r.helmet.confidence, 4) if r.helmet else None,
            } for r in u.riders],
        }

    @staticmethod
    def _annotate(img: np.ndarray, dets: list[Detection], violations: list[dict]) -> np.ndarray:
        out = viz.draw_detections(img, dets)
        for v in violations:
            out = viz.draw_violation(out, Box(*v["region"]), v["title"], v["confidence"])
        return out
