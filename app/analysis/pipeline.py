"""Single-frame analysis pipeline shared by image and video analysers.

Detection -> de-duplication -> relationship analysis -> helmet assessment -> rule inference
-> rule-database enrichment -> overlay/summary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.summary import build_overlay
from app.config import Settings
from app.detection import association
from app.detection.types import Detection, Finding, TwoWheelerUnit
from app.detection.violation_detector import evaluate_units
from app.detection.yolo_detector import YoloDetector
from app.rules.rule_engine import RuleEngine


@dataclass
class FrameResult:
    detections: list[Detection]
    units: list[TwoWheelerUnit]
    findings: list[Finding]
    violations: list[dict]
    overlay: list[dict]
    width: int
    height: int


def analyze_frame(frame: np.ndarray, detector: YoloDetector, rules: RuleEngine, s: Settings,
                  track: bool = False) -> FrameResult:
    h, w = frame.shape[:2]
    dets = association.dedupe(detector.detect(frame, track=track))
    units = association.build_units(dets, s)
    association.assess_helmets(units, dets, detector.helmet_available, s)
    findings = evaluate_units(units, detector.helmet_available, s)
    violations = [v for v in (rules.finalize(f) for f in findings) if v is not None]
    overlay = build_overlay(dets, units, violations, w, h, s, detector.helmet_available)
    return FrameResult(dets, units, findings, violations, overlay, w, h)
