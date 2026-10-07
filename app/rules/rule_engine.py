"""Traffic rule engine.

Separates *traffic-rule inference* from *object detection*: detectors and the
relationship analysis produce ``ViolationCandidate`` objects; this module looks each one
up in the editable ``traffic_rules.json``, drops disabled or under-confident candidates
(turning them into "insufficient evidence" observations), assigns a confidence band, and
attaches the legal reference and a human-readable explanation.

Nothing here hard-codes a legal section or fine — it all comes from the JSON file.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.config import Settings
from app.detection.types import Observation, ViolationCandidate

VERIFICATION_STATUS = "Requires human verification"


class LegalReference(BaseModel):
    act: Optional[str] = None
    section: Optional[str] = None
    title: Optional[str] = None
    penalty_section: Optional[str] = None
    penalty_title: Optional[str] = None
    notes: Optional[str] = None
    source_url: Optional[str] = None
    verification: Optional[str] = None


class TrafficRule(BaseModel):
    violation_id: str
    violation_name: str
    short_label: str
    description: str
    applicable_vehicle_type: list[str] = Field(default_factory=list)
    detection_logic: str
    required_models: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    status: str
    enabled: bool = True
    min_confidence: Optional[float] = None
    legal_reference: LegalReference = Field(default_factory=LegalReference)
    penalty_note: str = ""


class RuleSet(BaseModel):
    version: str
    jurisdiction: str
    last_reviewed: str
    disclaimer: str
    status_legend: dict[str, str] = Field(default_factory=dict)
    rules: list[TrafficRule]


class RuleEngine:
    def __init__(self, settings: Settings, path: Optional[Path] = None):
        self.s = settings
        self.path = Path(path or settings.rules_file)
        self._lock = threading.Lock()
        self._mtime = -1.0
        self._ruleset: Optional[RuleSet] = None

    # ------------------------------------------------------------------ loading
    @property
    def ruleset(self) -> RuleSet:
        """Rules are re-read automatically when the JSON file changes on disk."""
        with self._lock:
            mtime = self.path.stat().st_mtime
            if self._ruleset is None or mtime != self._mtime:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._ruleset = RuleSet.model_validate(data)
                self._mtime = mtime
            return self._ruleset

    def get(self, rule_id: str) -> Optional[TrafficRule]:
        return next((r for r in self.ruleset.rules if r.violation_id == rule_id), None)

    def is_active(self, rule_id: str, input_type: str) -> bool:
        r = self.get(rule_id)
        return bool(r and r.enabled and r.status != "not_implemented" and input_type in r.input_types)

    # ------------------------------------------------------------------ apply
    def threshold(self, rule: TrafficRule) -> float:
        return max(self.s.report_min_confidence, rule.min_confidence or 0.0)

    def evaluate(self, candidate: ViolationCandidate, input_type: str) -> tuple[Optional[dict], Optional[Observation]]:
        """Return (violation, None) if reportable, else (None, observation-or-None)."""
        rule = self.get(candidate.rule_id)
        if rule is None or not rule.enabled or rule.status == "not_implemented":
            return None, None
        if input_type not in rule.input_types:
            return None, None
        thr = self.threshold(rule)
        if candidate.confidence < thr:
            return None, Observation(
                rule.violation_id,
                f"{rule.short_label}: confidence {candidate.confidence:.0%} is below the reporting "
                f"threshold ({thr:.0%}) — insufficient visual evidence",
                candidate.region, candidate.confidence)
        band = self.s.confidence_band(candidate.confidence)
        title = rule.short_label if band != "high" else rule.violation_name
        return {
            "rule_id": rule.violation_id,
            "title": title,
            "violation_name": rule.violation_name,
            "label": "AI-detected possible violation",
            "confidence": round(candidate.confidence, 4),
            "confidence_band": band,
            "vehicle": {
                "type": candidate.vehicle.label,
                "confidence": round(candidate.vehicle.confidence, 4),
                "bbox": candidate.vehicle.box.as_list(),
                "track_id": candidate.vehicle.track_id,
                "detection_id": candidate.vehicle.det_id,
            },
            "subjects": [d.to_dict() for d in candidate.subjects],
            "evidence": candidate.evidence,
            "explanation": self.explain(rule, candidate),
            "region": candidate.region.as_list(),
            "details": candidate.details,
            "status": VERIFICATION_STATUS,
            "legal_reference": rule.legal_reference.model_dump(),
            "penalty_note": rule.penalty_note,
        }, None

    def apply(self, candidates: list[ViolationCandidate], input_type: str) -> tuple[list[dict], list[Observation]]:
        violations, observations = [], []
        for c in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            v, obs = self.evaluate(c, input_type)
            if v is not None:
                violations.append(v)
            elif obs is not None:
                observations.append(obs)
        return violations, observations

    # ------------------------------------------------------------- explanation
    @staticmethod
    def explain(rule: TrafficRule, c: ViolationCandidate) -> str:
        veh = f"{c.vehicle.label} (detection confidence {c.vehicle.confidence:.0%})"
        ref = rule.legal_reference
        law = f"{ref.section} of the {ref.act}" if ref.section and ref.act else "the configured traffic rule"
        if rule.violation_id == "NO_HELMET":
            d = c.details
            how = ("a bare-head detection was found in the head region"
                   if d.get("bare_head_detected") else "no helmet was detected in the rider's head region")
            text = (f"The system detected a {veh} with {d.get('riders_on_vehicle', 1)} associated rider(s). "
                    f"Rider {d.get('rider_index', 1)} (person confidence {d.get('person_confidence', 0):.0%}, "
                    f"rider-association score {d.get('association_score', 0):.2f}) appears seated on the vehicle, "
                    f"and {how}.")
        elif rule.violation_id == "TRIPLE_RIDING":
            d = c.details
            text = (f"The system detected a {veh} with {d.get('strongly_associated')} people whose position "
                    f"is consistent with sitting on the same vehicle.")
        elif rule.violation_id == "RED_LIGHT_JUMP":
            d = c.details
            text = (f"A tracked {veh} crossed the configured stop line while the detected signal state was "
                    f"{d.get('signal_state')} (signal agreement {d.get('signal_agreement', 0):.0%} over recent frames).")
        else:
            text = f"{c.evidence}."
        return (f"{text} Based on the available visual evidence this may relate to {law}. "
                f"This is an AI-detected possible violation and requires human verification.")

    # ------------------------------------------------------------------ export
    def as_dict(self) -> dict[str, Any]:
        return self.ruleset.model_dump()
