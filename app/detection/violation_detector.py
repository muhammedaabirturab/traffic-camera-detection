"""Turns the relationship analysis of a single frame into violation *candidates*.

A candidate is only a hypothesis with a confidence score. The rule engine
(app/rules/rule_engine.py) decides whether it is reported, how it is worded, and which
legal reference is attached. Anything the system cannot decide on is returned as an
``Observation`` ("insufficient visual evidence") instead of being silently dropped or
wrongly flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.detection.association import assess_helmets, build_two_wheeler_units
from app.detection.types import Detection, Observation, TwoWheelerUnit, ViolationCandidate

RULE_NO_HELMET = "NO_HELMET"
RULE_TRIPLE_RIDING = "TRIPLE_RIDING"
RULE_RED_LIGHT = "RED_LIGHT_JUMP"


@dataclass
class FrameAnalysis:
    detections: list[Detection]
    units: list[TwoWheelerUnit]
    candidates: list[ViolationCandidate] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


def _helmet_confidence(rider, unit: TwoWheelerUnit, s: Settings) -> float:
    """Confidence that this rider is not wearing a helmet.

    evidence  = confidence of a "bare head" detection if the helmet model has such a class,
                otherwise (1 - confidence of the strongest helmet box near the head)
    weighted by how sure we are that the person exists, is riding this bike, and that the
    bike exists, and by how large the rider is in the image.
    """
    p = rider.person
    if rider.no_helmet is not None:
        evidence = rider.no_helmet.confidence
    else:
        evidence = 1.0 - rider.max_nearby_helmet_conf
    size_factor = min(1.0, p.box.h / (2.5 * s.min_person_height_px))
    conf = (evidence
            * (p.confidence ** 0.5)
            * (rider.association ** 0.5)
            * (unit.vehicle.confidence ** 0.25)
            * (0.7 + 0.3 * size_factor))
    return round(max(0.0, min(1.0, conf)), 4)


def analyse_frame(detections: list[Detection], s: Settings, helmet_model_available: bool,
                  image_height: int) -> FrameAnalysis:
    units = build_two_wheeler_units(detections, s)
    assess_helmets(units, detections, s, helmet_model_available, image_height)
    fa = FrameAnalysis(detections=detections, units=units)

    for unit in units:
        # ------------------------------------------------------------ helmets
        for idx, rider in enumerate(unit.riders, start=1):
            if rider.status == "no_helmet":
                fa.candidates.append(ViolationCandidate(
                    rule_id=RULE_NO_HELMET,
                    confidence=_helmet_confidence(rider, unit, s),
                    vehicle=unit.vehicle,
                    subjects=[rider.person] + ([rider.no_helmet] if rider.no_helmet else []),
                    evidence=rider.reason,
                    region=rider.person.box.union(unit.vehicle.box),
                    details={
                        "rider_index": idx,
                        "riders_on_vehicle": len(unit.riders),
                        "association_score": rider.association,
                        "person_confidence": round(rider.person.confidence, 4),
                        "nearby_helmet_confidence": round(rider.max_nearby_helmet_conf, 4),
                        "bare_head_detected": rider.no_helmet is not None,
                    },
                ))
            elif rider.status in {"insufficient_evidence", "not_evaluated"}:
                fa.observations.append(Observation(RULE_NO_HELMET, f"Helmet check not possible: {rider.reason}",
                                                   rider.person.box, rider.association))

        # ------------------------------------------------------ triple riding
        n = len(unit.riders)
        if n > s.max_riders_allowed:
            strong = [r for r in unit.riders if r.association >= s.rider_association_strong]
            if len(strong) > s.max_riders_allowed:
                top = sorted(strong, key=lambda r: r.association * r.person.confidence, reverse=True)
                top = top[: s.max_riders_allowed + 1]
                mean = sum((r.association * r.person.confidence) ** 0.5 for r in top) / len(top)
                conf = round(min(1.0, mean * unit.vehicle.confidence ** 0.3), 4)
                fa.candidates.append(ViolationCandidate(
                    rule_id=RULE_TRIPLE_RIDING,
                    confidence=conf,
                    vehicle=unit.vehicle,
                    subjects=[r.person for r in unit.riders],
                    evidence=f"{len(strong)} people positioned on one two-wheeler (limit: {s.max_riders_allowed})",
                    region=unit.box,
                    details={"rider_count": n, "strongly_associated": len(strong),
                             "association_scores": [r.association for r in unit.riders]},
                ))
            else:
                fa.observations.append(Observation(
                    RULE_TRIPLE_RIDING,
                    f"{n} people near one two-wheeler but only {len(strong)} clearly seated on it — "
                    "insufficient visual evidence for a multiple-riding violation",
                    unit.box, max((r.association for r in unit.riders), default=0.0)))
    return fa


def rider_summary(units: list[TwoWheelerUnit]) -> dict:
    riders = [r for u in units for r in u.riders]
    return {
        "two_wheelers_with_riders": sum(1 for u in units if u.riders),
        "riders": len(riders),
        "with_helmet": sum(1 for r in riders if r.status == "helmet"),
        "without_helmet": sum(1 for r in riders if r.status == "no_helmet"),
        "undetermined": sum(1 for r in riders if r.status in {"insufficient_evidence", "not_evaluated", "unknown"}),
    }
