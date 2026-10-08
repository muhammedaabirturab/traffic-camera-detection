"""Traffic-rule inference: evaluate two-wheeler units and emit raw findings.

Object detection (YOLO), relationship analysis (association.py) and rule inference (this file) are kept apart
on purpose: YOLO only says *what is where*; this layer decides whether that arrangement looks like a violation.
"""
from __future__ import annotations

from statistics import mean

from app.config import Settings
from app.detection.types import Finding, TwoWheelerUnit


def evaluate_units(units: list[TwoWheelerUnit], helmet_model_loaded: bool, s: Settings) -> list[Finding]:
    findings: list[Finding] = []
    for u in units:
        if u.kind != "motorcycle":
            continue
        findings += _triple_riding(u, s)
        if helmet_model_loaded:
            findings += _helmets(u, s)
    return findings


def _triple_riding(u: TwoWheelerUnit, s: Settings) -> list[Finding]:
    n_strong, n_weak = len(u.riders), len(u.weak_links)
    if n_strong >= 3:
        pc = mean(r.person.confidence for r in u.riders)
        assoc = mean(r.score for r in u.riders)
        conf = min(u.bike_conf, pc) * (0.5 + 0.5 * assoc)
        ev = f"{n_strong} persons positioned on the same two-wheeler (permitted: driver + 1 pillion)."
        return [Finding("triple_riding", conf, u, ev, people=[r.person for r in u.riders],
                        extra={"rider_count": n_strong})]
    if n_strong == 2 and n_weak >= 1:
        # Two clear riders plus an ambiguous third person: do NOT call it a violation.
        pc = mean(r.person.confidence for r in u.riders)
        conf = min(u.bike_conf, pc) * 0.5
        ev = (f"2 persons clearly on the two-wheeler and {n_weak} more nearby whose position is ambiguous "
              "(possibly standing beside it). Insufficient visual evidence of a third rider.")
        return [Finding("triple_riding", conf, u, ev, status="insufficient_evidence",
                        people=[r.person for r in u.riders] + [w.person for w in u.weak_links],
                        extra={"rider_count": 2})]
    return []


def _helmets(u: TwoWheelerUnit, s: Settings) -> list[Finding]:
    out: list[Finding] = []
    for r in u.riders:
        if r.helmet_status not in ("no_helmet", "no_helmet_inferred"):
            continue
        vid = "no_helmet_rider" if r.role == "rider" else "no_helmet_pillion"
        who = "Driver" if r.role == "rider" else "Pillion passenger"
        if r.helmet_status == "no_helmet":
            conf = min(r.helmet_confidence, r.person.confidence) * (0.5 + 0.5 * r.score)
            ev = f"{who}: model detected a bare head (no helmet) in the head region."
        else:
            conf = r.helmet_confidence
            ev = f"{who}: no helmet detected in the head region (absence of a detection is weaker evidence)."
        out.append(Finding(vid, conf, u, ev, people=[r.person], extra={"helmet_status": r.helmet_status}))
    return out
