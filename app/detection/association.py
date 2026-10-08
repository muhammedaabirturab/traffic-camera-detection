"""Relationship analysis: which persons ride which two-wheeler, and do they wear helmets?

Everything here is plain geometry on bounding boxes (documented in docs/methodology.md). The goal is to
avoid the naive "person detected + helmet not detected = violation" shortcut: a person only counts as a
rider if they are positioned *on* a two-wheeler.
"""
from __future__ import annotations

from app.config import Settings
from app.detection.types import Box, Detection, RiderLink, TwoWheelerUnit

# ---------------------------------------------------------------------------- box helpers

def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def iomin(a: Box, b: Box) -> float:
    """Intersection over the smaller box - robust when one box contains the other."""
    m = min(area(a), area(b))
    return inter(a, b) / m if m > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# ---------------------------------------------------------------------------- duplicate filtering

def dedupe(dets: list[Detection], iou_thr: float = 0.7) -> list[Detection]:
    """Drop near-identical boxes of the same label (keeps the most confident one)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.confidence):
        if any(k.label == d.label and iou(k.box, d.box) > iou_thr for k in kept):
            continue
        kept.append(d)
    return kept


# ---------------------------------------------------------------------------- units

def build_units(dets: list[Detection], s: Settings) -> list[TwoWheelerUnit]:
    """Fuse COCO motorcycle/bicycle boxes with rider-model boxes into two-wheeler units and attach persons."""
    bikes = [d for d in dets if d.label == "motorcycle"]
    cycles = [d for d in dets if d.label == "bicycle"]
    riders = [d for d in dets if d.label == "rider_unit"]
    persons = [d for d in dets if d.label == "person"]

    units: list[TwoWheelerUnit] = []
    uid = 1
    for b in bikes:
        units.append(TwoWheelerUnit(uid, b.box, b.confidence, "motorcycle", "coco", track_id=b.track_id))
        uid += 1
    for r in riders:
        # Does this person+bike box correspond to a COCO motorcycle we already have?
        match = None
        for u in units:
            if u.kind == "motorcycle" and u.rider_box is None and iomin(u.bike_box, r.box) > 0.6:
                match = u
                break
        if match is not None:
            match.rider_box = r.box
            match.origin = "both"
            match.bike_conf = max(match.bike_conf, r.confidence) if match.bike_conf < 0.5 else match.bike_conf
        else:
            # Rider model only: the lower ~60% of a "person_bike" box is the machine.
            x1, y1, x2, y2 = r.box
            bike_box = (x1, y1 + 0.40 * (y2 - y1), x2, y2)
            units.append(TwoWheelerUnit(uid, bike_box, r.confidence, "motorcycle", "rider_model", rider_box=r.box))
            uid += 1
    for c in cycles:
        # Bicycles are counted but never evaluated against motorcycle rules.
        if not any(u.kind == "motorcycle" and iomin(u.bike_box, c.box) > 0.7 for u in units):
            units.append(TwoWheelerUnit(uid, c.box, c.confidence, "bicycle", "coco", track_id=c.track_id))
            uid += 1

    _attach_persons(units, persons, s)
    return units


def association_score(p: Detection, u: TwoWheelerUnit) -> float:
    """0..1 score that person `p` is sitting on two-wheeler `u`."""
    bx1, by1, bx2, by2 = u.bike_box
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    # rider region: the machine plus the space above it where the body sits
    region = (bx1 - 0.10 * bw, by1 - 1.3 * bh, bx2 + 0.10 * bw, by2 + 0.10 * bh)
    pa = max(1.0, area(p.box))
    overlap = inter(p.box, region) / pa

    pcx, _ = center(p.box)
    half = 0.5 * bw
    dx = 0.0 if bx1 <= pcx <= bx2 else min(abs(pcx - bx1), abs(pcx - bx2))
    horiz = max(0.0, 1.0 - dx / max(1.0, half))

    # seated riders: lower body reaches into the machine box; standing bystanders usually end below/away
    vertical = 1.0 if (by1 + 0.15 * bh) <= p.box[3] <= (by2 + 0.15 * bh) else 0.0
    # sane scale: rider height is comparable to the machine height
    ratio = p.height / bh
    scale = 1.0 if 0.7 <= ratio <= 2.8 else 0.0

    if u.rider_box is not None:
        contain = inter(p.box, u.rider_box) / pa
        score = 0.30 * overlap + 0.20 * horiz + 0.15 * vertical + 0.10 * scale + 0.25 * contain
    else:
        score = 0.40 * overlap + 0.25 * horiz + 0.20 * vertical + 0.15 * scale
    return float(max(0.0, min(1.0, score)))


def _attach_persons(units: list[TwoWheelerUnit], persons: list[Detection], s: Settings) -> None:
    motor = [u for u in units if u.kind == "motorcycle"]
    if not motor:
        return
    for p in persons:
        scored = [(association_score(p, u), u) for u in motor]
        score, best = max(scored, key=lambda t: t[0])
        if score >= s.min_association_score:
            best.riders.append(RiderLink(p, score, True))
        elif score >= s.weak_association_score:
            best.weak_links.append(RiderLink(p, score, False))
    for u in motor:
        # the front-most/largest person is the driver, everyone else is a pillion
        u.riders.sort(key=lambda r: -area(r.person.box))
        _demote_inconsistent(u, s)
        for i, r in enumerate(u.riders):
            r.role = "rider" if i == 0 else "pillion"


def _demote_inconsistent(u: TwoWheelerUnit, s: Settings) -> None:
    """A pillion sits at the same depth as the driver and inside the machine's footprint.

    People far behind the bike (much smaller in the image than the driver) or at its very edge are
    background pedestrians or riders of *other* bikes in a convoy: demote them to ambiguous.
    """
    if len(u.riders) < 2:
        return
    driver = u.riders[0].person
    dh, da = max(1.0, driver.height), max(1.0, area(driver.box))
    bx1, _, bx2, _ = u.bike_box
    margin = 0.10 * (bx2 - bx1)
    keep = [u.riders[0]]
    for r in u.riders[1:]:
        cx = center(r.person.box)[0]
        same_depth = area(r.person.box) >= 0.25 * da and r.person.height >= 0.5 * dh
        inside = (bx1 + margin) <= cx <= (bx2 - margin)
        if same_depth and inside:
            keep.append(r)
        else:
            r.strong = False
            u.weak_links.append(r)
    u.riders = keep


# ---------------------------------------------------------------------------- helmets

def head_region(person: Box) -> Box:
    x1, y1, x2, y2 = person
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.10 * w, y1 - 0.05 * h, x2 + 0.10 * w, y1 + 0.30 * h)


def assess_helmets(units: list[TwoWheelerUnit], dets: list[Detection], helmet_model_loaded: bool, s: Settings) -> None:
    """Fill `helmet_status` on every strongly-associated rider.

    No helmet model -> `not_assessed` (we refuse to guess). With a model: a helmet box in the head region
    -> `helmet`; an explicit no-helmet box -> `no_helmet`; neither -> `no_helmet_inferred` (weaker evidence);
    head cut off by the image border or too small to judge -> `unknown`.
    """
    helmets = [d for d in dets if d.label == "helmet"]
    bares = [d for d in dets if d.label == "no_helmet"]
    for u in units:
        for r in u.riders:
            if not helmet_model_loaded:
                r.helmet_status, r.helmet_confidence = "not_assessed", 0.0
                continue
            hr = head_region(r.person.box)
            head_h = hr[3] - hr[1]
            if head_h < 14:
                r.helmet_status = "unknown"
                continue
            h_hit = max((d for d in helmets if iomin(d.box, hr) > 0.4 and center(d.box)[1] < hr[3]),
                        key=lambda d: d.confidence, default=None)
            n_hit = max((d for d in bares if iomin(d.box, hr) > 0.4), key=lambda d: d.confidence, default=None)
            if h_hit and (not n_hit or h_hit.confidence >= n_hit.confidence):
                r.helmet_status, r.helmet_confidence = "helmet", h_hit.confidence
            elif n_hit:
                r.helmet_status, r.helmet_confidence = "no_helmet", n_hit.confidence
            else:
                # absence of a detection is weak evidence; confidence is deliberately damped
                r.helmet_status = "no_helmet_inferred"
                r.helmet_confidence = round(0.7 * r.person.confidence * r.score, 4)
