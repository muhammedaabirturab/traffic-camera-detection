"""Relationship analysis: which people are riding which two-wheeler, and is there a
helmet on each rider's head?

Object detection alone only says "there is a person" and "there is a motorcycle".
A traffic rule is about the *relationship* between them, so this module turns raw
boxes into ``TwoWheelerUnit`` objects using transparent geometric heuristics. Each
heuristic is documented in docs/methodology.md.

Coordinate convention: image pixels, origin top-left, y grows downwards.
"""

from __future__ import annotations

from app.config import Settings
from app.detection.types import Box, Detection, RiderAssessment, TwoWheelerUnit


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def rider_association_score(person: Box, bike: Box) -> float:
    """Score in [0, 1] for "this person is sitting on this two-wheeler".

    A seated rider produces a characteristic layout:
      1. Horizontal alignment – the person's centre lies over the bike's footprint.
      2. Vertical layout – the head is above the top of the bike box, while the hips/legs
         end inside the bike box (people standing *beside* a bike have their feet at or
         below the bike's bottom edge and are usually horizontally offset).
      3. Overlap – a meaningful share of the person box overlaps the bike box.
      4. Scale – person and bike have plausible relative sizes (rejects a distant
         pedestrian that happens to line up behind a nearby bike).
    The four cues are combined multiplicatively-ish so that one clearly failing cue
    vetoes the association.
    """
    if person.area <= 0 or bike.area <= 0:
        return 0.0

    # 1. horizontal: share of the person's width that overlaps the bike's x-range (slightly widened)
    wide = bike.expand(0.10, 0.0)
    x_overlap = max(0.0, min(person.x2, wide.x2) - max(person.x1, wide.x1)) / person.w
    centre_inside = wide.x1 <= person.cx <= wide.x2
    horizontal = x_overlap if centre_inside else x_overlap * 0.3

    # 2. vertical: person bottom should fall within the bike's vertical extent
    #    (from 25% down the bike box to a little below its bottom edge) ...
    lo, hi = bike.y1 + 0.25 * bike.h, bike.y2 + 0.10 * bike.h
    if lo <= person.y2 <= hi:
        bottom = 1.0
    else:
        dist = (lo - person.y2) if person.y2 < lo else (person.y2 - hi)
        bottom = _clamp(1.0 - dist / (0.5 * bike.h))
    #    ... and the person's top (head) should be above the middle of the bike box.
    top = 1.0 if person.y1 <= bike.y1 + 0.35 * bike.h else _clamp(1.0 - (person.y1 - (bike.y1 + 0.35 * bike.h)) / (0.4 * bike.h))
    vertical = bottom * top

    # 3. overlap of the person box with the bike box
    overlap = person.intersection(bike) / person.area
    overlap_score = _clamp(overlap / 0.30)  # 30% or more overlap counts as full evidence

    # 4. scale plausibility (person height relative to bike height)
    ratio = person.h / bike.h
    scale = 1.0 if 0.6 <= ratio <= 3.2 else _clamp(1.0 - abs(ratio - (0.6 if ratio < 0.6 else 3.2)) / 1.0)

    score = (horizontal ** 0.7) * vertical * (0.4 + 0.6 * overlap_score) * scale
    return round(_clamp(score), 4)


def head_region(person: Box, ratio: float) -> Box:
    """Approximate head region: top ``ratio`` of the person box, slightly widened."""
    h = person.h * ratio
    pad_x = person.w * 0.10
    return Box(person.x1 - pad_x, person.y1 - 0.10 * h, person.x2 + pad_x, person.y1 + h)


def helmet_in_head(helmet: Box, head: Box, min_overlap: float) -> float:
    """Return the share of the helmet box inside the head region (0 if below threshold)."""
    if helmet.area <= 0:
        return 0.0
    share = helmet.intersection(head) / helmet.area
    centre_ok = head.x1 <= helmet.cx <= head.x2 and head.y1 <= helmet.cy <= head.y2
    if share >= min_overlap and centre_ok:
        return share
    return 0.0


def build_two_wheeler_units(detections: list[Detection], s: Settings) -> list[TwoWheelerUnit]:
    """Associate people with motorcycles (one person -> at most one motorcycle)."""
    bikes = [d for d in detections if d.label == "motorcycle"]
    people = [d for d in detections if d.label == "person"]
    rider_boxes = [d for d in detections if d.label == "rider"]

    units = [TwoWheelerUnit(vehicle=b) for b in bikes]
    # A dedicated "person on two-wheeler" model (Kaggle reference dataset) supports the bike.
    for u in units:
        u.supported_by_rider_model = any(r.box.intersection(u.vehicle.box) / max(u.vehicle.box.area, 1) > 0.5
                                         for r in rider_boxes)

    for p in people:
        best_unit, best_score = None, 0.0
        for u in units:
            score = rider_association_score(p.box, u.vehicle.box)
            # Supporting evidence from the rider model: person inside a rider box.
            if score > 0 and any(p.box.intersection(r.box) / p.box.area > 0.6 and
                                 r.box.intersection(u.vehicle.box) > 0 for r in rider_boxes):
                score = min(1.0, score + 0.15)
            if score > best_score:
                best_unit, best_score = u, score
        if best_unit is not None and best_score >= s.rider_association_min:
            best_unit.riders.append(RiderAssessment(person=p, association=best_score))

    for u in units:
        u.riders.sort(key=lambda r: r.person.box.cx)
    return units


def assess_helmets(units: list[TwoWheelerUnit], detections: list[Detection], s: Settings,
                   helmet_model_available: bool, image_height: int) -> None:
    """Fill in ``RiderAssessment.status`` for every rider of every unit (in place)."""
    helmets = [d for d in detections if d.label == "helmet"]
    no_helmets = [d for d in detections if d.label == "no_helmet"]

    for u in units:
        for r in u.riders:
            pbox = r.person.box
            head = head_region(pbox, s.head_region_ratio)

            if not helmet_model_available:
                r.status, r.reason = "not_evaluated", "No helmet detection model is loaded"
                continue
            if pbox.h < s.min_person_height_px:
                r.status = "insufficient_evidence"
                r.reason = f"Rider too small to judge ({int(pbox.h)} px tall)"
                continue
            if pbox.y1 <= 2 and pbox.h < image_height * 0.9:
                r.status, r.reason = "insufficient_evidence", "Rider's head is cut off by the frame edge"
                continue

            matched_helmet = max(((h, helmet_in_head(h.box, head, s.helmet_head_overlap_min)) for h in helmets),
                                 key=lambda t: (t[1] > 0, t[0].confidence), default=(None, 0.0))
            matched_no = max(((n, helmet_in_head(n.box, head, s.helmet_head_overlap_min)) for n in no_helmets),
                             key=lambda t: (t[1] > 0, t[0].confidence), default=(None, 0.0))
            # Weak helmet boxes near (not exactly on) the head reduce confidence in an absence call.
            near = head.expand(0.5)
            r.max_nearby_helmet_conf = max((h.confidence for h in helmets if h.box.intersection(near) > 0), default=0.0)

            if matched_helmet[0] is not None and matched_helmet[1] > 0:
                r.helmet = matched_helmet[0]
                if matched_no[0] is not None and matched_no[1] > 0 and matched_no[0].confidence > r.helmet.confidence:
                    r.no_helmet = matched_no[0]
                    r.status, r.reason = "no_helmet", "Bare head detected on rider (stronger than helmet detection)"
                else:
                    r.status, r.reason = "helmet", "Helmet detected in rider's head region"
            elif matched_no[0] is not None and matched_no[1] > 0:
                r.no_helmet = matched_no[0]
                r.status, r.reason = "no_helmet", "Bare head detected in rider's head region"
            else:
                r.status, r.reason = "no_helmet", "Helmet not detected in rider's head region"
