"""Tests for relationship analysis, violation candidates, tracking/temporal validation and detectors.

Geometry used throughout (pixels, y grows downwards):
  motorcycle  (100, 200, 300, 320)
  seated rider (150,  90, 250, 290)   -> head region roughly y 81..150
  pedestrian   (320, 140, 370, 325)   -> standing beside the bike
"""

import numpy as np
import pytest

from app.detection.association import build_two_wheeler_units, head_region, rider_association_score
from app.detection.preprocessing import InvalidImageError, decode_image, enhance_low_light
from app.detection.signal import SignalStateEstimator, classify_light
from app.detection.tracker import TemporalAggregator
from app.detection.types import Box, canonical_label
from app.detection.violation_detector import analyse_frame
from app.detection.yolo_detector import DarknetDetector, deduplicate
from conftest import det

BIKE = (100, 200, 300, 320)
RIDER = (150, 90, 250, 290)
PEDESTRIAN = (320, 140, 370, 325)
HELMET = (175, 85, 225, 125)


# --------------------------------------------------------------------- association
def test_seated_rider_scores_high():
    assert rider_association_score(Box(*RIDER), Box(*BIKE)) >= 0.9


def test_pedestrian_beside_bike_is_not_a_rider():
    assert rider_association_score(Box(*PEDESTRIAN), Box(*BIKE)) < 0.3


def test_distant_small_person_aligned_with_bike_is_rejected():
    # Tiny person far behind the bike: horizontally aligned but wrong scale/vertical layout.
    assert rider_association_score(Box(190, 150, 205, 185), Box(*BIKE)) < 0.5


def test_person_only_assigned_to_one_bike(settings):
    dets = [det("motorcycle", BIKE), det("motorcycle", (400, 200, 600, 320)), det("person", RIDER)]
    units = build_two_wheeler_units(dets, settings)
    assert sum(len(u.riders) for u in units) == 1
    assert len(units[0].riders) == 1


def test_head_region_is_top_of_person():
    head = head_region(Box(*RIDER), 0.3)
    assert head.y2 == pytest.approx(90 + 0.3 * 200)
    assert head.x1 < 150 and head.x2 > 250


# --------------------------------------------------------------------- helmet rule
def test_helmet_on_head_means_no_violation(settings):
    dets = [det("motorcycle", BIKE), det("person", RIDER), det("helmet", HELMET, 0.8, source="helmet")]
    fa = analyse_frame(dets, settings, helmet_model_available=True, image_height=480)
    assert fa.units[0].riders[0].status == "helmet"
    assert not fa.candidates


def test_helmet_elsewhere_does_not_count(settings):
    # A helmet hanging off the handlebar (low on the bike) is not on the rider's head.
    dets = [det("motorcycle", BIKE), det("person", RIDER), det("helmet", (260, 240, 300, 280), 0.9, source="helmet")]
    fa = analyse_frame(dets, settings, True, 480)
    assert [c.rule_id for c in fa.candidates] == ["NO_HELMET"]


def test_missing_helmet_creates_candidate(settings):
    dets = [det("motorcycle", BIKE, 0.9), det("person", RIDER, 0.9)]
    fa = analyse_frame(dets, settings, True, 480)
    assert len(fa.candidates) == 1
    c = fa.candidates[0]
    assert c.rule_id == "NO_HELMET"
    assert 0.5 < c.confidence <= 1.0
    assert c.details["riders_on_vehicle"] == 1


def test_bare_head_class_is_used_as_evidence(settings):
    dets = [det("motorcycle", BIKE), det("person", RIDER), det("no_helmet", HELMET, 0.7, source="helmet")]
    fa = analyse_frame(dets, settings, True, 480)
    assert fa.candidates[0].details["bare_head_detected"] is True


def test_low_confidence_helmet_nearby_reduces_confidence(settings):
    base = analyse_frame([det("motorcycle", BIKE), det("person", RIDER)], settings, True, 480).candidates[0]
    weak = det("helmet", (150, 140, 190, 175), 0.5, source="helmet")  # near the head, not centred in it
    fa = analyse_frame([det("motorcycle", BIKE), det("person", RIDER), weak], settings, True, 480)
    assert fa.candidates and fa.candidates[0].confidence < base.confidence


def test_no_helmet_model_means_not_evaluated(settings):
    fa = analyse_frame([det("motorcycle", BIKE), det("person", RIDER)], settings, False, 480)
    assert not fa.candidates
    assert fa.observations and "No helmet detection model" in fa.observations[0].message


def test_tiny_rider_is_insufficient_evidence(settings):
    dets = [det("motorcycle", (100, 200, 130, 220)), det("person", (105, 185, 125, 218))]
    fa = analyse_frame(dets, settings, True, 480)
    assert not fa.candidates
    assert fa.units[0].riders[0].status == "insufficient_evidence"


def test_pedestrian_never_flagged_for_helmet(settings):
    fa = analyse_frame([det("motorcycle", BIKE), det("person", PEDESTRIAN)], settings, True, 480)
    assert not fa.candidates and not fa.units[0].riders


# --------------------------------------------------------------------- triple riding
TRIO = [(110, 100, 180, 300), (160, 95, 230, 295), (210, 100, 280, 300)]


def test_three_seated_riders_is_triple_riding(settings):
    helmets = [det("helmet", (p[0] + 15, p[1] - 5, p[2] - 15, p[1] + 35), 0.9, source="helmet") for p in TRIO]
    dets = [det("motorcycle", BIKE)] + [det("person", p) for p in TRIO] + helmets
    fa = analyse_frame(dets, settings, True, 480)
    rules = [c.rule_id for c in fa.candidates]
    assert rules == ["TRIPLE_RIDING"]


def test_two_riders_plus_bystander_is_not_triple_riding(settings):
    dets = [det("motorcycle", BIKE), det("person", TRIO[0]), det("person", TRIO[2]), det("person", PEDESTRIAN)]
    fa = analyse_frame(dets, settings, False, 480)
    assert "TRIPLE_RIDING" not in [c.rule_id for c in fa.candidates]


def test_borderline_third_person_gives_insufficient_evidence(settings):
    # Two clearly seated riders plus a third person hanging off the back edge (score ~0.56:
    # counted as associated, but below the "strong" threshold required for a triple-riding call).
    dets = [det("motorcycle", BIKE), det("person", TRIO[0]), det("person", TRIO[1]),
            det("person", (275, 100, 345, 300))]
    fa = analyse_frame(dets, settings, False, 480)
    assert "TRIPLE_RIDING" not in [c.rule_id for c in fa.candidates]
    assert any(o.rule_id == "TRIPLE_RIDING" for o in fa.observations)


# --------------------------------------------------------------------- utilities
def test_deduplicate_merges_motorcycle_and_bicycle_on_same_object():
    out = deduplicate([det("motorcycle", BIKE, 0.8), det("bicycle", (102, 201, 298, 318), 0.6)], 0.7)
    assert [d.label for d in out] == ["motorcycle"]


def test_deduplicate_keeps_distinct_objects():
    out = deduplicate([det("car", (0, 0, 100, 100)), det("car", (200, 0, 300, 100))], 0.7)
    assert len(out) == 2


@pytest.mark.parametrize("raw,expected", [("Motorbike", "motorcycle"), ("With Helmet", "helmet"),
                                          ("Without Helmet", "no_helmet"), ("traffic light", "traffic_light"),
                                          ("giraffe", None)])
def test_canonical_labels(raw, expected):
    assert canonical_label(raw) == expected


def test_decode_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")


def test_low_light_enhancement_only_for_dark_images():
    dark = np.full((64, 64, 3), 20, np.uint8)
    bright = np.full((64, 64, 3), 180, np.uint8)
    assert enhance_low_light(dark)[1] is True
    assert enhance_low_light(bright)[1] is False


# --------------------------------------------------------------------- temporal validation
def _frame(settings, helmet: bool, frame_idx: int, agg: TemporalAggregator):
    dets = [det("motorcycle", BIKE, track_id=1), det("person", RIDER, track_id=7)]
    if helmet:
        dets.append(det("helmet", HELMET, 0.85, source="helmet"))
    fa = analyse_frame(dets, settings, True, 480)
    agg.update(frame_idx, frame_idx / 10, np.zeros((480, 640, 3), np.uint8), fa, "unknown", 0.0)


def test_consistent_no_helmet_is_confirmed(settings):
    agg = TemporalAggregator(settings)
    for i in range(6):
        _frame(settings, helmet=False, frame_idx=i, agg=agg)
    confirmed, obs = agg.finalize()
    assert len(confirmed) == 1 and confirmed[0].frames_observed == 6


def test_single_frame_miss_is_not_reported(settings):
    agg = TemporalAggregator(settings)
    for i in range(8):
        _frame(settings, helmet=(i != 3), frame_idx=i, agg=agg)  # helmet missed once (motion blur)
    confirmed, obs = agg.finalize()
    assert not confirmed
    assert obs and "insufficient temporal evidence" in obs[0].message


def test_unique_vehicle_counting(settings):
    agg = TemporalAggregator(settings)
    for i in range(3):
        _frame(settings, helmet=True, frame_idx=i, agg=agg)
    assert agg.unique_counts()["motorcycle"] == 1


# --------------------------------------------------------------------- red light
def _red_light_image():
    img = np.zeros((480, 640, 3), np.uint8)
    img[20:80, 300:330] = (40, 40, 40)
    img[25:45, 305:325] = (0, 0, 255)  # lit red lamp (BGR)
    return img


def test_traffic_light_colour_classification():
    state, ratio = classify_light(_red_light_image(), det("traffic_light", (300, 20, 330, 80)))
    assert state == "red" and ratio > 0.1


def test_red_light_crossing_detected(settings):
    agg = TemporalAggregator(settings, stop_line_y=300, line_direction="down")
    img = _red_light_image()
    sig = SignalStateEstimator(3)
    for i, bottom in enumerate([200, 240, 280, 320, 360]):
        dets = [det("car", (200, bottom - 80, 300, bottom), track_id=3), det("traffic_light", (300, 20, 330, 80))]
        fa = analyse_frame(dets, settings, False, 480)
        state, agreement = sig.update(img, dets)
        agg.update(i, i / 5, img, fa, state, agreement)
    confirmed, _ = agg.finalize()
    assert [c.candidate.rule_id for c in confirmed] == ["RED_LIGHT_JUMP"]


def test_no_red_light_violation_on_green(settings):
    agg = TemporalAggregator(settings, stop_line_y=300)
    for i, bottom in enumerate([240, 280, 320, 360]):
        fa = analyse_frame([det("car", (200, bottom - 80, 300, bottom), track_id=3)], settings, False, 480)
        agg.update(i, i / 5, np.zeros((10, 10, 3), np.uint8), fa, "green", 1.0)
    assert not agg.finalize()[0]


def test_vehicle_already_past_line_is_ignored(settings):
    agg = TemporalAggregator(settings, stop_line_y=300)
    for i, bottom in enumerate([350, 330, 290, 260]):  # first seen beyond the line, moving up
        fa = analyse_frame([det("car", (200, bottom - 80, 300, bottom), track_id=4)], settings, False, 480)
        agg.update(i, i / 5, np.zeros((10, 10, 3), np.uint8), fa, "red", 1.0)
    # crossing from below to above is allowed for direction "any", but only once and only if
    # first seen on the approach side — here it was, so exactly one event is produced
    assert len(agg.finalize()[0]) == 1
    agg2 = TemporalAggregator(settings, stop_line_y=300, line_direction="down")
    for i, bottom in enumerate([350, 330, 290, 260]):
        fa = analyse_frame([det("car", (200, bottom - 80, 300, bottom), track_id=4)], settings, False, 480)
        agg2.update(i, i / 5, np.zeros((10, 10, 3), np.uint8), fa, "red", 1.0)
    assert not agg2.finalize()[0]


# --------------------------------------------------------------------- Darknet backend
def test_darknet_detector_loads_and_detects(tiny_darknet):
    d = DarknetDetector(str(tiny_darknet / "t.cfg"), str(tiny_darknet / "t.weights"), str(tiny_darknet / "t.names"),
                        "helmet", 0.3, "helmet")
    out = d.detect(np.full((128, 160, 3), 127, np.uint8))
    assert out and all(x.label == "helmet" for x in out)
    assert d.info()["canonical_classes"] == ["helmet"]
