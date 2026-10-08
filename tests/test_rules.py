from app.config import Settings
from app.detection import association
from app.detection.types import Detection, Finding, TwoWheelerUnit
from app.detection.violation_detector import evaluate_units
from app.rules.rule_engine import RuleEngine
from app.utils.helpers import confidence_level
from tests.conftest import triple_riding_scene

S = Settings()


def units_for(dets, helmet_loaded=False):
    d = association.dedupe(dets)
    units = association.build_units(d, S)
    association.assess_helmets(units, d, helmet_loaded, S)
    return units, d


def test_confidence_bands_are_configurable():
    assert confidence_level(0.90, S) == "high"
    assert confidence_level(0.85, S) == "high"
    assert confidence_level(0.70, S) == "medium"
    assert confidence_level(0.64, S) == "low"
    custom = Settings(high_confidence=0.95, medium_confidence=0.5)
    assert confidence_level(0.90, custom) == "medium"
    assert confidence_level(0.49, custom) == "low"


def test_rule_database_loads_and_has_required_fields():
    eng = RuleEngine(S)
    assert {"triple_riding", "no_helmet_rider", "red_light_jump"} <= set(eng.rules)
    for r in eng.rules.values():
        for key in ("violation_id", "violation_name", "description", "applicable_vehicle_type",
                    "detection_logic", "legal_reference", "penalty_note"):
            assert key in r, f"{r['violation_id']} missing {key}"
        assert "Rs" not in r["penalty_note"] and "INR" not in r["penalty_note"]  # no hard-coded fines


def test_rule_status_depends_on_installed_models():
    eng = RuleEngine(S)
    no_model = {r["violation_id"]: r["status"]["code"] for r in eng.catalogue({"helmet_model": False})["rules"]}
    with_model = {r["violation_id"]: r["status"]["code"] for r in eng.catalogue({"helmet_model": True})["rules"]}
    assert no_model["no_helmet_rider"] == "needs_model" and with_model["no_helmet_rider"] == "active"
    assert no_model["lane_violation"] == "not_implemented"


def test_three_riders_on_one_motorcycle_is_possible_violation():
    units, _ = units_for(triple_riding_scene())
    findings = evaluate_units(units, False, S)
    assert [f.violation_id for f in findings] == ["triple_riding"]
    v = RuleEngine(S).finalize(findings[0])
    assert v["status"] == "possible" and v["requires_human_verification"] is True
    assert v["rider_count"] == 3 and 0.5 < v["confidence"] <= 1


def test_two_riders_is_not_a_violation():
    units, _ = units_for(triple_riding_scene()[:3])
    assert evaluate_units(units, False, S) == []


def test_standing_bystanders_are_not_counted_as_riders():
    dets = triple_riding_scene()[:3] + [
        Detection("person", 0.9, (520, 120, 580, 400), "coco"),  # pedestrian well to the side
    ]
    units, _ = units_for(dets)
    assert len(units[0].riders) == 2
    assert evaluate_units(units, False, S) == []


def test_ambiguous_third_person_never_becomes_a_violation():
    ambiguous = Detection("person", 0.8, (370, 250, 430, 445), "coco")  # beside the bike: ambiguous (weak association)
    units, _ = units_for(triple_riding_scene()[:3] + [ambiguous])
    assert len(units[0].riders) == 2 and len(units[0].weak_links) == 1
    findings = evaluate_units(units, False, S)
    assert findings, "ambiguity should be reported as insufficient evidence"
    for f in findings:
        assert RuleEngine(S).finalize(f)["status"] == "insufficient_evidence"


def test_low_confidence_finding_downgraded():
    u = TwoWheelerUnit(1, (0, 0, 10, 10), 0.4)
    v = RuleEngine(S).finalize(Finding("triple_riding", 0.3, u, "x"))
    assert v["status"] == "insufficient_evidence" and v["confidence_level"] == "low"


def test_helmet_not_assessed_without_helmet_model():
    units, _ = units_for(triple_riding_scene()[:2])
    assert all(r.helmet_status == "not_assessed" for r in units[0].riders)
    assert all(f.violation_id != "no_helmet_rider" for f in evaluate_units(units, False, S))


def test_helmet_logic_with_helmet_model():
    base = triple_riding_scene()[:2]  # motorcycle + one person
    person = base[1]
    head = (person.box[0], person.box[1], person.box[2], person.box[1] + 45)
    units, _ = units_for(base + [Detection("helmet", 0.9, head, "helmet")], helmet_loaded=True)
    assert units[0].riders[0].helmet_status == "helmet"
    assert evaluate_units(units, True, S) == []
    units, _ = units_for(base + [Detection("no_helmet", 0.9, head, "helmet")], helmet_loaded=True)
    assert [x.violation_id for x in evaluate_units(units, True, S)] == ["no_helmet_rider"]
    units, _ = units_for(base, helmet_loaded=True)  # nothing in head region -> damped confidence
    f = evaluate_units(units, True, S)
    assert f and f[0].confidence < 0.7
