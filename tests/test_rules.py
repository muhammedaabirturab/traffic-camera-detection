import json

from app.detection.types import Box
from app.detection.types import ViolationCandidate
from app.rules.rule_engine import RuleEngine
from conftest import det

REQUIRED = {"violation_id", "violation_name", "description", "applicable_vehicle_type", "detection_logic",
            "legal_reference", "penalty_note"}


def cand(rule="NO_HELMET", conf=0.9):
    return ViolationCandidate(rule_id=rule, confidence=conf, vehicle=det("motorcycle", (0, 0, 10, 10)),
                              subjects=[det("person", (0, 0, 5, 10))], evidence="Helmet not detected",
                              region=Box(0, 0, 10, 10), details={"rider_index": 1, "riders_on_vehicle": 1,
                                                                 "association_score": 0.9, "person_confidence": 0.9})


def test_rule_file_has_required_fields(settings):
    rules = RuleEngine(settings).ruleset.rules
    assert rules
    for r in rules:
        assert REQUIRED <= set(r.model_dump())


def test_no_fabricated_sections_for_unverified_rules(settings):
    for r in RuleEngine(settings).ruleset.rules:
        ref = r.legal_reference
        if ref.section:
            assert ref.source_url and ref.verification, f"{r.violation_id} cites a section without a source"
        else:
            assert "not verified" in (ref.verification or "").lower()


def test_reported_violation_wording(settings):
    v, obs = RuleEngine(settings).evaluate(cand(conf=0.9), "image")
    assert obs is None
    assert v["status"] == "Requires human verification"
    assert v["label"] == "AI-detected possible violation"
    assert v["confidence_band"] == "high"
    assert v["legal_reference"]["section"] == "Section 129"
    assert "requires human verification" in v["explanation"]


def test_medium_confidence_uses_possible_wording(settings):
    v, _ = RuleEngine(settings).evaluate(cand(conf=0.7), "image")
    assert v["confidence_band"] == "medium"
    assert v["title"].startswith("Possible")


def test_low_confidence_becomes_insufficient_evidence(settings):
    v, obs = RuleEngine(settings).evaluate(cand(conf=0.2), "image")
    assert v is None and "insufficient visual evidence" in obs.message


def test_red_light_never_evaluated_on_images(settings):
    v, obs = RuleEngine(settings).evaluate(cand("RED_LIGHT_JUMP", 0.95), "image")
    assert v is None and obs is None


def test_disabled_rule_is_skipped(settings, tmp_path):
    data = json.loads(settings.rules_file.read_text())
    for r in data["rules"]:
        if r["violation_id"] == "NO_HELMET":
            r["enabled"] = False
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(data))
    v, obs = RuleEngine(settings, p).evaluate(cand(), "image")
    assert v is None and obs is None


def test_confidence_bands_are_configurable(settings):
    settings.conf_high = 0.95
    assert settings.confidence_band(0.9) == "medium"
    assert settings.confidence_band(0.5) == "low"
