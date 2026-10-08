"""Rule engine: turns raw findings into violation records using the configurable rule database."""
from __future__ import annotations

import json
import logging
from typing import Optional

from app.config import Settings
from app.detection.types import Finding
from app.utils.helpers import confidence_level

log = logging.getLogger(__name__)

DISCLAIMER = "AI-detected possible violation based on available visual evidence. Requires human verification."


class RuleEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.version = "unknown"
        self.last_reviewed = None
        self.rules: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        """(Re)read traffic_rules.json so edits take effect without code changes."""
        try:
            data = json.loads(self.s.rules_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read rule database: %s", exc)
            data = {"rules": []}
        self.version = data.get("version", "unknown")
        self.last_reviewed = data.get("last_reviewed")
        self.rules = {r["violation_id"]: r for r in data.get("rules", []) if "violation_id" in r}

    # ------------------------------------------------------------------ catalogue
    def rule_status(self, rule: dict, capabilities: dict) -> dict:
        """Runtime status of a rule given the loaded models: active / needs_model / experimental / ..."""
        if rule.get("not_implemented"):
            return {"code": "not_implemented", "label": "Not implemented"}
        if not rule.get("enabled", True):
            return {"code": "disabled", "label": "Disabled"}
        reqs = rule.get("requires", [])
        if "helmet_model" in reqs and not capabilities.get("helmet_model"):
            return {"code": "needs_model", "label": "Needs helmet model"}
        if "operator_stop_line" in reqs:
            return {"code": "experimental", "label": "Video only - experimental"}
        if rule.get("informational"):
            return {"code": "info", "label": "Informational"}
        return {"code": "active", "label": "Active"}

    def catalogue(self, capabilities: dict) -> dict:
        return {
            "version": self.version,
            "last_reviewed": self.last_reviewed,
            "disclaimer": "Educational project. Legal references must be verified against current official sources.",
            "rules": [{**r, "status": self.rule_status(r, capabilities)} for r in self.rules.values()],
        }

    # ------------------------------------------------------------------ evaluation
    def finalize(self, f: Finding) -> Optional[dict]:
        """Enrich a finding with rule metadata. Returns None if the rule is disabled/unknown."""
        rule = self.rules.get(f.violation_id)
        if rule is None or not rule.get("enabled", True):
            return None
        conf = float(f.confidence)
        status = f.status
        if conf < max(self.s.min_violation_confidence, rule.get("min_confidence", 0.0)):
            status = "insufficient_evidence"
        u = f.unit
        return {
            "violation_id": f.violation_id,
            "name": rule["violation_name"],
            "status": status,
            "status_label": "Possible Violation" if status == "possible" else "Insufficient visual evidence",
            "confidence": round(conf, 4),
            "confidence_level": confidence_level(conf, self.s),
            "vehicle": "Motorcycle" if u.kind == "motorcycle" else u.kind.title(),
            "unit_id": u.unit_id,
            "track_id": u.track_id,
            "box": [round(v, 1) for v in u.box],
            "evidence": f.evidence,
            "legal_reference": rule.get("legal_reference", "N/A"),
            "legal_verified": rule.get("legal_verified", False),
            "penalty_note": rule.get("penalty_note", ""),
            "requires_human_verification": True,
            "disclaimer": DISCLAIMER,
            **f.extra,
        }
