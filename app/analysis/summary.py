"""Overlay construction and the Traffic Intelligence Score (an analytical summary, NOT a legal judgment)."""
from __future__ import annotations

from app.config import Settings
from app.detection.types import Detection, TwoWheelerUnit, VEHICLE_LABELS


def build_overlay(dets: list[Detection], units: list[TwoWheelerUnit], violations: list[dict],
                  img_w: int, img_h: int, s: Settings, helmet_loaded: bool) -> list[dict]:
    """Flat list of drawable items shared by the server-side annotator and the React overlay."""
    def norm(b):
        return [round(b[0] / img_w, 5), round(b[1] / img_h, 5), round(b[2] / img_w, 5), round(b[3] / img_h, 5)]

    by_unit: dict[int, list[dict]] = {}
    for v in violations:
        by_unit.setdefault(v["unit_id"], []).append(v)

    def state_for(conf: float, vs: list[dict] | None = None) -> str:
        if vs:
            if any(v["status"] == "possible" for v in vs):
                return "violation"
            return "insufficient"
        return "low" if conf < s.medium_confidence else "normal"

    items: list[dict] = []
    used_person_ids: set[int] = set()
    for u in units:
        vs = by_unit.get(u.unit_id)
        items.append({
            "label": "motorcycle" if u.kind == "motorcycle" else "bicycle",
            "confidence": round(u.bike_conf, 4), "box_norm": norm(u.box if u.kind == "motorcycle" else u.bike_box),
            "state": state_for(u.bike_conf, vs), "track_id": u.track_id, "unit_id": u.unit_id,
            "detail": f"{len(u.riders)} rider(s)" if u.kind == "motorcycle" else None,
            "riders": len(u.riders) if u.kind == "motorcycle" else None,
        })
        for r in u.riders:
            used_person_ids.add(id(r.person))
            hs = r.helmet_status
            items.append({
                "label": r.role, "confidence": round(r.person.confidence, 4), "box_norm": norm(r.person.box),
                "state": "violation" if hs in ("no_helmet", "no_helmet_inferred") and any(
                    v["status"] == "possible" for v in (vs or [])) else state_for(r.person.confidence),
                "track_id": r.person.track_id, "unit_id": u.unit_id, "helmet_status": hs, "group": "rider",
                "detail": {"helmet": "helmet detected", "no_helmet": "no helmet detected",
                           "no_helmet_inferred": "no helmet found", "unknown": "head not assessable",
                           "not_assessed": "helmet not assessed"}.get(hs),
            })
        for w in u.weak_links:
            used_person_ids.add(id(w.person))
    for d in dets:
        if d.label in ("car", "bus", "truck"):
            items.append({"label": d.label, "confidence": round(d.confidence, 4), "box_norm": norm(d.box),
                          "state": state_for(d.confidence), "track_id": d.track_id, "unit_id": None})
        elif d.label == "person" and id(d) not in used_person_ids:
            items.append({"label": "person", "confidence": round(d.confidence, 4), "box_norm": norm(d.box),
                          "state": state_for(d.confidence), "track_id": d.track_id, "unit_id": None, "minor": True,
                          "group": "pedestrian"})
        elif d.label in ("helmet", "no_helmet") and helmet_loaded:
            items.append({"label": d.label, "confidence": round(d.confidence, 4), "box_norm": norm(d.box),
                          "state": "violation" if d.label == "no_helmet" else "normal", "track_id": None,
                          "unit_id": None, "minor": True})
    return items


def summarize(dets: list[Detection], units: list[TwoWheelerUnit], violations: list[dict], s: Settings) -> dict:
    possible = [v for v in violations if v["status"] == "possible"]
    insufficient = [v for v in violations if v["status"] != "possible"]
    vehicle_units = len(units) + sum(1 for d in dets if d.label in VEHICLE_LABELS - {"motorcycle", "bicycle"})
    counts = {"motorcycle": 0, "bicycle": 0, "car": 0, "bus": 0, "truck": 0}
    for u in units:
        counts[u.kind] += 1
    for d in dets:
        if d.label in ("car", "bus", "truck"):
            counts[d.label] += 1
    riders = sum(len(u.riders) for u in units)
    return {
        "vehicles_detected": vehicle_units,
        "motorcycles": counts["motorcycle"], "bicycles": counts["bicycle"],
        "cars": counts["car"], "buses": counts["bus"], "trucks": counts["truck"],
        "riders": riders,
        "possible_violations": len(possible),
        "insufficient_evidence": len(insufficient),
        "violation_types": sorted({v["violation_id"] for v in possible}),
    }


def intelligence_score(overlay_items: list[dict], violations: list[dict], summary: dict) -> dict:
    """Traffic Intelligence Score: a plain analytical summary of what was seen.

    risk_index = 100 * sum(confidence of possible violations) / max(1, number of vehicles)
    LOW = no possible violations, MODERATE < 50, HIGH >= 50.
    """
    main = [o for o in overlay_items if not o.get("minor") and o["label"] not in ("rider", "pillion")]
    possible = [v for v in violations if v["status"] == "possible"]
    flagged_units = {v["unit_id"] for v in possible}
    flagged_objs = sum(1 for o in main if o.get("unit_id") in flagged_units and o["label"] in ("motorcycle", "bicycle"))
    confs = [o["confidence"] for o in main]
    avg = sum(confs) / len(confs) if confs else 0.0
    n_veh = max(1, summary["vehicles_detected"])
    risk_index = min(100.0, 100.0 * sum(v["confidence"] for v in possible) / n_veh)
    level = "LOW" if not possible else ("MODERATE" if risk_index < 50 else "HIGH")
    return {
        "traffic_objects": len(main),
        "normal_objects": len(main) - flagged_objs,
        "possible_violations": len(possible),
        "average_confidence": round(avg, 4),
        "risk_index": round(risk_index, 1),
        "risk_level": level,
        "note": "Analytical summary of AI detections only - not a legal judgment.",
    }
