"""Video analysis: YOLO + ByteTrack, temporal validation, evidence frames and an annotated output video."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from app.analysis.evidence_generator import save_evidence
from app.analysis.pipeline import analyze_frame
from app.config import Settings
from app.detection.preprocessing import InvalidInputError, resize_max_side
from app.detection.tracker import IoUTracker
from app.detection.types import Finding, TwoWheelerUnit
from app.detection.yolo_detector import YoloDetector
from app.rules.rule_engine import DISCLAIMER, RuleEngine
from app.utils.helpers import format_timestamp, utc_now
from app.utils.video_io import VideoWriter
from app.utils.visualization import draw_overlay

VEHICLES = ("car", "bus", "truck", "motorcycle", "bicycle")


@dataclass
class VideoOptions:
    """Operator-supplied context for the experimental red-light rule (the model cannot see signal state)."""
    stop_line: Optional[float] = None  # 0..1 of frame height
    red_from: Optional[float] = None   # seconds
    red_to: Optional[float] = None
    direction: str = "down"            # direction of travel across the line: down | up


@dataclass
class _Track:
    hits: int = 0
    seen: int = 0
    first_idx: int = 0
    last_idx: int = 0
    best_conf: float = 0.0
    best_violation: Optional[dict] = None
    best_crop: Optional[np.ndarray] = None
    best_idx: int = 0
    immediate: bool = False  # events that are temporal by nature (stop-line crossing)


def analyze_video(path: str, analysis_id: str, filename: str, detector: YoloDetector, rules: RuleEngine,
                  s: Settings, opts: VideoOptions, progress: Callable[[float], None] = lambda p: None) -> dict:
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InvalidInputError("The video could not be opened - it may be corrupted or use an unsupported codec.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(s.video_max_seconds * fps)
    truncated = total > max_frames > 0
    n_target = min(total, max_frames) if total > 0 else max_frames

    media_dir = s.data_dir / "media" / analysis_id
    media_dir.mkdir(parents=True, exist_ok=True)
    out_path = media_dir / "processed.mp4"

    detector.reset_tracking()
    unit_tracker = IoUTracker()
    writer: Optional[VideoWriter] = None
    tracks: dict[tuple, _Track] = defaultdict(_Track)
    unit_seen: dict[int, int] = defaultdict(int)
    veh_ids: dict[str, dict[int, int]] = {k: defaultdict(int) for k in VEHICLES}  # label -> track -> appearances
    prev_y: dict[int, float] = {}
    hits_by_sec: dict[int, set] = defaultdict(set)
    conf_sum, conf_n, veh_detections = 0.0, 0, 0
    last_overlay: list[dict] = []
    analyzed = idx = 0
    size = (0, 0)
    hr_h = hr_w = 0

    try:
        while idx < max_frames or total == 0:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, s.max_side)
            h, w = frame.shape[:2]
            frame = frame[: h - h % 2, : w - w % 2]  # even dims for yuv420p
            h, w = frame.shape[:2]
            if writer is None:
                size, hr_h, hr_w = (w, h), h, w
                writer = VideoWriter(out_path, fps, size)
            t = idx / fps

            if idx % s.video_frame_skip == 0:
                analyzed += 1
                res = analyze_frame(frame, detector, rules, s, track=True)
                last_overlay = res.overlay
                _assign_unit_tracks(res.units, unit_tracker, idx)
                # overlay track ids for rider-model-only units
                for o in last_overlay:
                    if o.get("unit_id") is not None and o["label"] in ("motorcycle", "bicycle"):
                        u = next((u for u in res.units if u.unit_id == o["unit_id"]), None)
                        if u is not None:
                            o["track_id"] = u.track_id
                for u in res.units:
                    unit_seen[u.track_id] += 1
                    veh_ids[u.kind][u.track_id] += 1
                for d in res.detections:
                    if d.label in ("car", "bus", "truck"):
                        veh_detections += 1
                        if d.track_id is not None:
                            veh_ids[d.label][d.track_id] += 1
                veh_detections += len(res.units)
                for o in res.overlay:
                    if not o.get("minor") and o["label"] not in ("rider", "pillion"):
                        conf_sum += o["confidence"]
                        conf_n += 1

                annotated: Optional[np.ndarray] = None
                for v in res.violations:
                    key = (v["violation_id"], v["track_id"])
                    tr = tracks[key]
                    if tr.hits == 0:
                        tr.first_idx = idx
                    tr.seen = unit_seen[v["track_id"]]
                    if v["status"] != "possible":
                        continue
                    tr.hits += 1
                    tr.last_idx = idx
                    hits_by_sec[int(t)].add(key)
                    if v["confidence"] > tr.best_conf:
                        if annotated is None:
                            annotated = draw_overlay(frame, res.overlay)
                        tr.best_conf, tr.best_violation, tr.best_idx = v["confidence"], v, idx
                        tr.best_crop = annotated  # keep frame reference; cropped at the end
                        tr.best_crop = _crop(annotated, v["box"])

                if opts.stop_line is not None:
                    _check_red_light(res, opts, t, idx, h, prev_y, tracks, rules, frame, hits_by_sec)
                progress(min(0.99, idx / max(1, n_target)))

            out_frame = draw_overlay(frame, last_overlay)
            _hud(out_frame, idx, t, opts, h)
            writer.write(out_frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if analyzed == 0:
        raise InvalidInputError("No readable frames were found in this video.")

    # ---------------------------------------------------------------- temporal validation
    violations: list[dict] = []
    unconfirmed = 0
    for n, (key, tr) in enumerate(sorted(tracks.items(), key=lambda kv: -kv[1].best_conf), 1):
        if tr.best_violation is None:
            continue
        need = 1 if tr.immediate else s.video_min_violation_frames
        if tr.hits < need or (not tr.immediate and tr.hits < 0.3 * max(1, tr.seen)):
            unconfirmed += 1
            continue
        v = dict(tr.best_violation)
        ev_name = f"evidence_{len(violations) + 1}.jpg"
        crop_path = media_dir / ev_name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), tr.best_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        v.update({
            "evidence_image": f"/media/{analysis_id}/{ev_name}",
            "frame_number": tr.best_idx,
            "timestamp": format_timestamp(tr.best_idx / fps),
            "timestamp_seconds": round(tr.best_idx / fps, 2),
            "first_seen": format_timestamp(tr.first_idx / fps),
            "last_seen": format_timestamp(tr.last_idx / fps),
            "frames_observed": tr.hits,
            "evidence": v["evidence"] + (f" Observed in {tr.hits} analysed frames." if not tr.immediate else ""),
            "_key": list(map(str, key)),
        })
        violations.append(v)
    confirmed_keys = {tuple(v.pop("_key")) for v in violations}

    # per-second timeline (confirmed violations only)
    duration = idx / fps
    timeline = []
    for sec in range(int(duration) + 1):
        active = {k for k in hits_by_sec.get(sec, set()) if tuple(map(str, k)) in confirmed_keys}
        timeline.append({"second": sec, "violations": len(active)})

    uniq = {k: sum(1 for c in ids.values() if c >= 2) or (len(ids) if k in ("motorcycle", "bicycle") else 0)
            for k, ids in veh_ids.items()}
    unique_total = sum(uniq.values())
    avg = conf_sum / conf_n if conf_n else 0.0
    risk_index = min(100.0, 100.0 * sum(v["confidence"] for v in violations) / max(1, unique_total))
    notes = [f"Violations must persist for at least {s.video_min_violation_frames} analysed frames to be reported "
             f"(every {s.video_frame_skip} frame(s) analysed)."]
    if unconfirmed:
        notes.append(f"{unconfirmed} short-lived candidate(s) were discarded by temporal validation.")
    if truncated:
        notes.append(f"Only the first {s.video_max_seconds} seconds were analysed.")
    if not detector.helmet_available:
        notes.append("No helmet model is installed, so helmet compliance was not assessed.")
    if opts.stop_line is not None:
        notes.append("Red-light check is experimental: stop line and red-phase window were supplied by the operator.")

    return {
        "status": "success", "id": analysis_id, "kind": "video", "filename": filename, "created_at": utc_now(),
        "video": {"fps": round(fps, 2), "width": hr_w, "height": hr_h, "frames_total": idx, "frames_analyzed": analyzed,
                  "duration_seconds": round(duration, 2), "frame_skip": s.video_frame_skip, "truncated": truncated},
        "violations": sorted(violations, key=lambda v: v["timestamp_seconds"]),
        "summary": {
            "vehicles_detected": veh_detections, "unique_vehicles": unique_total,
            "motorcycles": uniq["motorcycle"], "bicycles": uniq["bicycle"], "cars": uniq["car"],
            "buses": uniq["bus"], "trucks": uniq["truck"],
            "possible_violations": len(violations), "violation_types": sorted({v["violation_id"] for v in violations}),
            "insufficient_evidence": unconfirmed,
        },
        "timeline": timeline,
        "intelligence": {
            "traffic_objects": unique_total, "normal_objects": max(0, unique_total - len(violations)),
            "possible_violations": len(violations), "average_confidence": round(avg, 4),
            "risk_index": round(risk_index, 1),
            "risk_level": "LOW" if not violations else ("MODERATE" if risk_index < 50 else "HIGH"),
            "note": "Analytical summary of AI detections only - not a legal judgment.",
        },
        "confidence": round(max((v["confidence"] for v in violations), default=avg), 4),
        "processing_time": round(time.perf_counter() - t0, 2),
        "media": {"processed_video": f"/media/{analysis_id}/processed.mp4"},
        "notes": notes, "disclaimer": DISCLAIMER,
    }


def _crop(annotated: np.ndarray, box) -> np.ndarray:
    from app.utils.visualization import crop_with_padding
    return crop_with_padding(annotated, box, pad=0.35)


def _assign_unit_tracks(units: list[TwoWheelerUnit], tracker: IoUTracker, idx: int) -> None:
    missing = [u for u in units if u.track_id is None]
    if missing:
        for u, tid in zip(missing, tracker.update([u.box for u in missing], idx)):
            u.track_id = tid


def _check_red_light(res, opts: VideoOptions, t: float, idx: int, h: int, prev_y: dict, tracks: dict,
                     rules: RuleEngine, frame: np.ndarray, hits_by_sec: dict) -> None:
    line = opts.stop_line * h
    red = (opts.red_from is None or t >= opts.red_from) and (opts.red_to is None or t <= opts.red_to)
    for d in res.detections:
        if d.label not in ("car", "bus", "truck", "motorcycle") or d.track_id is None:
            continue
        y = d.box[3]
        py = prev_y.get(d.track_id)
        prev_y[d.track_id] = y
        if py is None or not red:
            continue
        crossed = (py < line <= y) if opts.direction == "down" else (py > line >= y)
        if not crossed:
            continue
        unit = TwoWheelerUnit(unit_id=d.track_id, bike_box=d.box, bike_conf=d.confidence,
                              kind="motorcycle" if d.label == "motorcycle" else d.label, track_id=d.track_id)
        f = Finding("red_light_jump", d.confidence, unit,
                    f"Tracked {d.label} crossed the operator-defined stop line during the marked red phase.")
        v = rules.finalize(f)
        if v is None or v["status"] != "possible":
            continue
        key = ("red_light_jump", d.track_id)
        tr = tracks[key]
        if tr.hits == 0:
            tr.first_idx = idx
        tr.hits += 1
        tr.last_idx, tr.immediate, tr.seen = idx, True, 1
        hits_by_sec[int(t)].add(key)
        if d.confidence >= tr.best_conf:
            ann = draw_overlay(frame, [{"label": d.label, "confidence": d.confidence, "state": "violation",
                                        "track_id": d.track_id, "box_norm": [d.box[0] / frame.shape[1], d.box[1] / h,
                                                                             d.box[2] / frame.shape[1], d.box[3] / h]}])
            tr.best_conf, tr.best_violation, tr.best_idx = d.confidence, v, idx
            tr.best_crop = _crop(ann, v["box"])


def _hud(frame: np.ndarray, idx: int, t: float, opts: VideoOptions, h: int) -> None:
    if opts.stop_line is not None:
        y = int(opts.stop_line * h)
        red = (opts.red_from is None or t >= opts.red_from) and (opts.red_to is None or t <= opts.red_to)
        cv2.line(frame, (0, y), (frame.shape[1], y), (40, 40, 255) if red else (0, 200, 0), 2, cv2.LINE_AA)
    txt = f"f{idx}  {format_timestamp(t)}"
    cv2.putText(frame, txt, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
