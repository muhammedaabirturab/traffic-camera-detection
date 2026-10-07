"""Temporal validation for video.

Per-frame analysis is noisy: a helmet can be missed in one frame because of motion blur
or occlusion. Objects are therefore tracked across frames (ByteTrack / BoT-SORT via
Ultralytics, giving each object a persistent ``track_id``) and a violation is only
reported when it is observed consistently:

* seen in at least ``temporal_min_frames`` analysed frames, and
* in at least ``temporal_min_ratio`` of the frames where it *could* be evaluated.

The same tracks are used for the red-light module: a vehicle whose reference point
crosses the configured stop line while the smoothed signal state is red.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import Settings
from app.detection.types import VEHICLE_LABELS, Box, Detection, Observation, ViolationCandidate
from app.detection.violation_detector import RULE_RED_LIGHT, FrameAnalysis


@dataclass
class _Evidence:
    frame_index: int
    timestamp: float
    confidence: float
    candidate: ViolationCandidate
    frame: np.ndarray
    detections: list[Detection]


@dataclass
class _TrackRecord:
    rule_id: str
    key: tuple
    evaluable: int = 0
    hits: list[float] = field(default_factory=list)
    best: Optional[_Evidence] = None
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None


@dataclass
class ConfirmedViolation:
    candidate: ViolationCandidate
    confidence: float
    frame_index: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection]
    frames_observed: int
    frames_evaluable: int
    first_frame: int
    last_frame: int


class TemporalAggregator:
    def __init__(self, s: Settings, stop_line_y: Optional[float] = None, line_direction: str = "any"):
        self.s = s
        self.records: dict[tuple, _TrackRecord] = {}
        self.unique_tracks: dict[str, set[int]] = {}
        self.untracked_max: dict[str, int] = {}
        # red light
        self.stop_line_y = stop_line_y  # pixels; None disables the module
        self.line_direction = line_direction  # "down", "up" or "any"
        self._last_side: dict[int, int] = {}
        self._first_side: dict[int, int] = {}
        self._red_light_done: set[int] = set()
        self.red_light: list[ConfirmedViolation] = []
        self.signal_frames = {"red": 0, "yellow": 0, "green": 0, "unknown": 0}

    # ----------------------------------------------------------------- counting
    def _count(self, dets: list[Detection]) -> None:
        per_label: dict[str, int] = {}
        for d in dets:
            if d.label not in VEHICLE_LABELS and d.label != "person":
                continue
            if d.track_id is not None:
                self.unique_tracks.setdefault(d.label, set()).add(d.track_id)
            else:
                per_label[d.label] = per_label.get(d.label, 0) + 1
        for k, v in per_label.items():
            self.untracked_max[k] = max(self.untracked_max.get(k, 0), v)

    def unique_counts(self) -> dict[str, int]:
        labels = set(self.unique_tracks) | set(self.untracked_max)
        return {k: len(self.unique_tracks.get(k, ())) + self.untracked_max.get(k, 0) for k in sorted(labels)}

    # ----------------------------------------------------------------- per-frame
    def _record(self, key: tuple, rule_id: str, frame_idx: int) -> _TrackRecord:
        rec = self.records.get(key)
        if rec is None:
            rec = self.records[key] = _TrackRecord(rule_id=rule_id, key=key)
        rec.first_frame = frame_idx if rec.first_frame is None else rec.first_frame
        rec.last_frame = frame_idx
        return rec

    def update(self, frame_idx: int, timestamp: float, frame: np.ndarray, fa: FrameAnalysis,
               signal_state: str, signal_agreement: float) -> None:
        self._count(fa.detections)
        self.signal_frames[signal_state] = self.signal_frames.get(signal_state, 0) + 1

        # Helmet: evaluable whenever a tracked rider got a definite helmet / no-helmet verdict.
        for unit in fa.units:
            vid = unit.vehicle.track_id
            for rider in unit.riders:
                pid = rider.person.track_id
                if vid is None or pid is None or rider.status not in {"helmet", "no_helmet"}:
                    continue
                self._record(("NO_HELMET", vid, pid), "NO_HELMET", frame_idx).evaluable += 1
            if vid is not None and unit.riders:
                self._record(("TRIPLE_RIDING", vid), "TRIPLE_RIDING", frame_idx).evaluable += 1

        for cand in fa.candidates:
            vid = cand.vehicle.track_id
            if vid is None:
                continue
            key = (cand.rule_id, vid, cand.subjects[0].track_id) if cand.rule_id == "NO_HELMET" else (cand.rule_id, vid)
            if None in key:
                continue
            rec = self._record(key, cand.rule_id, frame_idx)
            rec.hits.append(cand.confidence)
            if rec.best is None or cand.confidence > rec.best.confidence:
                rec.best = _Evidence(frame_idx, timestamp, cand.confidence, cand, frame.copy(), list(fa.detections))

        if self.stop_line_y is not None:
            self._update_red_light(frame_idx, timestamp, frame, fa, signal_state, signal_agreement)

    # --------------------------------------------------------------- red light
    def _update_red_light(self, frame_idx, timestamp, frame, fa: FrameAnalysis, state: str, agreement: float):
        y_line = self.stop_line_y
        for det in fa.detections:
            if det.label not in VEHICLE_LABELS or det.track_id is None:
                continue
            tid = det.track_id
            # Reference point: bottom-centre of the box (where the vehicle touches the road).
            side = 1 if det.box.y2 > y_line else -1  # +1 below the line, -1 above
            prev = self._last_side.get(tid)
            self._first_side.setdefault(tid, side)
            self._last_side[tid] = side
            if prev is None or prev == side or tid in self._red_light_done:
                continue
            moving_down = prev == -1 and side == 1
            if self.line_direction == "down" and not moving_down:
                continue
            if self.line_direction == "up" and moving_down:
                continue
            if self._first_side[tid] != prev:
                continue  # vehicle was not seen approaching the line from its first side
            if state != "red":
                continue
            self._red_light_done.add(tid)
            conf = round(min(1.0, det.confidence ** 0.5 * agreement * 0.95), 4)
            cand = ViolationCandidate(
                rule_id=RULE_RED_LIGHT, confidence=conf, vehicle=det, subjects=[],
                evidence=f"{det.label.title()} crossed the configured stop line while the detected signal was red",
                region=det.box,
                details={"signal_state": state, "signal_agreement": round(agreement, 3),
                         "stop_line_y": round(y_line, 1), "direction": "down" if moving_down else "up"},
            )
            self.red_light.append(ConfirmedViolation(cand, conf, frame_idx, timestamp, frame.copy(),
                                                     list(fa.detections), 1, 1, frame_idx, frame_idx))

    # ------------------------------------------------------------------ results
    def finalize(self) -> tuple[list[ConfirmedViolation], list[Observation]]:
        s = self.s
        confirmed: list[ConfirmedViolation] = []
        observations: list[Observation] = []
        for rec in self.records.values():
            if not rec.hits or rec.best is None:
                continue
            n, ev = len(rec.hits), max(rec.evaluable, len(rec.hits))
            ratio = n / ev
            if n >= s.temporal_min_frames and ratio >= s.temporal_min_ratio:
                conf = statistics.median(rec.hits) * (0.85 + 0.15 * ratio)
                b = rec.best
                confirmed.append(ConfirmedViolation(b.candidate, round(min(1.0, conf), 4), b.frame_index,
                                                    b.timestamp, b.frame, b.detections, n, ev,
                                                    rec.first_frame or 0, rec.last_frame or 0))
            else:
                observations.append(Observation(
                    rec.rule_id,
                    f"Observed in only {n} of {ev} analysed frames for track #{rec.key[1]} — "
                    "insufficient temporal evidence, not reported",
                    rec.best.candidate.region, statistics.median(rec.hits)))
        confirmed.extend(self.red_light)
        confirmed.sort(key=lambda c: c.frame_index)
        return confirmed, observations

    @staticmethod
    def stop_line_box(width: int, y: float) -> Box:
        return Box(0, y - 1, width, y + 1)
