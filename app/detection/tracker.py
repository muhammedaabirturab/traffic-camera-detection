"""Tracking helpers.

Vehicle/person tracking itself is done by Ultralytics' ByteTrack (see YoloDetector.detect(track=True)).
Rider-model-only two-wheelers have no ByteTrack id, so `IoUTracker` gives them stable ids by greedy IoU matching.
"""
from __future__ import annotations

from app.detection.association import iou
from app.detection.types import Box


class IoUTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 8, id_offset: int = 100_000):
        self.iou_thr, self.max_age, self.next_id = iou_thr, max_age, id_offset
        self.tracks: dict[int, tuple[Box, int]] = {}  # id -> (last box, frame index last seen)

    def update(self, boxes: list[Box], frame_idx: int) -> list[int]:
        ids: list[int] = [-1] * len(boxes)
        used: set[int] = set()
        pairs = sorted(((iou(b, tb), i, tid) for i, b in enumerate(boxes) for tid, (tb, _) in self.tracks.items()),
                       reverse=True)
        for score, i, tid in pairs:
            if score < self.iou_thr:
                break
            if ids[i] == -1 and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] == -1:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = (b, frame_idx)
        self.tracks = {t: v for t, v in self.tracks.items() if frame_idx - v[1] <= self.max_age}
        return ids
