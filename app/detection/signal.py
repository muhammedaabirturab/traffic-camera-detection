"""Traffic-signal state estimation for the (video-only) red-light module.

COCO-pretrained YOLO detects the *traffic light* object but not its colour, so the
colour is estimated from the lit pixels inside the detected box using HSV thresholds,
then smoothed over several frames. When no light is visible or the colour is ambiguous
the state is ``unknown`` and the red-light rule is simply not evaluated.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np

from app.detection.types import Detection

_RANGES = {
    "red": [((0, 110, 110), (10, 255, 255)), ((165, 110, 110), (180, 255, 255))],
    "yellow": [((15, 110, 110), (35, 255, 255))],
    "green": [((40, 80, 100), (95, 255, 255))],
}


def classify_light(image: np.ndarray, det: Detection, min_ratio: float = 0.03) -> tuple[str, float]:
    """Return (state, lit_pixel_ratio) for one traffic-light detection."""
    h, w = image.shape[:2]
    b = det.box.clip(w, h)
    crop = image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 3:
        return "unknown", 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = crop.shape[0] * crop.shape[1]
    ratios = {}
    for colour, ranges in _RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        ratios[colour] = float(mask.sum() / 255) / total
    colour, ratio = max(ratios.items(), key=lambda kv: kv[1])
    return (colour, ratio) if ratio >= min_ratio else ("unknown", ratio)


class SignalStateEstimator:
    """Majority vote of the per-frame light colour over a sliding window."""

    def __init__(self, window: int = 5):
        self.history: deque[str] = deque(maxlen=window)

    def update(self, image: np.ndarray, detections: list[Detection]) -> tuple[str, float]:
        lights = [d for d in detections if d.label == "traffic_light"]
        state = "unknown"
        if lights:
            # Use the largest, most confident light: usually the one governing the camera's lane.
            best = max(lights, key=lambda d: d.confidence * (d.box.area ** 0.5))
            state, _ = classify_light(image, best)
        self.history.append(state)
        return self.current()

    def current(self) -> tuple[str, float]:
        if not self.history:
            return "unknown", 0.0
        counts = Counter(self.history)
        state, n = counts.most_common(1)[0]
        agreement = n / len(self.history)
        if state == "unknown" or agreement < 0.6:
            return "unknown", agreement
        return state, agreement

    @property
    def last_raw(self) -> Optional[str]:
        return self.history[-1] if self.history else None
