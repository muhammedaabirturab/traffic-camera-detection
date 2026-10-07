"""Drawing helpers for annotated frames and evidence images (OpenCV, BGR colours)."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from app.detection.types import Box, Detection

COLOURS = {
    "motorcycle": (255, 200, 0),
    "bicycle": (255, 230, 120),
    "car": (255, 140, 60),
    "bus": (210, 120, 255),
    "truck": (180, 110, 230),
    "person": (225, 225, 225),
    "rider": (200, 255, 150),
    "helmet": (110, 220, 80),
    "no_helmet": (0, 140, 255),
    "traffic_light": (0, 220, 255),
    "number_plate": (255, 255, 255),
}
VIOLATION = (60, 60, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, text: str, x: int, y: int, colour: tuple, scale: float) -> None:
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(th + base + 2, y)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), colour, -1)
    lum = 0.114 * colour[0] + 0.587 * colour[1] + 0.299 * colour[2]
    cv2.putText(img, text, (x + 3, y - base - 1), FONT, scale, (20, 20, 20) if lum > 140 else (255, 255, 255),
                thickness, cv2.LINE_AA)


def draw_detections(img: np.ndarray, dets: Iterable[Detection], show_track: bool = True,
                    labels: bool = True) -> np.ndarray:
    out = img.copy()
    scale = max(0.4, min(1.0, img.shape[1] / 1600))
    thick = max(1, int(round(img.shape[1] / 600)))
    for d in dets:
        c = COLOURS.get(d.label, (200, 200, 200))
        b = d.box
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, thick)
        if not labels:
            continue
        text = f"{d.label.replace('_', ' ').upper()} {d.confidence:.0%}"
        if show_track and d.track_id is not None:
            text = f"#{d.track_id} " + text
        _label(out, text, int(b.x1), int(b.y1), c, scale * 0.6)
    return out


def draw_violation(img: np.ndarray, region: Box, title: str, confidence: Optional[float] = None,
                   label: bool = True) -> np.ndarray:
    out = img.copy()
    thick = max(2, int(round(img.shape[1] / 400)))
    b = region.expand(0.03).clip(img.shape[1], img.shape[0])
    cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), VIOLATION, thick)
    if not label:
        return out
    text = f"POSSIBLE: {title.upper()}" + (f" {confidence:.0%}" if confidence is not None else "")
    _label(out, text, int(b.x1), int(b.y2) + 24, VIOLATION, max(0.45, img.shape[1] / 2400))
    return out


def draw_stop_line(img: np.ndarray, y: float, signal_state: str) -> np.ndarray:
    colour = {"red": (60, 60, 235), "green": (90, 200, 60), "yellow": (0, 210, 255)}.get(signal_state, (180, 180, 180))
    out = img.copy()
    cv2.line(out, (0, int(y)), (img.shape[1], int(y)), colour, max(2, img.shape[1] // 500))
    _label(out, f"STOP LINE | SIGNAL: {signal_state.upper()}", 8, int(y) - 6, colour, 0.5)
    return out


def draw_hud(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    _label(out, text, 8, 26, (40, 30, 20), 0.55)
    return out


def crop(img: np.ndarray, region: Box, pad: float = 0.15, min_side: int = 160) -> np.ndarray:
    h, w = img.shape[:2]
    b = region.expand(pad)
    if b.w < min_side:
        b = Box(b.cx - min_side / 2, b.y1, b.cx + min_side / 2, b.y2)
    if b.h < min_side:
        b = Box(b.x1, b.cy - min_side / 2, b.x2, b.cy + min_side / 2)
    b = b.clip(w, h)
    return img[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].copy()
