"""Annotated-image drawing. Three visual states: normal / possible violation / low-confidence."""
from __future__ import annotations

import cv2
import numpy as np

# BGR colours
COLORS = {
    "normal": (255, 200, 0),       # cyan-blue
    "violation": (40, 90, 255),    # orange-red
    "insufficient": (0, 190, 255), # amber
    "low": (175, 165, 160),        # grey
}


def draw_box(img: np.ndarray, box, label: str, state: str = "normal", thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    color = COLORS.get(state, COLORS["normal"])
    if state == "low":  # dashed look for low-confidence detections
        _dashed_rect(img, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not label:
        return
    scale = max(0.4, min(0.7, img.shape[1] / 1600))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 8
    cv2.rectangle(img, (x1, ty - th - 5), (x1 + tw + 8, ty + base - 2), color, -1)
    cv2.putText(img, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 1, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, t, dash=8) -> None:
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, t)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, t)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, t)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, t)


def _tid_label(tid) -> str:
    """ByteTrack ids as #n; ids from the fallback IoU tracker (>= 100000) as #Rn."""
    if tid is None:
        return ""
    return f" #{tid}" if tid < 100_000 else f" #R{tid - 100_000}"


def draw_overlay(frame: np.ndarray, overlay: list[dict]) -> np.ndarray:
    """Draw overlay items (see analysis.summary.build_overlay). Minor items (pedestrians) are skipped."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Keep the picture readable: rider boxes only appear when they are part of a violation.
    items = [o for o in overlay if not o.get("minor") and (o.get("group") != "rider" or o["state"] == "violation")]
    # draw violations last so they stay on top
    for o in sorted(items, key=lambda o: o["state"] == "violation"):
        x1, y1, x2, y2 = o["box_norm"]
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        tid = _tid_label(o.get("track_id"))
        label = f"{o['label'].upper()}{tid} {int(round(o['confidence'] * 100))}%"
        if o.get("riders"):
            label += f" ({o['riders']} on board)"
        if o["state"] == "violation":
            label = "! " + label
        draw_box(out, box, label, o["state"], 3 if o["state"] == "violation" else 2)
    return out


def crop_with_padding(img: np.ndarray, box, pad: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(x2 + pad * bw))
    y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(y2 + pad * bh))
    return img[y1:y2, x1:x2].copy()
