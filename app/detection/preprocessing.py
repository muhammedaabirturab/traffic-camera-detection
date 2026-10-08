"""Input validation and light preprocessing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}


class InvalidInputError(ValueError):
    """User-facing input problem (message is safe to show in the UI)."""


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise InvalidInputError("The uploaded file is empty.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if img is None or img.size == 0:
        raise InvalidInputError("The file could not be read as an image - it may be corrupted or not a real JPG/PNG.")
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        raise InvalidInputError("The image is too small to analyse (minimum 32x32 pixels).")
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    k = max_side / m
    return cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)


def check_extension(filename: str, allowed: set[str], kind: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise InvalidInputError(f"Unsupported {kind} format '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed))}.")
    return ext
