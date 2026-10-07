"""Input preprocessing: decoding, validation, resizing and low-light enhancement."""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 1920  # larger frames are downscaled; YOLO itself runs at Settings.inference_imgsz


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("File could not be decoded as an image")
    if img.shape[0] < 32 or img.shape[1] < 32:
        raise InvalidImageError("Image is too small to analyse (minimum 32x32 px)")
    return img


def limit_size(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def brightness(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def enhance_low_light(img: np.ndarray, threshold: float = 60.0) -> tuple[np.ndarray, bool]:
    """Apply CLAHE on the luminance channel for dark (night-time) footage.

    Returns the (possibly) enhanced image and whether enhancement was applied. Bright
    images are returned unchanged so daytime results are not altered.
    """
    if brightness(img) >= threshold:
        return img, False
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR), True


def preprocess(img: np.ndarray) -> tuple[np.ndarray, dict]:
    img = limit_size(img)
    img, enhanced = enhance_low_light(img)
    h, w = img.shape[:2]
    return img, {"width": w, "height": h, "low_light_enhanced": enhanced}
