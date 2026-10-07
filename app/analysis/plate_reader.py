"""Optional number-plate reading.

The Kaggle reference data contains no number-plate annotations, so this module is
switched off unless BOTH of these are provided:

* a plate detector at ``models/plate.pt`` (any Ultralytics YOLO model whose classes
  include "license plate" / "number plate"), and
* the optional ``easyocr`` package (``pip install -r requirements-optional.txt``).

Readings are always labelled as AI-generated and are never used to decide whether a
violation occurred.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

from app.detection.types import Box, Detection

log = logging.getLogger(__name__)
_reader = None
_reader_lock = threading.Lock()
_reader_failed = False


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def _get_reader():
    global _reader, _reader_failed
    with _reader_lock:
        if _reader is None and not _reader_failed:
            try:
                import easyocr
                _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:  # missing package or model download failure
                log.warning("easyocr unavailable; plate OCR disabled", exc_info=True)
                _reader_failed = True
        return _reader


def find_plate(vehicle: Box, detections: list[Detection]) -> Optional[Detection]:
    area = vehicle.expand(0.1)
    plates = [d for d in detections if d.label == "number_plate" and d.box.intersection(area) / max(d.box.area, 1) > 0.6]
    return max(plates, key=lambda d: d.confidence, default=None)


def read_plate(image: np.ndarray, plate: Detection) -> Optional[dict]:
    reader = _get_reader()
    if reader is None:
        return None
    h, w = image.shape[:2]
    b = plate.box.expand(0.05).clip(w, h)
    crop = image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
    if crop.size == 0:
        return None
    results = reader.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    if not results:
        return None
    text = re.sub(r"\s+", "", "".join(r[1] for r in results)).upper()
    conf = float(sum(r[2] for r in results) / len(results))
    return {
        "text": text,
        "ocr_confidence": round(conf, 3),
        "plate_detection_confidence": round(plate.confidence, 3),
        "bbox": plate.box.as_list(),
        "note": "AI-generated reading — may be incorrect; verify manually. Not legal proof of registration.",
    }
