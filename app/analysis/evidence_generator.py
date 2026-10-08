"""Evidence images: an annotated crop around each violation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.utils.visualization import crop_with_padding


def save_evidence(annotated_frame: np.ndarray, box, out_path: Path, max_side: int = 900) -> Path:
    """Crop `box` (pixels) from an already-annotated frame, with context padding, and save as JPEG."""
    crop = crop_with_padding(annotated_frame, box, pad=0.35)
    h, w = crop.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        crop = cv2.resize(crop, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
