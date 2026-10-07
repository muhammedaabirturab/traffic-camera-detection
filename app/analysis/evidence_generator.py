"""Evidence generation: annotated frames, cropped evidence images and thumbnails.

Files are written to ``data/results/<analysis_id>/`` and served by the API under
``/media/<analysis_id>/<file>`` so the dashboard and the history page can show them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.detection.types import Box, Detection
from app.utils import visualization as viz


class EvidenceWriter:
    def __init__(self, results_dir: Path, analysis_id: str):
        self.analysis_id = analysis_id
        self.dir = Path(results_dir) / analysis_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def url(self, name: str) -> str:
        return f"/media/{self.analysis_id}/{name}"

    def save(self, name: str, img: np.ndarray, quality: int = 90) -> str:
        cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return self.url(name)

    def thumbnail(self, img: np.ndarray, name: str = "thumb.jpg", width: int = 360) -> str:
        h, w = img.shape[:2]
        scale = width / w
        small = cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else img
        return self.save(name, small, quality=80)

    def violation_evidence(self, index: int, frame: np.ndarray, detections: list[Detection], region: Box,
                           title: str, confidence: float, extra: Optional[list[Detection]] = None) -> dict:
        """Save a full annotated frame and a close-up crop for one violation."""
        relevant = [d for d in detections if d.box.intersection(region.expand(0.2)) > 0]
        if extra:
            relevant += [d for d in extra if d not in relevant]
        annotated = viz.draw_detections(frame, relevant)
        annotated = viz.draw_violation(annotated, region, title, confidence)
        full_url = self.save(f"violation_{index:02d}_frame.jpg", annotated)
        # The close-up uses thin boxes without text so the rider stays clearly visible.
        plain = viz.draw_violation(viz.draw_detections(frame, relevant, labels=False), region, title, label=False)
        crop_url = self.save(f"violation_{index:02d}_crop.jpg", viz.crop(plain, region))
        return {"frame": full_url, "crop": crop_url}
