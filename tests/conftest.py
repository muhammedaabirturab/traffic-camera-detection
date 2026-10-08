"""Shared fixtures. Tests use a scripted detector so they exercise the rule/API plumbing deterministically
(model accuracy is measured separately by scripts/validate.py - these tests never fake metrics)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection.types import Detection
from app.main import create_app


class ScriptedDetector:
    """Stands in for YoloDetector: returns the detections it was given."""

    def __init__(self, dets=None, helmet=False):
        self.dets = dets or []
        self.helmet = helmet

    @property
    def helmet_available(self):
        return self.helmet

    def detect(self, frame, track=False):
        return [Detection(d.label, d.confidence, d.box, d.source, d.track_id) for d in self.dets]

    def reset_tracking(self):
        pass

    def status(self):
        return {"device": "cpu", "coco": None, "rider": None, "helmet": None, "problems": {}}


def triple_riding_scene():
    """A motorcycle with three persons seated on it (pixel boxes for a 640x480 frame)."""
    return [
        Detection("motorcycle", 0.92, (200, 260, 360, 440), "coco", 7),
        Detection("person", 0.90, (225, 140, 290, 400), "coco", 11),
        Detection("person", 0.88, (270, 150, 330, 395), "coco", 12),
        Detection("person", 0.86, (310, 170, 360, 390), "coco", 13),
    ]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", database_url=f"sqlite:///{(tmp_path / 'h.db').as_posix()}",
                    video_frame_skip=1, video_min_violation_frames=2)


@pytest.fixture
def make_client(settings):
    clients = []

    def _make(detector):
        c = TestClient(create_app(settings, detector))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def jpeg_bytes():
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 180, 40), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()
