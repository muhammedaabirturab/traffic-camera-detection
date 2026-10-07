"""API tests with the YOLO models replaced by a deterministic fake, so they run offline and fast."""

import os
import tempfile

import cv2
import numpy as np
import pytest

os.environ["TG_DATA_DIR"] = tempfile.mkdtemp(prefix="tg-test-")
os.environ["TG_MODELS_DIR"] = tempfile.mkdtemp(prefix="tg-models-")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from conftest import det  # noqa: E402

SCENE = [det("motorcycle", (100, 200, 300, 320), 0.9), det("person", (150, 90, 250, 290), 0.88),
         det("car", (400, 150, 600, 300), 0.8)]


class FakeTracker:
    def track(self, image, tracker):
        out = []
        for d in SCENE:
            c = det(d.label, d.box.as_list(), d.confidence, track_id=1 + SCENE.index(d))
            out.append(c)
        return out


@pytest.fixture(scope="module")
def client():
    m = main.models
    m._loaded = True
    m.vehicle = object()  # non-None marks the vehicle detector as available
    m.load = lambda: None
    m.detect_all = lambda img: [det(d.label, d.box.as_list(), d.confidence) for d in SCENE]
    m.detect_auxiliary = lambda img: []
    m.status = lambda: {k: {"available": k == "vehicle"} for k in ("vehicle", "helmet", "rider", "plate")}
    m.new_tracking_detector = lambda: FakeTracker()
    type(m).helmet_available = property(lambda self: True)
    with TestClient(main.app) as c:
        yield c


def _jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((480, 640, 3), 120, np.uint8))[1].tobytes()


def _video(path):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for _ in range(20):
        w.write(np.full((480, 640, 3), 120, np.uint8))
    w.release()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_rules_endpoint(client):
    ids = [r["violation_id"] for r in client.get("/api/rules").json()["rules"]]
    assert "NO_HELMET" in ids and "TRIPLE_RIDING" in ids


def test_analyze_image_flow(client):
    r = client.post("/api/analyze/image", files={"file": ("street.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["summary"]["vehicles"] == 2
    assert [v["rule_id"] for v in body["violations"]] == ["NO_HELMET"]
    v = body["violations"][0]
    assert v["status"] == "Requires human verification"
    assert client.get(v["evidence_images"]["crop"]).status_code == 200
    # stored in history and can be reopened
    item = client.get(f"/api/history/{body['analysis_id']}").json()
    assert item["violations"] == 1 and item["result"]["analysis_id"] == body["analysis_id"]
    assert client.get("/api/stats").json()["violations_by_type"]["NO_HELMET"] >= 1


def test_rejects_wrong_type(client):
    r = client.post("/api/analyze/image", files={"file": ("doc.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/analyze/image", files={"file": ("x.jpg", b"garbage", "image/jpeg")})
    assert r.status_code == 422


def test_analyze_video_sync(client, tmp_path):
    p = tmp_path / "clip.avi"
    _video(p)
    r = client.post("/api/analyze/video?wait=true", files={"file": ("clip.avi", p.read_bytes(), "video/x-msvideo")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["unique_vehicles"] == 2
    assert [v["rule_id"] for v in body["violations"]] == ["NO_HELMET"]  # persisted across frames
    assert body["violations"][0]["frames_observed"] >= 3


def test_video_stop_line_validation(client, tmp_path):
    p = tmp_path / "clip.avi"
    _video(p)
    r = client.post("/api/analyze/video", data={"stop_line": "1.5"},
                    files={"file": ("clip.avi", p.read_bytes(), "video/x-msvideo")})
    assert r.status_code == 422


def test_history_delete(client):
    r = client.post("/api/analyze/image", files={"file": ("a.png", _jpeg(), "image/png")})
    aid = r.json()["analysis_id"]
    assert client.delete(f"/api/history/{aid}").status_code == 200
    assert client.get(f"/api/history/{aid}").status_code == 404


def test_model_info_reports_untrained(client):
    info = client.get("/api/model/info").json()
    assert info["trained"] is False
