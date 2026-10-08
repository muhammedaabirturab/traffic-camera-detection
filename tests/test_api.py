import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import ScriptedDetector, triple_riding_scene


def make_video(path, frames=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(frames):
        w.write(np.full((480, 640, 3), 60 + i, np.uint8))
    w.release()


def wait_job(c, job_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("video job timed out")


def test_health(make_client):
    body = make_client(ScriptedDetector()).get("/api/health").json()
    assert body["status"] == "ok" and body["models_loaded"] is True


def test_rules_and_model_info_endpoints(make_client):
    c = make_client(ScriptedDetector())
    rules = c.get("/api/rules").json()
    assert any(r["violation_id"] == "triple_riding" for r in rules["rules"])
    info = c.get("/api/model/info").json()
    if not info["metrics_available"]:
        assert "unavailable" in info["metrics_message"]


def test_image_upload_flags_triple_riding_and_saves_history(make_client, jpeg_bytes):
    c = make_client(ScriptedDetector(triple_riding_scene()))
    r = c.post("/api/analyze/image", files={"file": ("cam.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["summary"]["possible_violations"] == 1
    v = body["violations"][0]
    assert v["violation_id"] == "triple_riding" and v["requires_human_verification"]
    assert c.get(v["evidence_image"]).status_code == 200
    assert c.get(body["media"]["annotated"]).status_code == 200
    assert body["intelligence"]["risk_level"] in ("MODERATE", "HIGH")
    hist = c.get("/api/history").json()
    assert hist["items"][0]["id"] == body["id"] and hist["items"][0]["violations"] == 1
    assert c.get(f"/api/history/{body['id']}").json()["id"] == body["id"]
    assert c.delete(f"/api/history/{body['id']}").status_code == 200
    assert c.get(f"/api/history/{body['id']}").status_code == 404


def test_empty_detections_return_clean_result(make_client, jpeg_bytes):
    c = make_client(ScriptedDetector([]))
    body = c.post("/api/analyze/image", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert body["violations"] == [] and body["intelligence"]["risk_level"] == "LOW"
    assert any("No traffic objects" in n for n in body["notes"])


def test_invalid_uploads_get_friendly_errors(make_client, jpeg_bytes):
    c = make_client(ScriptedDetector())
    r = c.post("/api/analyze/image", files={"file": ("x.jpg", b"not really a jpg", "image/jpeg")})
    assert r.status_code == 400 and r.json()["code"] == "invalid_input" and "Traceback" not in r.text
    r = c.post("/api/analyze/image", files={"file": ("x.gif", jpeg_bytes, "image/gif")})
    assert r.status_code == 400 and "Unsupported" in r.json()["message"]
    r = c.post("/api/analyze/image")
    assert r.status_code == 422 and r.json()["status"] == "error"
    r = c.post("/api/analyze/video", files={"file": ("v.mp4", b"garbage", "video/mp4")})
    assert r.status_code == 202
    job = wait_job(c, r.json()["job_id"])
    assert job["status"] == "error" and "video" in job["error"].lower()


def test_missing_models_reported_not_crashed(settings, monkeypatch):
    import app.detection.yolo_detector as yd

    def fail(self):
        raise yd.ModelUnavailableError("No YOLO weights could be loaded.")

    monkeypatch.setattr(yd.YoloDetector, "_load", fail)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health").json()["status"] == "degraded"
        r = c.post("/api/analyze/image", files={"file": ("a.jpg", b"x", "image/jpeg")})
        assert r.status_code == 503 and "weights" in r.json()["message"]


def test_video_analysis_temporal_validation(make_client, tmp_path):
    c = make_client(ScriptedDetector(triple_riding_scene()))
    vid = tmp_path / "clip.mp4"
    make_video(vid)
    r = c.post("/api/analyze/video", files={"file": ("clip.mp4", vid.read_bytes(), "video/mp4")})
    assert r.status_code == 202
    job = wait_job(c, r.json()["job_id"])
    assert job["status"] == "done", job
    res = c.get(f"/api/history/{job['result_id']}").json()
    assert res["kind"] == "video" and res["video"]["frames_analyzed"] == 12
    assert res["summary"]["possible_violations"] == 1
    v = res["violations"][0]
    assert v["frames_observed"] >= 2 and "timestamp" in v
    assert c.get(res["media"]["processed_video"]).status_code == 200
    assert len(res["timeline"]) >= 1


def test_video_single_frame_blip_is_discarded(make_client, tmp_path):
    class Blip(ScriptedDetector):
        n = 0

        def detect(self, frame, track=False):
            Blip.n += 1
            self.dets = triple_riding_scene() if Blip.n == 3 else triple_riding_scene()[:2]
            return super().detect(frame, track)

    c = make_client(Blip())
    vid = tmp_path / "blip.mp4"
    make_video(vid)
    r = c.post("/api/analyze/video", files={"file": ("b.mp4", vid.read_bytes(), "video/mp4")})
    job = wait_job(c, r.json()["job_id"])
    res = c.get(f"/api/history/{job['result_id']}").json()
    assert res["summary"]["possible_violations"] == 0
    assert res["summary"]["insufficient_evidence"] == 1


def test_long_file_names_keep_their_extension(make_client, jpeg_bytes):
    c = make_client(ScriptedDetector([]))
    r = c.post("/api/analyze/image", files={"file": ("a" * 200 + ".jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200 and r.json()["filename"].endswith(".jpg")
