"""TrafficGuard AI — FastAPI application.

Run with ``python run.py`` (or ``uvicorn app.main:app``). The built React dashboard in
``frontend/dist`` is served from the same server, so a single command is enough for a
demo. API documentation is available at ``/docs``.
"""

from __future__ import annotations

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.analysis.image_analyzer import ImageAnalyzer
from app.analysis.jobs import JobManager
from app.analysis.video_analyzer import InvalidVideoError, VideoAnalyzer
from app.config import get_settings
from app.detection.preprocessing import InvalidImageError
from app.detection.yolo_detector import ModelManager
from app.model_info import model_info
from app.rules.rule_engine import RuleEngine
from app.storage.history import HistoryStore
from app.utils.helpers import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT, extension, new_id, safe_filename, utc_now_iso
from app.utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("trafficguard")

settings = get_settings()
models = ModelManager(settings)
rules = RuleEngine(settings)
history = HistoryStore(settings.db_path)
jobs = JobManager()
image_analyzer = ImageAnalyzer(settings, models, rules)
video_analyzer = VideoAnalyzer(settings, models, rules)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load models in the background so the server (and the dashboard) start instantly.
    threading.Thread(target=models.load, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="TrafficGuard AI",
    description="YOLO-based Indian traffic violation detection & analysis API. "
                "All outputs are AI-detected *possible* violations that require human verification.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory=settings.results_dir), name="media")
if settings.metrics_dir.exists():
    app.mount("/model-assets", StaticFiles(directory=settings.metrics_dir), name="model-assets")


# ---------------------------------------------------------------------- helpers
async def _read_upload(file: UploadFile, allowed: set[str], max_mb: int, kind: str) -> tuple[str, bytes]:
    name = safe_filename(file.filename or f"upload.{kind}")
    ext = extension(name)
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported {kind} type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (limit {max_mb} MB)")
    return name, data


# ---------------------------------------------------------------------- system
@app.get("/api/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "version": __version__,
        "models_loaded": models._loaded,
        "detectors": {k: v["available"] for k, v in models.status().items()} if models._loaded else None,
        "active_jobs": len(jobs.active()),
    }


@app.get("/api/settings", tags=["system"])
def get_thresholds():
    return {"thresholds": settings.public_thresholds(),
            "note": "Edit app/config.py or set TG_* environment variables, then restart the server."}


# ---------------------------------------------------------------------- analysis
@app.post("/api/analyze/image", tags=["analysis"])
async def analyze_image(file: UploadFile = File(..., description="Traffic image (JPG, JPEG or PNG)")):
    name, data = await _read_upload(file, ALLOWED_IMAGE_EXT, settings.max_image_mb, "image")
    analysis_id, created = new_id(), utc_now_iso()
    history.create(analysis_id, "image", name, created)
    try:
        result = await run_in_threadpool(image_analyzer.analyze_bytes, data, name, analysis_id, created)
    except InvalidImageError as exc:
        history.delete(analysis_id)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Image analysis failed")
        history.fail(analysis_id, str(exc))
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    history.complete(analysis_id, result)
    return result


@app.post("/api/analyze/video", tags=["analysis"])
async def analyze_video(
    file: UploadFile = File(..., description="Traffic video (MP4, AVI or MOV)"),
    stop_line: Optional[float] = Form(None, ge=0.05, le=0.95,
                                      description="Optional stop-line position as a fraction of frame height "
                                                  "(enables the red-light module)"),
    line_direction: Literal["any", "down", "up"] = Form("any", description="Direction of travel across the stop line"),
    wait: bool = Query(False, description="Process synchronously and return the full result"),
):
    name, data = await _read_upload(file, ALLOWED_VIDEO_EXT, settings.max_video_mb, "video")
    analysis_id, created = new_id(), utc_now_iso()
    upload_path = settings.uploads_dir / f"{analysis_id}{extension(name)}"
    upload_path.write_bytes(data)
    history.create(analysis_id, "video", name, created)

    def work(progress=None):
        try:
            result = video_analyzer.analyze(upload_path, name, analysis_id, created, stop_line, line_direction, progress)
        except InvalidVideoError as exc:
            history.fail(analysis_id, str(exc))
            raise
        history.complete(analysis_id, result)
        return result

    if wait:
        try:
            return await run_in_threadpool(work)
        except InvalidVideoError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            log.exception("Video analysis failed")
            history.fail(analysis_id, str(exc))
            raise HTTPException(500, f"Analysis failed: {exc}") from exc

    job_id = jobs.submit(work, analysis_id, on_error=lambda err: history.fail(analysis_id, err))
    return JSONResponse(status_code=202, content={"status": "processing", "job_id": job_id, "analysis_id": analysis_id})


@app.get("/api/jobs/{job_id}", tags=["analysis"])
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ---------------------------------------------------------------------- rules / model
@app.get("/api/rules", tags=["rules"])
def get_rules():
    return rules.as_dict()


@app.get("/api/model/info", tags=["model"])
def get_model_info():
    return model_info(settings, models)


@app.post("/api/model/reload", tags=["model"])
async def reload_models():
    await run_in_threadpool(models.reload)
    return {"status": "reloaded", "detectors": models.status()}


# ---------------------------------------------------------------------- history
@app.get("/api/history", tags=["history"])
def list_history(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                 kind: Optional[Literal["image", "video"]] = None):
    return {"items": history.list(limit, offset, kind)}


@app.get("/api/history/{analysis_id}", tags=["history"])
def get_history(analysis_id: str):
    item = history.get(analysis_id)
    if item is None:
        raise HTTPException(404, "Analysis not found")
    return item


@app.delete("/api/history/{analysis_id}", tags=["history"])
def delete_history(analysis_id: str):
    if not history.delete(analysis_id):
        raise HTTPException(404, "Analysis not found")
    shutil.rmtree(settings.results_dir / safe_filename(analysis_id), ignore_errors=True)
    for p in settings.uploads_dir.glob(f"{safe_filename(analysis_id)}.*"):
        p.unlink(missing_ok=True)
    return {"status": "deleted"}


@app.get("/api/stats", tags=["history"])
def stats():
    return history.stats()


# ---------------------------------------------------------------------- frontend
dist: Path = settings.frontend_dist
if (dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "TrafficGuard AI API is running. Build the dashboard with "
                           "`cd frontend && npm install && npm run build`, or run `npm run dev`. API docs: /docs"}
