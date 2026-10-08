"""TrafficGuard AI - FastAPI application.   Run:  python -m app.main"""
from __future__ import annotations

import logging
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.analysis.image_analyzer import analyze_image
from app.analysis.jobs import JobManager
from app.analysis.video_analyzer import VideoOptions, analyze_video
from app.config import ROOT_DIR, Settings, get_settings
from app.detection.preprocessing import IMAGE_EXTS, VIDEO_EXTS, InvalidInputError, check_extension
from app.detection.yolo_detector import InferenceError, ModelUnavailableError, YoloDetector
from app.model_info import model_info
from app.rules.rule_engine import RuleEngine
from app.storage.history import HistoryStore
from app.utils.helpers import new_id, safe_filename
from app.utils.logging_config import setup_logging

log = logging.getLogger("trafficguard")
START = time.time()


def error(status: int, message: str, code: str = "error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"status": "error", "code": code, "message": message})


def create_app(settings: Optional[Settings] = None, detector: Optional[YoloDetector] = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "media").mkdir(exist_ok=True)
    state = {"detector": detector, "load_error": None}
    rules = RuleEngine(s)
    history = HistoryStore(s.db_file, s.data_dir / "media")
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["detector"] is None:
            try:
                state["detector"] = YoloDetector(s)
            except ModelUnavailableError as exc:
                state["load_error"] = str(exc)
                log.error("%s", exc)
            except Exception:
                state["load_error"] = "The detection models failed to load. Check the server log."
                log.exception("Model loading failed")
        yield

    app = FastAPI(title="TrafficGuard AI", version="1.0.0",
                  description="YOLO-based Indian traffic violation detection (educational project).", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------------ error handling (no tracebacks to the UI)
    @app.exception_handler(InvalidInputError)
    async def _bad_input(_, exc: InvalidInputError):
        return error(400, str(exc), "invalid_input")

    @app.exception_handler(InferenceError)
    async def _inference(_, exc: InferenceError):
        return error(500, str(exc), "inference_error")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, exc: StarletteHTTPException):
        return error(exc.status_code, str(exc.detail), "http_error")

    @app.exception_handler(RequestValidationError)
    async def _validation(_, exc: RequestValidationError):
        return error(422, "The request was missing a file or had invalid parameters.", "validation_error")

    @app.exception_handler(Exception)
    async def _unhandled(_, exc: Exception):
        log.exception("Unhandled error")
        return error(500, "Something went wrong while processing the request.", "server_error")

    def need_detector() -> YoloDetector:
        d = state["detector"]
        if d is None:
            raise HTTPException(503, state["load_error"] or "Detection models are not loaded. See app/models/README.md.")
        return d

    def read_upload(file: UploadFile, max_mb: int) -> bytes:
        data = file.file.read(max_mb * 1024 * 1024 + 1)
        if len(data) > max_mb * 1024 * 1024:
            raise InvalidInputError(f"File is larger than the {max_mb} MB limit.")
        return data

    # ------------------------------------------------------------------ routes
    @app.get("/api/health")
    def health():
        d = state["detector"]
        return {
            "status": "ok" if d else "degraded",
            "models_loaded": d is not None,
            "message": None if d else state["load_error"],
            "helmet_model": bool(d and d.helmet_available),
            "device": d.status()["device"] if d else None,
            "uptime_seconds": round(time.time() - START, 1),
            "version": app.version,
        }

    @app.get("/api/model/info")
    def api_model_info():
        return model_info(state["detector"], s)

    @app.get("/api/rules")
    def api_rules():
        rules.reload()
        d = state["detector"]
        return rules.catalogue({"helmet_model": bool(d and d.helmet_available)})

    @app.post("/api/analyze/image")
    def api_analyze_image(file: UploadFile = File(...)):
        det = need_detector()
        name = safe_filename(file.filename or "image")
        check_extension(name, IMAGE_EXTS, "image")
        data = read_upload(file, s.max_image_mb)
        result = analyze_image(data, name, det, rules, s)
        history.save(result)
        return result

    @app.post("/api/analyze/video", status_code=202)
    def api_analyze_video(file: UploadFile = File(...), stop_line: Optional[float] = Form(None),
                          red_from: Optional[float] = Form(None), red_to: Optional[float] = Form(None),
                          direction: str = Form("down")):
        det = need_detector()
        name = safe_filename(file.filename or "video")
        ext = check_extension(name, VIDEO_EXTS, "video")
        if stop_line is not None and not 0.05 <= stop_line <= 0.95:
            raise InvalidInputError("stop_line must be between 0.05 and 0.95 (fraction of frame height).")
        if direction not in ("down", "up"):
            raise InvalidInputError("direction must be 'down' or 'up'.")
        aid = new_id()
        tmp_dir = s.data_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / f"{aid}{ext}"
        limit, written = s.max_video_mb * 1024 * 1024, 0
        with tmp.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise InvalidInputError(f"Video is larger than the {s.max_video_mb} MB limit.")
                out.write(chunk)
        if written == 0:
            tmp.unlink(missing_ok=True)
            raise InvalidInputError("The uploaded file is empty.")
        opts = VideoOptions(stop_line, red_from, red_to, direction)
        jobs.submit(aid, name,
                    lambda cb: analyze_video(str(tmp), aid, name, det, rules, s, opts, cb),
                    on_done=history.save, cleanup=tmp)
        return {"status": "queued", "job_id": aid}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown job.")
        return j

    @app.get("/api/history")
    def api_history(limit: int = 100, offset: int = 0):
        return {"items": history.list(min(max(limit, 1), 500), max(offset, 0)), "stats": history.stats()}

    @app.get("/api/history/{aid}")
    def api_history_item(aid: str):
        r = history.get(aid)
        if r is None:
            raise HTTPException(404, "Analysis not found.")
        return r

    @app.delete("/api/history/{aid}")
    def api_history_delete(aid: str):
        if not history.delete(aid):
            raise HTTPException(404, "Analysis not found.")
        return {"status": "deleted"}

    app.mount("/media", StaticFiles(directory=s.data_dir / "media"), name="media")
    figures = ROOT_DIR / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    app.mount("/figures", StaticFiles(directory=figures), name="figures")  # training curves for the Model page

    dist = ROOT_DIR / "frontend" / "dist"
    if (dist / "index.html").exists():  # optional: serve the built dashboard from the same port
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str, request: Request):
            if path.startswith(("api/", "media/", "figures/")):
                raise HTTPException(404, "Not found.")
            f = (dist / path).resolve()
            return FileResponse(f if f.is_file() and dist.resolve() in f.parents else dist / "index.html")

    app.state.s = s
    app.state.history = history
    app.state.rules = rules
    app.state.jobs = jobs
    app.state.holder = state
    return app


app = create_app()


def main() -> None:
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)


if __name__ == "__main__":
    main()
