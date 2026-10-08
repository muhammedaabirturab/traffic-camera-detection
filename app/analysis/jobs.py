"""Background execution of video analyses (one at a time - the GPU/CPU is the bottleneck)."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from app.detection.preprocessing import InvalidInputError
from app.detection.yolo_detector import InferenceError

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit(self, job_id: str, filename: str, work: Callable[[Callable[[float], None]], dict],
               on_done: Callable[[dict], None], cleanup: Path | None = None) -> None:
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "filename": filename, "error": None}

        def progress(p: float) -> None:
            with self.lock:
                self.jobs[job_id].update(status="running", progress=round(p, 3))

        def run() -> None:
            try:
                progress(0.0)
                result = work(progress)
                on_done(result)
                with self.lock:
                    self.jobs[job_id].update(status="done", progress=1.0, result_id=result["id"])
            except (InvalidInputError, InferenceError) as exc:
                with self.lock:
                    self.jobs[job_id].update(status="error", error=str(exc))
            except Exception:  # never leak a traceback to the UI
                log.exception("Video job %s failed", job_id)
                with self.lock:
                    self.jobs[job_id].update(status="error", error="Video analysis failed unexpectedly. Check the server log.")
            finally:
                if cleanup is not None:
                    cleanup.unlink(missing_ok=True)

        self.pool.submit(run)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return dict(j) if j else None
