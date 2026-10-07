"""Minimal background job runner for long video analyses.

Video analysis on a laptop CPU can take minutes, so the API returns a job id
immediately and the dashboard polls ``GET /api/jobs/{id}`` for progress. One worker
thread is used because inference is CPU/GPU bound; additional jobs queue up.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from app.utils.helpers import new_id, utc_now_iso

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tg-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[Callable[[float, str], None]], Any], analysis_id: str,
               on_error: Optional[Callable[[str], None]] = None) -> str:
        job_id = new_id()
        with self._lock:
            self._jobs[job_id] = {"job_id": job_id, "analysis_id": analysis_id, "status": "queued",
                                  "progress": 0.0, "message": "Queued", "created_at": utc_now_iso(), "error": None}

        def progress(fraction: float, message: str) -> None:
            with self._lock:
                self._jobs[job_id].update(progress=round(float(fraction), 3), message=message, status="running")

        def run():
            try:
                progress(0.0, "Starting")
                fn(progress)
                with self._lock:
                    self._jobs[job_id].update(status="completed", progress=1.0, message="Completed")
            except Exception as exc:  # reported to the client, logged with traceback
                log.exception("Job %s failed", job_id)
                with self._lock:
                    self._jobs[job_id].update(status="failed", message="Failed", error=str(exc))
                if on_error:
                    on_error(str(exc))

        self._pool.submit(run)
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def active(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values() if j["status"] in {"queued", "running"}]
