"""Small helpers used across the backend."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(name: str) -> str:
    """Strip directories and unusual characters from an uploaded filename."""
    base = Path(name or "upload").name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip() or "upload"
    return base[:120]


def extension(name: str) -> str:
    return Path(name or "").suffix.lower()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start

    @property
    def seconds(self) -> float:
        return round(time.perf_counter() - self.start, 3)
