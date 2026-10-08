from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from app.config import Settings


def confidence_level(conf: float, s: Settings) -> str:
    """Map a 0..1 confidence to the configurable bands: high / medium / low."""
    if conf >= s.high_confidence:
        return "high"
    if conf >= s.medium_confidence:
        return "medium"
    return "low"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(name: str, default: str = "upload") -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or default).strip("._") or default
    stem, dot, ext = base.rpartition(".")
    if not dot:
        return base[:80]
    return f"{stem[:80]}.{ext[:8]}"  # shorten the stem, never the extension


def format_timestamp(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"
