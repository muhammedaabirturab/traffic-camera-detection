"""Logging setup shared by the API server and the CLI scripts."""

import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = (level or os.getenv("TG_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ultralytics and uvicorn are chatty at INFO; keep the console readable during demos.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
