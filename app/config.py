"""Central configuration. Every value can be overridden with an environment variable or a `.env` file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- models -----------------------------------------------------------------------------
    # Rider detector: YOLO fine-tuned on the `person_bike` dataset (scripts/train.py).
    model_path: Path = APP_DIR / "models" / "rider_detector.pt"
    # General vehicle/person detector (COCO). Downloaded automatically by Ultralytics if missing.
    coco_model_path: Path = APP_DIR / "models" / "yolov8n.pt"
    # Optional helmet detector. Not shipped: the reference dataset has no helmet labels.
    helmet_model_path: Path = APP_DIR / "models" / "helmet_detector.pt"
    helmet_class_names: str = "helmet,with helmet,with_helmet,hardhat"
    no_helmet_class_names: str = "no_helmet,no helmet,without helmet,without_helmet,head"

    # --- inference --------------------------------------------------------------------------
    confidence_threshold: float = Field(0.40, ge=0.01, le=0.99)
    iou_threshold: float = Field(0.45, ge=0.05, le=0.95)
    device: str = "auto"  # auto -> CUDA if available, else CPU
    image_size: int = 640
    max_side: int = 1280  # larger inputs are downscaled before inference

    # --- confidence bands (UI + rule engine) --------------------------------------------------
    high_confidence: float = 0.85
    medium_confidence: float = 0.65
    min_violation_confidence: float = 0.50  # below this a finding becomes "insufficient evidence"
    min_association_score: float = 0.60  # person <-> two-wheeler "rider" link
    weak_association_score: float = 0.40

    # --- video ------------------------------------------------------------------------------
    video_frame_skip: int = Field(2, ge=1)  # analyse every Nth frame
    video_min_violation_frames: int = Field(3, ge=1)  # temporal validation
    video_max_seconds: int = 120

    # --- storage / server -------------------------------------------------------------------
    database_url: str = ""  # empty -> sqlite in data/
    data_dir: Path = ROOT_DIR / "data"
    max_image_mb: int = 15
    max_video_mb: int = 200
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @property
    def db_file(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            p = Path(self.database_url.removeprefix("sqlite:///"))
            return p if p.is_absolute() else ROOT_DIR / p
        return self.data_dir / "trafficguard.db"

    @property
    def rules_path(self) -> Path:
        return APP_DIR / "rules" / "traffic_rules.json"

    def helmet_names(self) -> set[str]:
        return {n.strip().lower() for n in self.helmet_class_names.split(",") if n.strip()}

    def no_helmet_names(self) -> set[str]:
        return {n.strip().lower() for n in self.no_helmet_class_names.split(",") if n.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
