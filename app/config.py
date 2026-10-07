"""Central configuration for TrafficGuard AI.

Every threshold that influences whether something is reported as a possible
violation lives here, so it can be tuned (and explained in a viva) without
touching detection code. Values can be overridden with environment variables
prefixed ``TG_`` (for example ``TG_CONF_HIGH=0.9``) or a ``.env`` file in the
project root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TG_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    # ------------------------------------------------------------------ paths
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"  # uploads, results, history DB (git-ignored)
    rules_file: Path = PROJECT_ROOT / "app" / "rules" / "traffic_rules.json"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    # ------------------------------------------------------------------ models
    # General traffic detector. COCO-pretrained Ultralytics YOLO: detects
    # person, bicycle, car, motorcycle, bus, truck, traffic light (real COCO classes).
    # Downloaded automatically by Ultralytics on first use if not present.
    vehicle_model: str = "yolo11n.pt"
    # Helmet detector. Either an Ultralytics model you trained (helmet.pt) or the
    # Darknet YOLOv3 helmet model from the Kaggle reference (cfg + weights + names).
    helmet_model: str = "helmet.pt"
    helmet_darknet_cfg: str = "yolov3-helmet.cfg"
    helmet_darknet_weights: str = "yolov3-helmet.weights"
    helmet_darknet_names: str = "helmet.names"
    # Optional "person on two-wheeler" detector trained on the Kaggle
    # detect-person-on-motorbike-or-scooter dataset (Ultralytics or Darknet).
    rider_model: str = "rider.pt"
    rider_darknet_cfg: str = "yolov3_pb.cfg"
    rider_darknet_weights: str = "yolov3-obj_final.weights"
    rider_darknet_names: str = "rider.names"
    # Optional number-plate detector (no plate data in the reference dataset).
    plate_model: str = "plate.pt"
    enable_plate_ocr: bool = True  # only used if a plate model AND easyocr are available

    inference_imgsz: int = 640
    device: str = "cpu"  # "cpu", "0" for first CUDA GPU, "mps" on Apple silicon

    # ------------------------------------------------------- detection filters
    # Raw detections below these confidences are discarded before reasoning.
    min_conf_vehicle: float = 0.35
    min_conf_person: float = 0.35
    min_conf_helmet: float = 0.30
    min_conf_rider: float = 0.35
    min_conf_traffic_light: float = 0.30
    nms_iou: float = 0.5
    duplicate_iou: float = 0.7  # same-class boxes overlapping more than this are merged

    # ---------------------------------------------------- relationship analysis
    rider_association_min: float = 0.50  # person <-> motorcycle score to count as a rider
    rider_association_strong: float = 0.62  # required for every rider in a triple-riding call
    head_region_ratio: float = 0.30  # top fraction of the person box treated as head region
    helmet_head_overlap_min: float = 0.30  # fraction of helmet box that must lie in head region
    min_person_height_px: int = 48  # smaller riders are "insufficient evidence" for helmet checks
    max_riders_allowed: int = 2  # MV Act s.128: driver + one pillion

    # ------------------------------------------------------- confidence system
    conf_high: float = 0.85
    conf_medium: float = 0.65
    report_min_confidence: float = 0.45  # below this -> "insufficient visual evidence"

    # ---------------------------------------------------------------- video
    video_target_fps: float = 6.0  # frames analysed per second of footage (stride is derived)
    video_max_seconds: float = 180.0  # longer clips are truncated for responsiveness
    tracker: str = "bytetrack.yaml"  # or "botsort.yaml"
    temporal_min_frames: int = 3  # a violation must be observed in at least this many frames
    temporal_min_ratio: float = 0.6  # ...and in this share of frames where it could be evaluated
    signal_smoothing_frames: int = 5  # majority vote window for traffic-light colour

    # ---------------------------------------------------------------- uploads
    max_image_mb: int = 20
    max_video_mb: int = 300

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "history.sqlite3"

    @property
    def metrics_dir(self) -> Path:
        return self.models_dir / "metrics"

    def confidence_band(self, value: float) -> str:
        if value >= self.conf_high:
            return "high"
        if value >= self.conf_medium:
            return "medium"
        return "low"

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.data_dir, self.uploads_dir, self.results_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

    def public_thresholds(self) -> dict:
        """Thresholds exposed to the UI (Settings / About page)."""
        keys = [
            "min_conf_vehicle", "min_conf_person", "min_conf_helmet", "min_conf_rider",
            "rider_association_min", "rider_association_strong", "head_region_ratio",
            "helmet_head_overlap_min", "min_person_height_px", "max_riders_allowed",
            "conf_high", "conf_medium", "report_min_confidence", "video_target_fps",
            "temporal_min_frames", "temporal_min_ratio", "tracker", "inference_imgsz", "device",
        ]
        return {k: getattr(self, k) for k in keys}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
