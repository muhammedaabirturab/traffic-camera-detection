"""YOLO inference wrapper.

Up to three Ultralytics YOLO models cooperate:

* COCO model   - vehicles (car, bus, truck, motorcycle, bicycle) and persons. Pretrained, downloaded on demand.
* rider model  - fine-tuned on the reference dataset; one class `person_bike` (a person on a two-wheeler).
* helmet model - OPTIONAL. Only used if a helmet-labelled model exists at HELMET_MODEL_PATH.

Every model is optional except that at least one must load; the app reports which are active.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from app.config import Settings
from app.detection.types import Detection

log = logging.getLogger(__name__)

COCO_LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class ModelUnavailableError(RuntimeError):
    """Raised when no detector could be loaded."""


class InferenceError(RuntimeError):
    """Raised when a model fails while running."""


def resolve_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


class YoloDetector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.device = resolve_device(settings.device)
        self.lock = threading.Lock()
        self.coco = None  # used for single images (predict)
        self.coco_tracker = None  # separate instance so tracker state never leaks into predict()
        self.rider = None
        self.helmet = None
        self.load_errors: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        from ultralytics import YOLO

        def try_load(key: str, path, optional_msg: str):
            try:
                if not path.exists():
                    if key == "coco":  # Ultralytics can fetch official COCO weights itself
                        path.parent.mkdir(parents=True, exist_ok=True)
                        from ultralytics.utils.downloads import attempt_download_asset
                        attempt_download_asset(str(path))
                    if not path.exists():
                        self.load_errors[key] = optional_msg
                        return None
                return YOLO(str(path))
            except Exception as exc:
                log.warning("Could not load %s model: %s", key, exc)
                self.load_errors[key] = f"{optional_msg} ({type(exc).__name__})"
                return None

        self.coco = try_load("coco", self.s.coco_model_path, f"COCO weights not found at {self.s.coco_model_path}")
        if self.coco is not None:
            self.coco_tracker = YOLO(str(self.s.coco_model_path))
        self.rider = try_load("rider", self.s.model_path, f"Rider model not found at {self.s.model_path} - run scripts/train.py")
        self.helmet = try_load("helmet", self.s.helmet_model_path, "No helmet model installed (optional)")
        if self.coco is None and self.rider is None:
            raise ModelUnavailableError(
                "No YOLO weights could be loaded. Place weights in app/models/ or run scripts/train.py (see app/models/README.md)."
            )
        log.info("Detector ready on device=%s coco=%s rider=%s helmet=%s", self.device,
                 self.coco is not None, self.rider is not None, self.helmet is not None)

    # ------------------------------------------------------------------ status
    @property
    def helmet_available(self) -> bool:
        return self.helmet is not None

    def status(self) -> dict:
        def info(model, path):
            return None if model is None else {"path": path.name, "classes": list(model.names.values())}
        return {
            "device": "cuda:0" if self.device != "cpu" else "cpu",
            "coco": info(self.coco, self.s.coco_model_path),
            "rider": info(self.rider, self.s.model_path),
            "helmet": info(self.helmet, self.s.helmet_model_path),
            "problems": self.load_errors,
        }

    # ------------------------------------------------------------------ inference
    def _run(self, model, frame: np.ndarray, track: bool = False):
        kw = dict(conf=self.s.confidence_threshold, iou=self.s.iou_threshold, imgsz=self.s.image_size,
                  device=self.device, verbose=False)
        try:
            if track:
                return model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
            return model.predict(frame, **kw)[0]
        except Exception as exc:
            log.exception("Inference failed")
            raise InferenceError("The detection model failed while processing this input.") from exc

    @staticmethod
    def _parse(result, mapper, source: str) -> list[Detection]:
        out: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else [None] * len(cls)
        for b, c, k, tid in zip(xyxy, conf, cls, ids):
            label = mapper(int(k), result.names)
            if label is None:
                continue
            out.append(Detection(label=label, confidence=float(c), box=tuple(float(v) for v in b),
                                 source=source, track_id=None if tid is None else int(tid)))
        return out

    def _helmet_label(self, k: int, names: dict) -> Optional[str]:
        n = str(names.get(k, "")).lower()
        if n in self.s.helmet_names():
            return "helmet"
        if n in self.s.no_helmet_names():
            return "no_helmet"
        return None

    def detect(self, frame: np.ndarray, track: bool = False) -> list[Detection]:
        """Run every available model on a BGR frame and return canonical detections."""
        dets: list[Detection] = []
        with self.lock:
            if self.coco is not None:
                model = self.coco_tracker if track else self.coco
                res = self._run(model, frame, track=track)
                dets += self._parse(res, lambda k, _n: COCO_LABELS.get(k), "coco")
            if self.rider is not None:
                res = self._run(self.rider, frame)
                dets += self._parse(res, lambda k, _n: "rider_unit", "rider")
            if self.helmet is not None:
                res = self._run(self.helmet, frame)
                dets += self._parse(res, self._helmet_label, "helmet")
        return dets

    def reset_tracking(self) -> None:
        """Forget tracker state before a new video."""
        if self.coco_tracker is not None and getattr(self.coco_tracker, "predictor", None) is not None:
            for t in getattr(self.coco_tracker.predictor, "trackers", []) or []:
                t.reset()
