"""YOLO detector backends.

Two backends are supported so the project can use both modern and reference models:

* ``UltralyticsDetector`` – Ultralytics YOLO (YOLO11 / YOLOv8 ``.pt`` files). Used for the
  COCO-pretrained traffic model and for any model trained with ``scripts/train.py``.
* ``DarknetDetector`` – original Darknet YOLOv3 ``.cfg`` + ``.weights`` files run through
  OpenCV DNN. This lets the pretrained helmet / rider models published with the Kaggle
  reference notebook be dropped into ``models/`` and used unchanged.

``ModelManager`` decides which detectors are available from the files present in
``models/`` and exposes them to the analysers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import Settings
from app.detection.types import Box, Detection, canonical_label

log = logging.getLogger(__name__)


class BaseDetector:
    backend = "base"

    def __init__(self, source: str, default_label: Optional[str], min_conf: float):
        self.source = source
        self.default_label = default_label
        self.min_conf = min_conf
        self.class_names: list[str] = []
        self.weights: str = ""
        # Single-purpose models (the rider model) report every box under one label,
        # whatever the dataset happened to call its class.
        self.force_label: Optional[str] = None

    # A raw class name is mapped to a canonical label. Single-purpose models (helmet,
    # rider, plate) fall back to their default label when the class name is unknown,
    # e.g. a Darknet model whose only class is called "obj".
    def _label(self, raw: str) -> Optional[str]:
        if self.force_label:
            return self.force_label
        lab = canonical_label(raw)
        if lab is None and self.default_label is not None:
            return self.default_label
        return lab

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "source": self.source,
            "backend": self.backend,
            "weights": Path(self.weights).name if self.weights else None,
            "classes": self.class_names,
            "canonical_classes": sorted({lab for c in self.class_names if (lab := self._label(c))}),
            "min_confidence": self.min_conf,
        }


class UltralyticsDetector(BaseDetector):
    backend = "ultralytics"

    def __init__(self, weights: str, source: str, min_conf: float, imgsz: int, device: str,
                 default_label: Optional[str] = None, nms_iou: float = 0.5):
        super().__init__(source, default_label, min_conf)
        from ultralytics import YOLO  # imported lazily: heavy import

        self.weights = weights
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device
        self.nms_iou = nms_iou
        names = self.model.names
        self.class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        self._lock = threading.Lock()

    def _convert(self, result) -> list[Detection]:
        out: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, k, tid in zip(xyxy, conf, cls, ids):
            raw = self.class_names[k] if k < len(self.class_names) else str(k)
            label = self._label(raw)
            if label is None:
                continue
            out.append(Detection(label=label, raw_label=raw, confidence=float(c),
                                 box=Box(*map(float, b)), source=self.source,
                                 track_id=int(tid) if tid is not None else None))
        return out

    def detect(self, image: np.ndarray) -> list[Detection]:
        with self._lock:
            res = self.model.predict(image, conf=self.min_conf, iou=self.nms_iou, imgsz=self.imgsz,
                                     device=self.device, verbose=False)[0]
        return self._convert(res)

    def track(self, image: np.ndarray, tracker: str) -> list[Detection]:
        """Detection + multi-object tracking (ByteTrack / BoT-SORT) for video frames.

        Each video job must use its own detector instance because tracker state is kept
        inside the Ultralytics model object (see ModelManager.new_tracking_detector).
        """
        res = self.model.track(image, conf=self.min_conf, iou=self.nms_iou, imgsz=self.imgsz,
                               device=self.device, persist=True, tracker=tracker, verbose=False)[0]
        return self._convert(res)


class DarknetDetector(BaseDetector):
    """Darknet YOLOv3 model executed with OpenCV DNN (as in the Kaggle reference notebook)."""

    backend = "darknet-opencv"

    def __init__(self, cfg: str, weights: str, names: Optional[str], source: str, min_conf: float,
                 default_label: Optional[str], nms_iou: float = 0.4, input_size: int = 416):
        super().__init__(source, default_label, min_conf)
        import cv2

        if not hasattr(cv2.dnn, "readNetFromDarknet"):
            raise RuntimeError("This OpenCV build has no Darknet importer; install opencv-python<5")
        self.cv2 = cv2
        self.weights = weights
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.out_layers = self.net.getUnconnectedOutLayersNames()
        self.nms_iou = nms_iou
        self.input_size = self._input_size_from_cfg(cfg) or input_size
        if names and Path(names).exists():
            self.class_names = [ln.strip() for ln in Path(names).read_text().splitlines() if ln.strip()]
        else:
            self.class_names = [default_label or "object"]
        self._lock = threading.Lock()

    @staticmethod
    def _input_size_from_cfg(cfg: str) -> Optional[int]:
        try:
            for line in Path(cfg).read_text().splitlines():
                line = line.strip().replace(" ", "")
                if line.startswith("width="):
                    return int(line.split("=")[1])
        except (OSError, ValueError):
            pass
        return None

    def detect(self, image: np.ndarray) -> list[Detection]:
        cv2 = self.cv2
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        with self._lock:
            self.net.setInput(blob)
            outputs = self.net.forward(self.out_layers)
        boxes, confs, classes = [], [], []
        for output in outputs:
            for row in output.reshape(-1, output.shape[-1]):
                scores = row[5:]
                if scores.size == 0:
                    continue
                k = int(np.argmax(scores))
                # OpenCV's YOLO region layer already multiplies class scores by objectness.
                score = float(scores[k])
                if score < self.min_conf:
                    continue
                cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confs.append(score)
                classes.append(k)
        out: list[Detection] = []
        if not boxes:
            return out
        keep = cv2.dnn.NMSBoxes(boxes, confs, self.min_conf, self.nms_iou)
        for i in np.array(keep).flatten():
            x, y, bw, bh = boxes[i]
            raw = self.class_names[classes[i]] if classes[i] < len(self.class_names) else str(classes[i])
            label = self._label(raw)
            if label is None:
                continue
            out.append(Detection(label=label, raw_label=raw, confidence=confs[i],
                                 box=Box(float(max(0, x)), float(max(0, y)), float(min(w, x + bw)), float(min(h, y + bh))),
                                 source=self.source))
        return out


def deduplicate(detections: list[Detection], iou_thr: float) -> list[Detection]:
    """Remove duplicate boxes (same object reported twice, e.g. by two models or as
    both ``motorcycle`` and ``bicycle``). Keeps the most confident box."""
    groups = {"motorcycle": "two_wheel", "bicycle": "two_wheel", "car": "four_wheel",
              "truck": "four_wheel", "bus": "four_wheel"}
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        g = groups.get(det.label, det.label)
        if any(groups.get(k.label, k.label) == g and k.box.iou(det.box) > iou_thr for k in kept):
            continue
        kept.append(det)
    return kept


class ModelManager:
    """Loads whichever detectors are present in ``models/`` and reports their status."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self._loaded = False
        self.vehicle: Optional[UltralyticsDetector] = None
        self.helmet: Optional[BaseDetector] = None
        self.rider: Optional[BaseDetector] = None
        self.plate: Optional[BaseDetector] = None
        self.errors: dict[str, str] = {}

    # ------------------------------------------------------------------ loading
    def _path(self, name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else self.s.models_dir / p

    def _vehicle_weights(self) -> str:
        p = self._path(self.s.vehicle_model)
        # Ultralytics downloads official weights (yolo11n.pt etc.) to the given path if missing.
        return str(p)

    def _load_optional(self, source: str, pt_name: str, cfg: str, weights: str, names: str,
                       min_conf: float, default_label: str) -> Optional[BaseDetector]:
        pt = self._path(pt_name) if pt_name else None
        try:
            if pt is not None and pt.is_file():
                return UltralyticsDetector(str(pt), source, min_conf, self.s.inference_imgsz, self.s.device,
                                           default_label=default_label, nms_iou=self.s.nms_iou)
            if not (cfg and weights):
                return None
            cfg_p, w_p = self._path(cfg), self._path(weights)
            if cfg_p.is_file() and w_p.is_file():
                return DarknetDetector(str(cfg_p), str(w_p), str(self._path(names)), source, min_conf,
                                       default_label=default_label)
        except Exception as exc:  # a broken optional model must not take the app down
            log.exception("Failed to load %s detector", source)
            self.errors[source] = str(exc)
        return None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            s = self.s
            base_conf = min(s.min_conf_vehicle, s.min_conf_person, s.min_conf_traffic_light)
            try:
                self.vehicle = UltralyticsDetector(self._vehicle_weights(), "vehicle", base_conf,
                                                   s.inference_imgsz, s.device, nms_iou=s.nms_iou)
            except Exception as exc:
                log.exception("Failed to load the vehicle detector")
                self.errors["vehicle"] = str(exc)
            self.helmet = self._load_optional("helmet", s.helmet_model, s.helmet_darknet_cfg,
                                              s.helmet_darknet_weights, s.helmet_darknet_names,
                                              s.min_conf_helmet, "helmet")
            self.rider = self._load_optional("rider", s.rider_model, s.rider_darknet_cfg,
                                             s.rider_darknet_weights, s.rider_darknet_names,
                                             s.min_conf_rider, "rider")
            if self.rider is not None:
                self.rider.force_label = "rider"
            self.plate = self._load_optional("plate", s.plate_model, "", "", "", 0.35, "number_plate")
            self._loaded = True
            log.info("Detectors loaded: %s", {k: v["available"] for k, v in self.status().items()})

    def reload(self) -> None:
        with self._lock:
            self._loaded = False
            self.vehicle = self.helmet = self.rider = self.plate = None
            self.errors = {}
        self.load()

    def new_tracking_detector(self) -> UltralyticsDetector:
        """Fresh vehicle-model instance with its own tracker state (one per video job)."""
        s = self.s
        base_conf = min(s.min_conf_vehicle, s.min_conf_person, s.min_conf_traffic_light)
        return UltralyticsDetector(self._vehicle_weights(), "vehicle", base_conf, s.inference_imgsz,
                                   s.device, nms_iou=s.nms_iou)

    # ------------------------------------------------------------------ inference
    def filter_by_confidence(self, dets: list[Detection]) -> list[Detection]:
        s = self.s
        out = []
        for d in dets:
            if d.label in {"motorcycle", "bicycle", "car", "bus", "truck"} and d.confidence < s.min_conf_vehicle:
                continue
            if d.label == "person" and d.confidence < s.min_conf_person:
                continue
            if d.label == "traffic_light" and d.confidence < s.min_conf_traffic_light:
                continue
            out.append(d)
        return out

    def detect_auxiliary(self, image: np.ndarray) -> list[Detection]:
        """Run the optional helmet / rider / plate detectors on a frame."""
        dets: list[Detection] = []
        for det in (self.helmet, self.rider, self.plate):
            if det is not None:
                dets.extend(det.detect(image))
        return dets

    def detect_all(self, image: np.ndarray) -> list[Detection]:
        self.load()
        if self.vehicle is None:
            raise RuntimeError(f"Vehicle detector unavailable: {self.errors.get('vehicle', 'unknown error')}")
        dets = self.filter_by_confidence(self.vehicle.detect(image))
        dets.extend(self.detect_auxiliary(image))
        return deduplicate(dets, self.s.duplicate_iou)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        out = {}
        for name in ("vehicle", "helmet", "rider", "plate"):
            det = getattr(self, name)
            entry = {"available": det is not None}
            if det is not None:
                entry.update(det.info())
            elif name in self.errors:
                entry["error"] = self.errors[name]
            out[name] = entry
        return out

    @property
    def helmet_available(self) -> bool:
        return self.helmet is not None and any(
            lab in {"helmet", "no_helmet"} for c in self.helmet.class_names if (lab := self.helmet._label(c)))
