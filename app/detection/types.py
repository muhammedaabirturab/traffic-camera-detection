"""Shared data structures for the detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Canonical labels used everywhere after detection. Detector-specific class
# names (COCO "motorcycle", Darknet "Helmet", a custom "Without Helmet" ...)
# are normalised to these via LABEL_ALIASES.
VEHICLE_LABELS = {"motorcycle", "bicycle", "car", "bus", "truck"}
TWO_WHEELER_LABELS = {"motorcycle"}

LABEL_ALIASES = {
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motor bike": "motorcycle",
    "scooter": "motorcycle", "two-wheeler": "motorcycle", "two_wheeler": "motorcycle",
    "bicycle": "bicycle", "bike": "bicycle",
    "car": "car", "bus": "bus", "truck": "truck", "lorry": "truck",
    "person": "person", "pedestrian": "person",
    "traffic light": "traffic_light", "traffic_light": "traffic_light",
    "helmet": "helmet", "with helmet": "helmet", "with_helmet": "helmet", "withhelmet": "helmet",
    "without helmet": "no_helmet", "without_helmet": "no_helmet", "no helmet": "no_helmet",
    "no_helmet": "no_helmet", "nohelmet": "no_helmet", "head": "no_helmet",
    "rider": "rider", "person on bike": "rider", "person_on_bike": "rider",
    "motorcyclist": "rider", "person on motorbike": "rider",
    "license plate": "number_plate", "license_plate": "number_plate", "licence plate": "number_plate",
    "number plate": "number_plate", "number_plate": "number_plate", "plate": "number_plate",
}


def canonical_label(raw: str) -> Optional[str]:
    """Map a detector class name to a canonical label, or None if irrelevant."""
    return LABEL_ALIASES.get(raw.strip().lower())


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def intersection(self, other: "Box") -> float:
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, iw) * max(0.0, ih)

    def iou(self, other: "Box") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x1, other.x1), min(self.y1, other.y1), max(self.x2, other.x2), max(self.y2, other.y2))

    def expand(self, fx: float, fy: Optional[float] = None) -> "Box":
        fy = fx if fy is None else fy
        dx, dy = self.w * fx, self.h * fy
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, width: int, height: int) -> "Box":
        return Box(max(0, self.x1), max(0, self.y1), min(width, self.x2), min(height, self.y2))

    def as_list(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]


@dataclass
class Detection:
    """A single object found by a YOLO model, after label normalisation."""

    label: str  # canonical label
    raw_label: str  # class name as produced by the model
    confidence: float
    box: Box
    source: str  # which detector produced it: "vehicle", "helmet", "rider", "plate"
    track_id: Optional[int] = None
    det_id: int = -1  # index within the frame, assigned by the pipeline

    def to_dict(self) -> dict:
        return {
            "id": self.det_id,
            "label": self.label,
            "raw_label": self.raw_label,
            "confidence": round(self.confidence, 4),
            "bbox": self.box.as_list(),
            "source": self.source,
            "track_id": self.track_id,
        }


@dataclass
class RiderAssessment:
    """Outcome of the helmet check for one person associated with a two-wheeler."""

    person: Detection
    association: float
    helmet: Optional[Detection] = None
    no_helmet: Optional[Detection] = None
    status: str = "unknown"  # "helmet", "no_helmet", "insufficient_evidence", "not_evaluated"
    reason: str = ""
    max_nearby_helmet_conf: float = 0.0


@dataclass
class TwoWheelerUnit:
    """A motorcycle together with the people judged to be riding it."""

    vehicle: Detection
    riders: list[RiderAssessment] = field(default_factory=list)
    supported_by_rider_model: bool = False

    @property
    def box(self) -> Box:
        b = self.vehicle.box
        for r in self.riders:
            b = b.union(r.person.box)
        return b


@dataclass
class ViolationCandidate:
    """Output of relationship analysis, before the rule engine attaches legal context."""

    rule_id: str
    confidence: float
    vehicle: Detection
    subjects: list[Detection]
    evidence: str  # short machine-generated statement of what was (not) seen
    region: Box
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("vehicle"), d.pop("subjects"), d.pop("region")
        return d


@dataclass
class Observation:
    """Something the system looked at but could not decide on (insufficient evidence)."""

    rule_id: str
    message: str
    region: Optional[Box] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "bbox": self.region.as_list() if self.region else None,
            "confidence": round(self.confidence, 4),
        }
