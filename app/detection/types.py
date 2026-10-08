"""Plain data structures shared by the detection, rule and analysis layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

VEHICLE_LABELS = {"motorcycle", "bicycle", "car", "bus", "truck"}


@dataclass
class Detection:
    label: str  # canonical: person, motorcycle, bicycle, car, bus, truck, rider_unit, helmet, no_helmet
    confidence: float
    box: Box
    source: str = "coco"  # coco | rider | helmet
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    def to_dict(self, img_w: int, img_h: int) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "box_norm": [round(x1 / img_w, 5), round(y1 / img_h, 5), round(x2 / img_w, 5), round(y2 / img_h, 5)],
            "source": self.source,
            "track_id": self.track_id,
        }


@dataclass
class RiderLink:
    person: Detection
    score: float  # 0..1 association score with a two-wheeler
    strong: bool
    helmet_status: str = "not_assessed"  # helmet | no_helmet | no_helmet_inferred | unknown | not_assessed
    helmet_confidence: float = 0.0
    role: str = "rider"  # rider | pillion (front-most / largest person is the driver)


@dataclass
class TwoWheelerUnit:
    unit_id: int
    bike_box: Box
    bike_conf: float
    kind: str = "motorcycle"  # motorcycle | bicycle
    origin: str = "coco"  # coco | rider_model | both
    rider_box: Optional[Box] = None  # box from the rider model (person + bike), if matched
    track_id: Optional[int] = None
    riders: list[RiderLink] = field(default_factory=list)  # strongly associated persons
    weak_links: list[RiderLink] = field(default_factory=list)  # ambiguous persons (not counted)

    @property
    def box(self) -> Box:
        """Box covering the bike and its riders (used for drawing / evidence crops)."""
        boxes = [self.bike_box] + ([self.rider_box] if self.rider_box else []) + [r.person.box for r in self.riders]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class Finding:
    """One rule evaluation outcome for one two-wheeler (before rule-database enrichment)."""
    violation_id: str
    confidence: float
    unit: TwoWheelerUnit
    evidence: str
    status: str = "possible"  # possible | insufficient_evidence
    people: list[Detection] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
