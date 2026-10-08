import numpy as np
import pytest

from app.detection import association
from app.detection.preprocessing import InvalidInputError, check_extension, decode_image
from app.detection.tracker import IoUTracker
from app.detection.types import Detection
from app.detection.yolo_detector import COCO_LABELS, YoloDetector


class _T:  # minimal torch-like tensor
    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    xyxy = _T([[10, 10, 50, 80], [0, 0, 5, 5], [100, 100, 200, 220]])
    conf = _T([0.9, 0.5, 0.7])
    cls = _T([3, 4, 2])  # motorcycle, airplane (ignored), car
    id = _T([5, 6, 7])

    def __len__(self):
        return 3


class _Result:
    boxes = _Boxes()
    names = {}


def test_detection_parsing_keeps_only_traffic_classes_and_track_ids():
    dets = YoloDetector._parse(_Result(), lambda k, _n: COCO_LABELS.get(k), "coco")
    assert [d.label for d in dets] == ["motorcycle", "car"]
    assert dets[0].track_id == 5 and dets[0].box == (10.0, 10.0, 50.0, 80.0)


def test_empty_result_parses_to_nothing():
    class R:
        boxes = None
        names = {}

    assert YoloDetector._parse(R(), lambda k, n: "car", "coco") == []


def test_duplicate_filtering_keeps_best_box():
    a = Detection("car", 0.9, (0, 0, 100, 100))
    b = Detection("car", 0.6, (2, 2, 101, 99))
    c = Detection("person", 0.8, (2, 2, 101, 99))
    kept = association.dedupe([a, b, c])
    assert len(kept) == 2 and a in kept and b not in kept


def test_iou_tracker_keeps_ids_stable():
    t = IoUTracker()
    i1 = t.update([(0, 0, 50, 50)], 0)
    i2 = t.update([(4, 2, 54, 52)], 1)
    i3 = t.update([(300, 300, 350, 350)], 2)
    assert i1 == i2 and i3 != i1


def test_invalid_images_are_rejected_with_friendly_errors():
    with pytest.raises(InvalidInputError):
        decode_image(b"")
    with pytest.raises(InvalidInputError):
        decode_image(b"this is not an image")
    with pytest.raises(InvalidInputError):
        check_extension("notes.gif", {".jpg", ".png"}, "image")
