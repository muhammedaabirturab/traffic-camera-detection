import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.detection.types import Box, Detection  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    s.ensure_dirs()
    return s


def det(label: str, box, conf: float = 0.9, track_id=None, source: str = "vehicle") -> Detection:
    return Detection(label=label, raw_label=label, confidence=conf, box=Box(*box), source=source, track_id=track_id)


@pytest.fixture
def tiny_darknet(tmp_path) -> Path:
    """A minimal valid Darknet YOLO model (1 conv + 1 yolo layer, class 'Helmet')."""
    d = tmp_path / "dn"
    d.mkdir()
    (d / "t.cfg").write_text(
        "[net]\nwidth=64\nheight=64\nchannels=3\n\n"
        "[convolutional]\nsize=1\nstride=1\npad=0\nfilters=18\nactivation=linear\n\n"
        "[yolo]\nmask=0,1,2\nanchors=10,13,16,30,33,23\nclasses=1\nnum=3\n"
    )
    with open(d / "t.weights", "wb") as f:
        f.write(struct.pack("<iii", 0, 2, 0))
        f.write(struct.pack("<q", 0))
        bias = np.zeros(18, np.float32)
        bias[4::6] = 5.0  # objectness
        bias[5::6] = 5.0  # class score
        f.write(bias.tobytes())
        f.write(np.zeros(18 * 3, np.float32).tobytes())
    (d / "t.names").write_text("Helmet\n")
    return d
