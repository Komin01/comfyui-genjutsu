import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synth import make_frames, reference_image  # noqa: E402


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


@pytest.fixture
def clip():
    frames, masks, boxes = make_frames(T=24, H=96, W=128)
    return frames, masks, boxes


@pytest.fixture
def cfg(tmp_path):
    from genjutsu.config import GenjutsuConfig

    return GenjutsuConfig().with_overrides({
        "backend": "classical",
        "cache": {"root": str(tmp_path / "cache")},
        "segmentation": {"model": "grabcut"},
        "optical_flow": {"model": "dis"},
    })


@pytest.fixture
def product_ref():
    from genjutsu.references import build_reference

    return build_reference(reference_image(), "product", 1.0, "disc")
