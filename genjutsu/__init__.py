"""Genjutsu-OSS core library.

A model-independent video *transformation* engine: it decides what in a source
video must stay invariant, what may change, and what must be transferred into
the new generation. Everything here works on numpy arrays so it can be used
from ComfyUI nodes, scripts, or tests without a GPU; model-backed stages load
their weights lazily and only when selected in config.

Array conventions used throughout:
    frames : float32 (T, H, W, 3) in [0, 1], RGB
    masks  : float32 (T, H, W) in [0, 1]
    depth  : float32 (T, H, W), larger = nearer (normalized per video)
    flow   : float32 (T-1, H, W, 2), flow[t] maps frame t -> t+1 in pixels
"""

__version__ = "0.1.0"

from .config import load_config, GenjutsuConfig  # noqa: F401
from .controls import (  # noqa: F401
    ControlPackage,
    Operation,
    RefCategory,
    Reference,
    ReferencePackage,
    TemporalSettings,
    Strengths,
)
