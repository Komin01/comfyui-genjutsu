from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..config import GenjutsuConfig
from ..controls import ControlPackage, Operation

_REGISTRY: dict[str, type] = {}


def register_backend(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


_INSTANCES: dict[str, "VideoBackend"] = {}


def get_backend(name: str, config: Optional[GenjutsuConfig] = None, reuse: bool = True) -> "VideoBackend":
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend '{name}'. Available: {available_backends()}")
    if reuse and name in _INSTANCES:
        return _INSTANCES[name]
    inst = _REGISTRY[name](config or GenjutsuConfig())
    if reuse:
        _INSTANCES[name] = inst
    return inst


REGION_OPS = {Operation.OBJECT_SWAP, Operation.PRODUCT_SWAP, Operation.OUTFIT_SWAP}
CHARACTER_OPS = {Operation.CHARACTER_SWAP, Operation.MOTION_TRANSFER}
RECAST_OPS = {Operation.ENVIRONMENT_RECAST, Operation.STYLE_RECAST, Operation.FULL_RECAST}


class VideoBackend(abc.ABC):
    """Contract every generator implements. All methods return float32
    frames (T,H,W,3) in [0,1] at the ControlPackage's resolution."""

    name = "base"
    #: operations this backend can serve; the pipeline checks before running
    supports: set = set()
    #: True if the backend handles long clips internally (no outer chunking)
    internal_chunking: bool = False

    def __init__(self, config: GenjutsuConfig):
        self.config = config

    # dispatch --------------------------------------------------------------
    def generate(self, cp: ControlPackage) -> np.ndarray:
        op = Operation(cp.operation)
        if op not in self.supports:
            raise NotImplementedError(f"Backend '{self.name}' does not support {op.value}")
        if op in REGION_OPS:
            return self.generate_region(cp)
        if op in CHARACTER_OPS:
            return self.generate_character(cp)
        return self.generate_recast(cp)

    # capabilities ------------------------------------------------------------
    def generate_t2v(self, prompt: str, negative_prompt: str, width: int, height: int, num_frames: int, seed: int = 0, **kw) -> np.ndarray:
        raise NotImplementedError

    def generate_i2v(self, image: np.ndarray, prompt: str, negative_prompt: str, num_frames: int, seed: int = 0, **kw) -> np.ndarray:
        raise NotImplementedError

    def generate_character(self, cp: ControlPackage) -> np.ndarray:
        raise NotImplementedError

    def generate_region(self, cp: ControlPackage) -> np.ndarray:
        raise NotImplementedError

    def generate_recast(self, cp: ControlPackage) -> np.ndarray:
        raise NotImplementedError

    def chunks_internally(self, op) -> bool:
        return self.internal_chunking

    def unload(self) -> None:
        """Free GPU memory."""

    def describe(self) -> dict:
        return {"name": self.name, "supports": sorted(o.value for o in self.supports), "internal_chunking": self.internal_chunking}
