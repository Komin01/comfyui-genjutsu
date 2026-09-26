"""ControlPackage and ReferencePackage: the model-independent contract between
video understanding and generation. Backends consume these; nothing upstream
knows which backend will run."""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


class Operation(str, enum.Enum):
    MOTION_TRANSFER = "motion_transfer"
    CHARACTER_SWAP = "character_swap"
    OBJECT_SWAP = "object_swap"
    PRODUCT_SWAP = "product_swap"
    OUTFIT_SWAP = "outfit_swap"
    ENVIRONMENT_RECAST = "environment_recast"
    STYLE_RECAST = "style_recast"
    FULL_RECAST = "full_recast"


class RefCategory(str, enum.Enum):
    CHARACTER = "character"
    PRODUCT = "product"
    OBJECT = "object"
    OUTFIT = "outfit"
    LOCATION = "location"
    STYLE = "style"
    PROP = "prop"
    TEXTURE = "texture"
    LOGO = "logo"


# Which invariants each operation holds fixed by default (section 44).
DEFAULT_INVARIANTS: dict[Operation, list[str]] = {
    Operation.MOTION_TRANSFER: ["motion", "timing", "camera"],
    Operation.CHARACTER_SWAP: ["motion", "timing", "camera", "lighting", "background"],
    Operation.OBJECT_SWAP: ["motion", "timing", "camera", "lighting", "background", "identity"],
    Operation.PRODUCT_SWAP: ["motion", "timing", "camera", "lighting", "background", "identity"],
    Operation.OUTFIT_SWAP: ["motion", "timing", "camera", "lighting", "background", "identity"],
    Operation.ENVIRONMENT_RECAST: ["motion", "timing", "camera", "identity"],
    Operation.STYLE_RECAST: ["motion", "timing", "camera", "composition"],
    Operation.FULL_RECAST: ["motion", "timing", "camera"],
}


@dataclass
class Reference:
    image: np.ndarray  # (H, W, 3) float32 [0,1]
    category: RefCategory
    weight: float = 1.0
    name: str = ""
    mask: Optional[np.ndarray] = None  # (H, W) optional subject cut-out
    embedding: Optional[np.ndarray] = None  # identity / appearance embedding
    metadata: dict = field(default_factory=dict)


@dataclass
class ReferencePackage:
    references: list[Reference] = field(default_factory=list)

    def add(self, ref: Reference) -> "ReferencePackage":
        return ReferencePackage(self.references + [ref])

    def by_category(self, *cats: RefCategory) -> list[Reference]:
        return [r for r in self.references if r.category in cats]

    def primary(self, *cats: RefCategory) -> Optional[Reference]:
        refs = sorted(self.by_category(*cats), key=lambda r: -r.weight)
        return refs[0] if refs else None

    def __len__(self) -> int:
        return len(self.references)


@dataclass
class TemporalSettings:
    enabled: bool = True
    strength: float = 0.75
    smoothing_strength: float = 0.3
    repair_strength: float = 0.6
    chunk_frames: int = 81
    overlap_frames: int = 9


@dataclass
class Strengths:
    identity: float = 0.85
    motion: float = 1.0
    reference: float = 0.75
    preservation: float = 1.0
    style: float = 0.5


@dataclass
class ControlPackage:
    frames: np.ndarray  # (T,H,W,3)
    fps: float
    operation: Operation = Operation.OBJECT_SWAP
    source_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    edit_mask: Optional[np.ndarray] = None  # (T,H,W) what may change
    preserve_mask: Optional[np.ndarray] = None  # (T,H,W) what must not change
    soft_mask: Optional[np.ndarray] = None  # (T,H,W) blend weight for generated
    subject_mask: Optional[np.ndarray] = None
    pose_frames: Optional[np.ndarray] = None  # (T,H,W,3) rendered skeleton
    pose_keypoints: Optional[list] = None
    face_frames: Optional[np.ndarray] = None  # (T,h,w,3) face crops (Wan Animate)
    depth: Optional[np.ndarray] = None
    flow_fw: Optional[np.ndarray] = None
    flow_bw: Optional[np.ndarray] = None
    camera: Optional[dict] = None
    scene_cuts: list[int] = field(default_factory=list)
    references: ReferencePackage = field(default_factory=ReferencePackage)
    invariants: list[str] = field(default_factory=list)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    strengths: Strengths = field(default_factory=Strengths)
    seed: int = 0
    steps: int = 30
    guidance: float = 5.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.invariants:
            self.invariants = list(DEFAULT_INVARIANTS.get(Operation(self.operation), []))

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        return int(self.frames.shape[2]), int(self.frames.shape[1])  # (W, H)

    def slice(self, start: int, end: int) -> "ControlPackage":
        """Temporal chunk view for chunked generation."""

        def s(a, off=0):
            if a is None:
                return None
            return a[start : max(start, end - off)]

        import copy

        cp = copy.copy(self)
        cp.frames = self.frames[start:end]
        for name in ("edit_mask", "preserve_mask", "soft_mask", "subject_mask", "pose_frames", "face_frames", "depth"):
            setattr(cp, name, s(getattr(self, name)))
        cp.flow_fw = s(self.flow_fw, 1)
        cp.flow_bw = s(self.flow_bw, 1)
        cp.scene_cuts = [c - start for c in self.scene_cuts if start <= c < end]
        return cp

    def describe(self) -> dict:
        """JSON-serializable summary (arrays reduced to shapes)."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            elif isinstance(v, ReferencePackage):
                out[k] = [
                    {"name": r.name, "category": r.category.value, "weight": r.weight, "shape": list(r.image.shape)}
                    for r in v.references
                ]
            elif isinstance(v, (TemporalSettings, Strengths)):
                out[k] = asdict(v)
            elif isinstance(v, enum.Enum):
                out[k] = v.value
            elif k == "pose_keypoints":
                out[k] = None if v is None else f"<{len(v)} frames>"
            else:
                out[k] = v
        return json.loads(json.dumps(out, default=str))
