"""Identity lock: face embeddings (InsightFace) measured across the video,
with per-frame similarity to the reference and low-score frame detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .video import to_uint8

log = logging.getLogger("genjutsu.identity")

_FA = None


def _face_app(device: str = "cuda"):
    global _FA
    if _FA is None:
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        _FA = FaceAnalysis(name="buffalo_l", providers=providers)
        _FA.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return _FA


def face_embedding(image: np.ndarray, device: str = "cuda") -> Optional[np.ndarray]:
    """Normalized 512-d ArcFace embedding of the largest face, or None."""
    faces = _face_app(device).get(cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR))
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    e = f.normed_embedding.astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-8)


@dataclass
class IdentityReport:
    scores: list  # per-frame cosine similarity (None where no face)
    mean: float
    below_threshold: list = field(default_factory=list)
    threshold: float = 0.45

    def as_dict(self):
        return {"mean": self.mean, "threshold": self.threshold, "below_threshold": self.below_threshold, "scores": self.scores}


def identity_scores(frames: np.ndarray, reference_embedding: np.ndarray, threshold: float = 0.45, stride: int = 1, device: str = "cuda") -> IdentityReport:
    scores: list = [None] * len(frames)
    for t in range(0, len(frames), max(1, stride)):
        e = face_embedding(frames[t], device)
        if e is not None:
            scores[t] = float(np.dot(e, reference_embedding))
    vals = [s for s in scores if s is not None]
    below = [t for t, s in enumerate(scores) if s is not None and s < threshold]
    return IdentityReport(scores, float(np.mean(vals)) if vals else float("nan"), below, threshold)


def segments_to_regenerate(below: list, total: int, pad: int = 4, min_gap: int = 8) -> list[tuple[int, int]]:
    """Group flagged frames into padded [start,end) ranges for re-generation."""
    if not below:
        return []
    below = sorted(below)
    segs = [[below[0], below[0] + 1]]
    for t in below[1:]:
        if t - segs[-1][1] <= min_gap:
            segs[-1][1] = t + 1
        else:
            segs.append([t, t + 1])
    return [(max(0, a - pad), min(total, b + pad)) for a, b in segs]
