"""GenjutsuScore: objective comparison of transformations / backends
(section 40). Each metric is in [0,1], higher is better; metrics whose
model is unavailable are reported as None rather than guessed."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from .motion import DISFlow, compute_flow
from .temporal import flicker_score


def ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Gray SSIM (Wang et al.), optionally averaged over a mask."""
    ga = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2GRAY)
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = cv2.GaussianBlur(ga, (11, 11), 1.5), cv2.GaussianBlur(gb, (11, 11), 1.5)
    saa = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a**2
    sbb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b**2
    sab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (saa + sbb + C2))
    if mask is not None:
        w = mask.astype(np.float32)
        return float((m * w).sum() / max(w.sum(), 1.0))
    return float(m.mean())


def motion_similarity(src_flow: np.ndarray, out_flow: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """exp(-EPE / mean magnitude): 1 = output moves exactly like the source."""
    n = min(len(src_flow), len(out_flow))
    if n == 0:
        return 1.0
    epe = np.linalg.norm(src_flow[:n] - out_flow[:n], axis=-1)
    mag = np.linalg.norm(src_flow[:n], axis=-1)
    if mask is not None:
        w = mask[1 : n + 1]
        e, m = (epe * w).sum() / max(w.sum(), 1), (mag * w).sum() / max(w.sum(), 1)
    else:
        e, m = epe.mean(), mag.mean()
    return float(np.exp(-e / (m + 1.0)))


@dataclass
class GenjutsuScore:
    identity: Optional[float] = None
    motion: Optional[float] = None
    camera: Optional[float] = None
    background: Optional[float] = None
    object: Optional[float] = None
    temporal: Optional[float] = None
    prompt_alignment: Optional[float] = None

    def overall(self) -> Optional[float]:
        vals = [v for v in asdict(self).values() if v is not None]
        return float(np.mean(vals)) if vals else None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall()
        return d

    def table(self) -> str:
        lines = ["GenjutsuScore"]
        for k, v in self.as_dict().items():
            lines.append(f"    {k.replace('_', ' ').title():<18}{'n/a' if v is None else f'{v:.3f}'}")
        return "\n".join(lines)


def evaluate(
    source: np.ndarray,
    output: np.ndarray,
    preserve_mask: Optional[np.ndarray] = None,
    edit_mask: Optional[np.ndarray] = None,
    reference_image: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    identity_embedding: Optional[np.ndarray] = None,
    max_side: int = 480,
    sample_stride: int = 1,
    object_mask: Optional[np.ndarray] = None,
) -> GenjutsuScore:
    from .video import resize_frames, resize_masks

    n = min(len(source), len(output))
    source, output = source[:n:sample_stride], output[:n:sample_stride]
    H, W = source.shape[1:3]
    s = min(1.0, max_side / max(H, W))
    w, h = int(W * s), int(H * s)
    src, out = resize_frames(source, w, h), resize_frames(output, w, h)
    pm = resize_masks(preserve_mask[:n:sample_stride], w, h) if preserve_mask is not None else None
    em = resize_masks(edit_mask[:n:sample_stride], w, h) if edit_mask is not None else None
    om = resize_masks(object_mask[:n:sample_stride], w, h) if object_mask is not None else em

    est = DISFlow()
    sf, sb = compute_flow(src, est)
    of, ob = compute_flow(out, est)
    score = GenjutsuScore()
    score.motion = motion_similarity(sf, of)
    score.temporal = min(1.0, flicker_score(out, of, ob) / max(flicker_score(src, sf, sb), 1e-6))

    from .camera import camera_similarity, estimate_camera

    score.camera = camera_similarity(estimate_camera(src, em), estimate_camera(out, em))

    if pm is not None:
        score.background = float(np.mean([ssim(a, b, m) for a, b, m in zip(src, out, pm)]))

    if reference_image is not None and om is not None:
        from .references import color_signature

        ref_sig = color_signature(reference_image, reference_mask)
        sims = []
        for f, m in zip(out, om):
            if (m > 0.5).sum() < 16:
                continue
            sig = color_signature(f, m)
            sims.append(float(np.minimum(sig, ref_sig).sum()))  # histogram intersection
        score.object = float(np.mean(sims)) if sims else None

    if identity_embedding is not None:
        try:
            from .identity import identity_scores

            rep = identity_scores(output, identity_embedding, stride=max(1, len(output) // 16))
            score.identity = None if np.isnan(rep.mean) else float(np.clip(rep.mean, 0, 1))
        except Exception:
            score.identity = None
    return score
