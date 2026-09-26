"""Human pose: DWPose (whole-body, via rtmlib/ONNX) rendered as OpenPose-style
skeleton frames for conditioning, plus face crops for Wan-Animate."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .video import to_uint8

log = logging.getLogger("genjutsu.pose")

# rtmlib to_openpose=True layout (134 keypoints)
BODY = slice(0, 18)
FEET = slice(18, 24)
FACE = slice(24, 92)
LHAND = slice(92, 113)
RHAND = slice(113, 134)


@dataclass
class PoseResult:
    rendered: np.ndarray  # (T,H,W,3) skeleton on black, [0,1]
    keypoints: list  # per frame: (N_people, 134, 2)
    scores: list  # per frame: (N_people, 134)
    face_crops: Optional[np.ndarray] = None  # (T,S,S,3)
    face_boxes: Optional[list] = None


class DWPose:
    def __init__(self, mode: str = "performance", device: str = "cuda", backend: str = "onnxruntime"):
        from rtmlib import Wholebody

        try:
            self.model = Wholebody(to_openpose=True, mode=mode, backend=backend, device=device)
        except Exception:
            self.model = Wholebody(to_openpose=True, mode=mode, backend=backend, device="cpu")

    def __call__(self, frame_rgb: np.ndarray):
        kp, sc = self.model(cv2.cvtColor(to_uint8(frame_rgb), cv2.COLOR_RGB2BGR))
        return np.asarray(kp, np.float32), np.asarray(sc, np.float32)


def render_skeleton(shape_hw, keypoints, scores, kpt_thr: float = 0.3, line_width: int = 0) -> np.ndarray:
    from rtmlib import draw_skeleton

    H, W = shape_hw
    canvas = np.zeros((H, W, 3), np.uint8)
    if len(keypoints):
        lw = line_width or max(2, int(round(min(H, W) / 180)))
        canvas = draw_skeleton(canvas, keypoints, scores, openpose_skeleton=True, kpt_thr=kpt_thr, radius=lw, line_width=lw)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _select_person(kps: np.ndarray, scs: np.ndarray, prev_center: Optional[np.ndarray]) -> int:
    """Keep the same person across frames: nearest body center to previous,
    else the most confident / largest."""
    if len(kps) == 1 or prev_center is None:
        return int(np.argmax(scs[:, BODY].mean(1))) if len(kps) else -1
    centers = np.array([k[BODY][s[BODY] > 0.3].mean(0) if (s[BODY] > 0.3).any() else k[BODY].mean(0) for k, s in zip(kps, scs)])
    return int(np.argmin(((centers - prev_center) ** 2).sum(1)))


def face_crops(frames: np.ndarray, keypoints: list, scores: list, size: int = 512, expand: float = 1.6, thr: float = 0.3):
    T, H, W = frames.shape[:3]
    crops = np.zeros((T, size, size, 3), np.float32)
    boxes: list = [None] * T
    last = None
    for t in range(T):
        box = None
        if len(keypoints[t]):
            f = keypoints[t][0][FACE]
            s = scores[t][0][FACE]
            good = f[s > thr]
            if len(good) >= 8:
                (x0, y0), (x1, y1) = good.min(0), good.max(0)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                half = max(x1 - x0, y1 - y0) * expand / 2
                box = [cx - half, cy - half, cx + half, cy + half]
        if box is None:
            box = last
        if box is None:
            continue
        if last is not None:  # light temporal smoothing of the crop window
            box = [0.7 * b + 0.3 * lb for b, lb in zip(box, last)]
        last = box
        boxes[t] = box
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        pad = max(0, -x0, -y0, x1 - W, y1 - H)
        src = cv2.copyMakeBorder(frames[t], pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else frames[t]
        crop = src[y0 + pad : y1 + pad, x0 + pad : x1 + pad]
        if crop.size:
            crops[t] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    # back-fill frames before the first detection
    first = next((t for t in range(T) if boxes[t] is not None), None)
    if first is not None:
        crops[:first] = crops[first]
    return crops, boxes


def extract_pose(frames: np.ndarray, device: str = "cuda", with_faces: bool = True, face_size: int = 512, single_person: bool = True) -> PoseResult:
    model = DWPose(device=device)
    T, H, W = frames.shape[:3]
    rendered = np.zeros((T, H, W, 3), np.float32)
    kps_all, scs_all = [], []
    prev_center = None
    for t in range(T):
        kps, scs = model(frames[t])
        if single_person and len(kps):
            i = _select_person(kps, scs, prev_center)
            kps, scs = kps[i : i + 1], scs[i : i + 1]
            ok = scs[0][BODY] > 0.3
            if ok.any():
                prev_center = kps[0][BODY][ok].mean(0)
        kps_all.append(kps)
        scs_all.append(scs)
        rendered[t] = render_skeleton((H, W), kps, scs)
    fc, fb = (face_crops(frames, kps_all, scs_all, face_size) if with_faces else (None, None))
    return PoseResult(rendered, kps_all, scs_all, fc, fb)


def smooth_keypoints(keypoints: list, scores: list, alpha: float = 0.5) -> list:
    """One-euro-style EMA on the tracked person's keypoints to remove jitter."""
    out = []
    prev = None
    for kp, sc in zip(keypoints, scores):
        if not len(kp):
            out.append(kp)
            continue
        cur = kp.copy()
        if prev is not None and prev.shape == cur.shape:
            conf = (sc > 0.3)[..., None]
            cur = np.where(conf, alpha * prev + (1 - alpha) * cur, cur)
        prev = cur
        out.append(cur)
    return out
