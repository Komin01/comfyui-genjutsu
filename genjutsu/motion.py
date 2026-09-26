"""Optical flow (RAFT or OpenCV DIS fallback), warping, and forward-backward
occlusion estimation.

Convention: flow_fw[t] maps pixels of frame t to frame t+1;
            flow_bw[t] maps pixels of frame t+1 back to frame t.
To bring frame t into alignment with frame t+1 use warp(frame_t, flow_bw[t]).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("genjutsu.motion")


def _gray8(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


class DISFlow:
    name = "dis"

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        self._dis = cv2.DISOpticalFlow_create(preset)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._dis.calc(_gray8(a), _gray8(b), None).astype(np.float32)


class RAFTFlow:
    """torchvision RAFT-Large. Loaded lazily; needs torch + torchvision."""

    name = "raft"

    def __init__(self, device: str = "cuda", iters: int = 12):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(self.device)
        self.iters = iters

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = a.shape[:2]
        H8, W8 = (H + 7) // 8 * 8, (W + 7) // 8 * 8

        def prep(x):
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float()
            t = torch.nn.functional.pad(t, (0, W8 - W, 0, H8 - H), mode="replicate")
            return (t * 2 - 1).to(self.device)

        with torch.inference_mode():
            flows = self.model(prep(a), prep(b), num_flow_updates=self.iters)
        return flows[-1][0, :, :H, :W].permute(1, 2, 0).float().cpu().numpy()


def make_flow_estimator(model: str = "raft", fallback: str = "dis", device: str = "cuda"):
    if model == "raft":
        try:
            return RAFTFlow(device=device)
        except Exception as e:  # torch/torchvision/weights unavailable
            log.warning("RAFT unavailable (%s); falling back to %s", e, fallback)
    return DISFlow()


def compute_flow(frames: np.ndarray, estimator=None, both: bool = True, max_side: int = 0):
    """Return (flow_fw, flow_bw) each (T-1,H,W,2). max_side computes at lower
    resolution and upsamples (flow vectors rescaled)."""
    est = estimator or DISFlow()
    T, H, W = frames.shape[:3]
    scale = 1.0
    src = frames
    if max_side and max(H, W) > max_side:
        scale = max_side / max(H, W)
        w, h = int(W * scale), int(H * scale)
        src = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])

    def up(fl):
        if scale == 1.0:
            return fl
        return cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) / scale

    fw = np.zeros((max(T - 1, 0), H, W, 2), np.float32)
    bw = np.zeros_like(fw) if both else None
    for t in range(T - 1):
        fw[t] = up(est(src[t], src[t + 1]))
        if both:
            bw[t] = up(est(src[t + 1], src[t]))
    return fw, bw


def _grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return gx, gy


def warp(img: np.ndarray, flow: np.ndarray, border=cv2.BORDER_REPLICATE, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Backward warp: out(x) = img(x + flow(x))."""
    H, W = flow.shape[:2]
    gx, gy = _grid(H, W)
    return cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1], interp, borderMode=border)


def fb_consistency(flow_fw_t: np.ndarray, flow_bw_t: np.ndarray, alpha: float = 0.01, beta: float = 0.5) -> np.ndarray:
    """Occlusion/unreliability mask for frame t+1 (1 = flow unreliable),
    from forward-backward check (Sundaram et al. 2010)."""
    fw_at_bw = warp(flow_fw_t, flow_bw_t)
    diff = flow_bw_t + fw_at_bw
    sq = (diff**2).sum(-1)
    mag = (flow_bw_t**2).sum(-1) + (fw_at_bw**2).sum(-1)
    return (sq > alpha * mag + beta).astype(np.float32)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt((flow**2).sum(-1))


def flow_to_rgb(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """Visualize one flow field as RGB [0,1]."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    m = max_mag or (np.percentile(mag, 99) + 1e-6)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / m * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
