"""Restoration: upscaling (Real-ESRGAN via spandrel, Lanczos fallback),
restrained sharpening, and optional RIFE-free frame interpolation."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("genjutsu.restoration")


class SpandrelUpscaler:
    """Loads any spandrel-supported SR model file (Real-ESRGAN, 4x-UltraSharp,
    etc.). ComfyUI already depends on spandrel, so this works in-graph."""

    def __init__(self, model_path: str, device: str = "cuda", tile: int = 512):
        import torch
        from spandrel import ModelLoader

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ModelLoader().load_from_file(model_path).to(self.device).eval()
        self.scale = int(self.model.scale)
        self.tile = tile

    def _run(self, x):
        with self.torch.inference_mode():
            return self.model(x)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].float().to(self.device)
        H, W = frame.shape[:2]
        if not self.tile or max(H, W) <= self.tile:
            y = self._run(x)
        else:
            s, t, pad = self.scale, self.tile, 16
            y = torch.zeros(1, 3, H * s, W * s, device=self.device)
            for y0 in range(0, H, t):
                for x0 in range(0, W, t):
                    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
                    yb, xb = min(y0 + t + pad, H), min(x0 + t + pad, W)
                    out = self._run(x[..., ya:yb, xa:xb])
                    oy, ox = (y0 - ya) * s, (x0 - xa) * s
                    h, w = (min(y0 + t, H) - y0) * s, (min(x0 + t, W) - x0) * s
                    y[..., y0 * s : y0 * s + h, x0 * s : x0 * s + w] = out[..., oy : oy + h, ox : ox + w]
        return y[0].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()


def upscale(frames: np.ndarray, scale: float = 2.0, model_path: Optional[str] = None, device: str = "cuda") -> np.ndarray:
    T, H, W = frames.shape[:3]
    tw, th = int(round(W * scale)), int(round(H * scale))
    if model_path:
        try:
            sr = SpandrelUpscaler(model_path, device)
            out = []
            for f in frames:
                y = sr(f)
                if y.shape[:2] != (th, tw):
                    y = cv2.resize(y, (tw, th), interpolation=cv2.INTER_AREA)
                out.append(y)
            return np.stack(out).astype(np.float32)
        except Exception as e:
            log.warning("SR model unavailable (%s); using Lanczos", e)
    return np.stack([cv2.resize(f, (tw, th), interpolation=cv2.INTER_LANCZOS4) for f in frames]).clip(0, 1).astype(np.float32)


def sharpen(frames: np.ndarray, amount: float = 0.15, radius: float = 1.2, threshold: float = 0.01) -> np.ndarray:
    """Unsharp mask with a threshold and hard cap on amount: AI frames
    over-sharpen easily (halos, crunchy skin), so the default is gentle."""
    amount = float(np.clip(amount, 0.0, 0.6))
    if amount <= 0:
        return frames
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        blur = cv2.GaussianBlur(f, (0, 0), radius)
        detail = f - blur
        detail = np.where(np.abs(detail) > threshold, detail, 0)
        out[t] = f + amount * detail
    return out.clip(0, 1)


def restore(frames: np.ndarray, scale: float = 1.0, model_path: Optional[str] = None, sharpen_amount: float = 0.15, device: str = "cuda") -> np.ndarray:
    out = frames
    if scale and scale != 1.0:
        out = upscale(out, scale, model_path, device)
    return sharpen(out, sharpen_amount)
