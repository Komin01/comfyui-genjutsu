"""Depth estimation: Video Depth Anything (temporally consistent), a
per-frame Depth Anything V2 fallback stabilized with optical flow, and depth
utilities for layering / occlusion."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("genjutsu.depth")

VDA_REPO = {
    "vits": ("depth-anything/Video-Depth-Anything-Small", "video_depth_anything_vits.pth"),
    "vitb": ("depth-anything/Video-Depth-Anything-Base", "video_depth_anything_vitb.pth"),
    "vitl": ("depth-anything/Video-Depth-Anything-Large", "video_depth_anything_vitl.pth"),
}
VDA_CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def normalize_depth(depth: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Per-video (not per-frame) normalization to [0,1], larger = nearer.
    Per-video normalization is what keeps the sequence temporally stable."""
    lo, hi = np.percentile(depth, lo_pct), np.percentile(depth, hi_pct)
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def video_depth_anything(frames: np.ndarray, encoder: str = "vitl", fps: float = 24.0, device: str = "cuda", input_size: int = 518):
    import torch
    from huggingface_hub import hf_hub_download
    from video_depth_anything.video_depth import VideoDepthAnything  # repo: DepthAnything/Video-Depth-Anything

    dev = device if torch.cuda.is_available() else "cpu"
    repo, fname = VDA_REPO[encoder]
    model = VideoDepthAnything(**VDA_CFG[encoder])
    model.load_state_dict(torch.load(hf_hub_download(repo, fname), map_location="cpu"), strict=True)
    model = model.to(dev).eval()
    f8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    depths, _ = model.infer_video_depth(f8, fps, input_size=input_size, device=dev)
    return np.asarray(depths, np.float32)


def depth_anything_v2_per_frame(frames: np.ndarray, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cuda"):
    import torch
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model_id, device=0 if (device == "cuda" and torch.cuda.is_available()) else -1)
    out = []
    for f in frames:
        pred = pipe(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)))["predicted_depth"]
        d = pred.squeeze().float().cpu().numpy()
        out.append(cv2.resize(d, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_CUBIC))
    return np.stack(out).astype(np.float32)


def stabilize_depth(depth: np.ndarray, flow_bw: Optional[np.ndarray], strength: float = 0.5) -> np.ndarray:
    """Align per-frame scale/shift to the flow-warped previous frame, then
    blend. Turns flickery per-frame depth into a usable temporal signal."""
    if len(depth) < 2:
        return depth
    from .motion import warp

    out = depth.copy()
    for t in range(1, len(depth)):
        prev = warp(out[t - 1], flow_bw[t - 1]) if flow_bw is not None else out[t - 1]
        a, b = np.polyfit(depth[t].ravel()[::7], prev.ravel()[::7], 1)  # least-squares scale/shift
        aligned = a * depth[t] + b
        out[t] = (1 - strength) * aligned + strength * prev
    return out


def estimate_depth(
    frames: np.ndarray,
    model: str = "video_depth_anything",
    checkpoint: str = "vitl",
    fps: float = 24.0,
    device: str = "cuda",
    flow_bw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns normalized depth (T,H,W), 1 = nearest."""
    if model == "video_depth_anything":
        try:
            d = video_depth_anything(frames, checkpoint, fps, device)
            if d.shape[1:] != frames.shape[1:3]:
                d = np.stack([cv2.resize(x, (frames.shape[2], frames.shape[1])) for x in d])
            return normalize_depth(d)
        except (ImportError, ModuleNotFoundError, OSError) as e:
            log.warning("Video Depth Anything unavailable (%s); trying per-frame Depth Anything V2", e)
            model = "depth_anything_v2"
    if model == "depth_anything_v2":
        d = depth_anything_v2_per_frame(frames, device=device)
        return normalize_depth(stabilize_depth(d, flow_bw))
    raise ValueError(f"Unknown depth model {model}")


def layer_order(depth: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Per-pixel: 1 where A is in front of B inside their overlap region,
    using median depth of each object near the overlap (robust to noisy
    per-pixel depth). Returns (T,H,W) float."""
    out = np.zeros(mask_a.shape, np.float32)
    for t in range(len(depth)):
        a, b = mask_a[t] > 0.5, mask_b[t] > 0.5
        if not a.any() or not b.any():
            continue
        da, db = np.median(depth[t][a]), np.median(depth[t][b])
        overlap = (cv2.dilate(a.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & b
        if da > db + margin:
            out[t][overlap | a] = 1.0
    return out
