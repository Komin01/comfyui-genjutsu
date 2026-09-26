"""Helpers shared by the ComfyUI nodes: tensor <-> numpy conversion, lazy
ComfyUI imports (so the package is importable outside ComfyUI for tests),
and a pipeline factory bound to ComfyUI's progress bar."""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

CATEGORY = "Genjutsu"

# Custom socket types
VIDEO = "GENJUTSU_VIDEO"
FLOW = "GENJUTSU_FLOW"
DEPTH = "GENJUTSU_DEPTH"
CAMERA = "GENJUTSU_CAMERA"
POSE = "GENJUTSU_POSE"
REFS = "GENJUTSU_REFS"
CONTROLS = "GENJUTSU_CONTROLS"
SCORE = "GENJUTSU_SCORE"

OPERATIONS = ["object_swap", "product_swap", "character_swap", "motion_transfer", "outfit_swap",
              "environment_recast", "style_recast", "full_recast"]
REF_CATEGORIES = ["auto", "character", "product", "object", "outfit", "location", "style", "prop", "texture", "logo"]
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")


def _torch():
    import torch

    return torch


def to_np(x: Any) -> Optional[np.ndarray]:
    """ComfyUI IMAGE (B,H,W,C) / MASK (B,H,W) tensor -> float32 numpy."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return x.detach().float().cpu().numpy()


def to_image(a: np.ndarray):
    """numpy (T,H,W,3) -> ComfyUI IMAGE tensor (falls back to numpy outside ComfyUI)."""
    a = np.ascontiguousarray(a, np.float32)
    try:
        return _torch().from_numpy(a)
    except ImportError:
        return a


to_mask = to_image


def mask_seq(mask: Any, T: int, H: int, W: int) -> Optional[np.ndarray]:
    """Accept a single mask or a sequence; broadcast / resize to (T,H,W)."""
    m = to_np(mask)
    if m is None:
        return None
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:  # IMAGE used as a mask
        m = m[..., 0]
    if m.shape[1:] != (H, W):
        from genjutsu.video import resize_masks

        m = resize_masks(m, W, H)
    if len(m) == 1 and T > 1:
        m = np.repeat(m, T, 0)
    if len(m) < T:
        m = np.concatenate([m, np.repeat(m[-1:], T - len(m), 0)], 0)
    return m[:T]


def progress_cb(total_steps: int = 100):
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(total_steps)

        def cb(stage: str, frac: float):
            bar.update_absolute(int(frac * total_steps), total_steps)

        return cb
    except Exception:
        return None


def input_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_input_directory()
    except Exception:
        return os.path.abspath("input")


def output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.path.abspath("output")


def list_videos() -> list[str]:
    d = input_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                out.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(out)


def resolve_video(name: str) -> str:
    if os.path.isabs(name) and os.path.exists(name):
        return name
    p = os.path.join(input_dir(), name)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Video not found: {name}")


def get_config(overrides: Optional[dict] = None):
    from genjutsu.config import load_config

    cfg = load_config()
    if "cache" not in (overrides or {}):
        try:
            import folder_paths

            cfg = cfg.with_overrides({"cache": {"root": os.path.join(folder_paths.get_temp_directory(), "genjutsu_cache")}})
        except Exception:
            pass
    return cfg.with_overrides(overrides or {})


def parse_box(s: str) -> Optional[list]:
    s = (s or "").strip()
    if not s:
        return None
    vals = [float(v) for v in s.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("box must be 'x0,y0,x1,y1'")
    return vals


def parse_points(s: str) -> tuple[Optional[list], Optional[list]]:
    """'x,y; x,y; -x,y' -> points and labels (a leading '-' marks a negative point)."""
    s = (s or "").strip()
    if not s:
        return None, None
    pts, labels = [], []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        neg = part.startswith("-")
        x, y = [float(v) for v in part.lstrip("-+").split(",")]
        pts.append([x, y])
        labels.append(0 if neg else 1)
    return pts, labels
