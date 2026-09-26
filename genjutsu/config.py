"""Configuration loading (config.yaml) with defaults for every component."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "backend": "wan22",
    "device": "cuda",
    "dtype": "bfloat16",
    "models": {
        "wan22_t2v": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "wan22_i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "wan22_ti2v": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "wan_vace": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "wan_animate": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
    },
    "segmentation": {"model": "sam2", "checkpoint": "facebook/sam2.1-hiera-large"},
    "depth": {"model": "video_depth_anything", "checkpoint": "vits"},
    "pose": {"model": "dwpose"},
    "optical_flow": {"model": "raft", "fallback": "dis"},
    "identity": {"enabled": True, "model": "insightface", "threshold": 0.45},
    "temporal": {
        "enabled": True,
        "strength": 0.75,
        "smoothing_strength": 0.3,
        "repair_strength": 0.6,
        "inconsistency_threshold": 0.08,
    },
    "compositing": {"enabled": True, "dilate_px": 6, "feather_px": 12, "color_match": True},
    "upscale": {"enabled": False, "model": "realesrgan", "scale": 2, "sharpen": 0.15},
    "chunking": {"chunk_frames": 81, "overlap_frames": 9},
    "performance": {"low_vram": False, "cpu_offload": True, "tile_size": 0},
    "cache": {"enabled": True, "root": "./genjutsu_cache"},
    "output": {"codec": "h264", "crf": 16, "container": "mp4"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class GenjutsuConfig:
    data: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def with_overrides(self, overrides: dict) -> "GenjutsuConfig":
        return GenjutsuConfig(_deep_merge(self.data, overrides))


def load_config(path: str | os.PathLike | None = None) -> GenjutsuConfig:
    """Load config.yaml, merged over DEFAULTS. Missing file -> defaults."""
    if path is None:
        env = os.environ.get("GENJUTSU_CONFIG")
        path = env or Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        return GenjutsuConfig()
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return GenjutsuConfig(_deep_merge(DEFAULTS, user))
