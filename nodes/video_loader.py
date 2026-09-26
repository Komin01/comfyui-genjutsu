from __future__ import annotations

import numpy as np

from ._util import CATEGORY, VIDEO, list_videos, resolve_video, to_image


class GenjutsuVideoLoader:
    """Decode a video into frames and keep its metadata (fps, audio) so the
    output node can re-attach the original audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_videos() or ["<put videos in ComfyUI/input>"], {"video_upload": True}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 = all"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "keep every Nth frame"}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16, "tooltip": "downscale longest side; 0 = native"}),
            },
            "optional": {"video_path": ("STRING", {"default": "", "tooltip": "absolute path overrides the dropdown"})},
        }

    RETURN_TYPES = ("IMAGE", VIDEO, "FLOAT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "video_info", "fps", "width", "height", "frame_count", "duration")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, video, video_path="", **kw):
        import os

        p = video_path or video
        try:
            p = resolve_video(p)
            return f"{p}:{os.path.getmtime(p)}:{kw}"
        except Exception:
            return float("nan")

    def load(self, video, max_frames, start_frame, stride, max_side, video_path=""):
        from genjutsu.video import load_frames

        path = resolve_video(video_path.strip() or video)
        frames, info = load_frames(path, start_frame, max_frames, stride, max_side)
        meta = {
            "path": path, "fps": info.fps, "width": info.width, "height": info.height, "frame_count": len(frames),
            "has_audio": info.has_audio, "audio_codec": info.audio_codec, "start_frame": start_frame,
            "audio_offset_s": start_frame / (info.fps * stride) if info.fps else 0.0,
        }
        dur = len(frames) / info.fps if info.fps else 0.0
        return (to_image(frames), meta, float(info.fps), info.width, info.height, len(frames), float(dur))


NODE_CLASS_MAPPINGS = {"GenjutsuVideoLoader": GenjutsuVideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuVideoLoader": "Genjutsu Video Loader"}
