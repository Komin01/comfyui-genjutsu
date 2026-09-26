"""The simplified front door (section 36): one node exposing only
SOURCE / OPERATION / REFERENCES / PROMPT / PRESERVATION / GENERATION / OUTPUT.
Advanced users build the same graph from the individual nodes."""
from __future__ import annotations

import json
import os
import time

from ._util import CATEGORY, OPERATIONS, REFS, get_config, list_videos, output_dir, parse_box, progress_cb, resolve_video, to_image

QUALITY = {
    "draft": {"steps": 12, "max_side": 640, "max_area": 640 * 368},
    "standard": {"steps": 30, "max_side": 832, "max_area": 832 * 480},
    "high": {"steps": 40, "max_side": 1280, "max_area": 1280 * 720},
}


class GenjutsuOneClick:
    @classmethod
    def INPUT_TYPES(cls):
        from genjutsu.backends import available_backends

        return {
            "required": {
                "video": (list_videos() or ["<put videos in ComfyUI/input>"], {"video_upload": True}),
                "operation": (OPERATIONS, {"default": "object_swap"}),
                "references": (REFS,),
                "target": ("STRING", {"default": "", "tooltip": "what to select/replace: 'watch', 'person', 'bottle'..."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "preserve_motion": ("BOOLEAN", {"default": True}),
                "preserve_camera": ("BOOLEAN", {"default": True}),
                "preserve_background": ("BOOLEAN", {"default": True}),
                "preserve_identity": ("BOOLEAN", {"default": True}),
                "preserve_lighting": ("BOOLEAN", {"default": True}),
                "quality": (list(QUALITY), {"default": "standard"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "backend": (available_backends(), {"default": "wan22"}),
                "codec": (["h264", "h265", "prores", "vp9"], {"default": "h264"}),
                "upscale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "box": ("STRING", {"default": "", "tooltip": "x0,y0,x1,y1 on frame 0 instead of a text target"}),
                "protect": ("STRING", {"default": "hand", "tooltip": "comma-separated things that must stay in front (occluders)"}),
                "video_path": ("STRING", {"default": ""}),
                "save_project": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_path", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, video, operation, references, target, prompt, preserve_motion, preserve_camera, preserve_background,
            preserve_identity, preserve_lighting, quality, strength, seed, backend, codec, upscale, max_frames,
            box="", protect="hand", video_path="", save_project=True):
        from genjutsu.pipeline import GenjutsuPipeline

        q = QUALITY[quality]
        cfg = get_config({
            "backend": backend, "output": {"codec": codec},
            "generation": {"max_side": q["max_side"], "max_area": q["max_area"]},
            "upscale": {"enabled": upscale > 1.0, "scale": upscale},
        })
        preserve = [k for k, v in {"motion": preserve_motion, "camera": preserve_camera, "background": preserve_background,
                                   "identity": preserve_identity, "lighting": preserve_lighting}.items() if v]
        src = resolve_video(video_path.strip() or video)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ext = {"prores": "mov", "vp9": "webm"}.get(codec, "mp4")
        out = os.path.join(output_dir(), "genjutsu", f"{operation}_{stamp}.{ext}")
        proj = os.path.join(output_dir(), "genjutsu", f"{operation}_{stamp}") if save_project else None
        pipe = GenjutsuPipeline(cfg, project=os.path.splitext(os.path.basename(src))[0], progress=progress_cb())
        res = pipe.run(
            src, operation, references, target=target, box=parse_box(box), prompt=prompt, preserve=preserve,
            protect_targets=[p.strip() for p in protect.split(",") if p.strip()] if operation in ("object_swap", "product_swap") else None,
            seed=seed, steps=q["steps"], backend=backend, out_path=out, project_path=proj, max_frames=max_frames,
            strength=strength,
        )
        report = json.dumps({k: v for k, v in res.report.items()}, indent=2, default=str)
        rel = os.path.relpath(res.output_path, output_dir())
        sub, name = os.path.split(rel)
        ext = name.rsplit(".", 1)[-1]
        return {"ui": {"text": [report], "gifs": [{"filename": name, "subfolder": sub, "type": "output", "format": f"video/{ext}"}]},
                "result": (to_image(res.frames), res.output_path, report)}


NODE_CLASS_MAPPINGS = {"GenjutsuOneClick": GenjutsuOneClick}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuOneClick": "Genjutsu ✦ One-Click Transform"}
