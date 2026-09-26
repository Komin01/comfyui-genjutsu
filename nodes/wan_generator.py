from __future__ import annotations

import json

from ._util import CATEGORY, CONTROLS, get_config, progress_cb, to_image


class GenjutsuGenerate:
    """Run the configured video backend on a ControlPackage. Long clips are
    generated in overlapping chunks; every finished chunk is cached, so a
    crash or OOM resumes from the last completed chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        from genjutsu.backends import available_backends

        return {
            "required": {
                "controls": (CONTROLS,),
                "backend": (available_backends(), {"default": "wan22"}),
                "max_side": ("INT", {"default": 832, "min": 256, "max": 1920, "step": 16, "tooltip": "generation resolution (longest side)"}),
                "max_area": ("INT", {"default": 832 * 480, "min": 65536, "max": 1920 * 1088, "step": 1024}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "low_vram": ("BOOLEAN", {"default": False, "tooltip": "sequential offload: slowest, smallest footprint"}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vace_model": ("STRING", {"default": "", "tooltip": "override models.wan_vace (HF id or local folder)"}),
                "animate_model": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, controls, backend, max_side, max_area, cpu_offload, low_vram, vae_tiling, vace_model="", animate_model=""):
        from genjutsu.pipeline import GenjutsuPipeline

        models = {}
        if vace_model.strip():
            models["wan_vace"] = vace_model.strip()
        if animate_model.strip():
            models["wan_animate"] = animate_model.strip()
        cfg = get_config({
            "backend": backend, "models": models,
            "generation": {"max_side": max_side, "max_area": max_area},
            "performance": {"cpu_offload": cpu_offload, "low_vram": low_vram, "tile_size": 1 if vae_tiling else 0},
        })
        pipe = GenjutsuPipeline(cfg, project="comfyui", progress=progress_cb())
        out = pipe.generate(controls, backend)
        info = {"backend": backend, "frames": int(out.shape[0]), "size": [int(out.shape[2]), int(out.shape[1])], "prompt": controls.prompt}
        return (to_image(out), json.dumps(info, indent=2))


NODE_CLASS_MAPPINGS = {"GenjutsuGenerate": GenjutsuGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuGenerate": "Genjutsu Generate (Wan 2.2 / backends)"}
