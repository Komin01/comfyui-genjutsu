from __future__ import annotations

import numpy as np

from ._util import CATEGORY, mask_seq, to_image, to_np


class GenjutsuCompositor:
    """output = original*preserve + generated*edit + blended boundary, with
    color / local-lighting matching of the generated content first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "generated": ("IMAGE",),
                "soft_mask": ("MASK",),
                "mode": (["multiband", "alpha", "poisson"], {"default": "multiband"}),
                "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "local_lighting": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {"preserve_mask": ("MASK",), "edit_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("composite", "mask_leakage")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, original, generated, soft_mask, mode, color_match, local_lighting, preserve_mask=None, edit_mask=None):
        from genjutsu.compositing import composite, mask_leakage

        o = to_np(original)
        g = to_np(generated)
        n = min(len(o), len(g))
        o, g = o[:n], g[:n]
        T, H, W = o.shape[:3]
        s = mask_seq(soft_mask, T, H, W)
        p = mask_seq(preserve_mask, T, H, W)
        e = mask_seq(edit_mask, T, H, W)
        if e is None:
            e = (s > 0.5).astype(np.float32)
        out = composite(o, g, s, p, e, mode, color_match, local_lighting)
        leak = mask_leakage(o, out, p) if p is not None else 0.0
        return (to_image(out), float(leak))


class GenjutsuColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "original": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "chroma": (["reinhard", "offset"], {"default": "reinhard", "tooltip": "offset keeps object colors, only fixes white balance"}),
                "temporal_window": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2}),
            },
            "optional": {"region": ("MASK", {"tooltip": "where to sample statistics (e.g. a ring around the edit)"})},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, generated, original, strength, chroma, temporal_window, region=None):
        from genjutsu.compositing import color_match

        g, o = to_np(generated), to_np(original)
        n = min(len(g), len(o))
        r = mask_seq(region, n, *g.shape[1:3]) if region is not None else None
        if o.shape[1:3] != g.shape[1:3]:
            from genjutsu.video import resize_frames

            o = resize_frames(o, g.shape[2], g.shape[1])
        return (to_image(color_match(g[:n], o[:n], r, r, strength, temporal_window, True, chroma)),)


class GenjutsuUpscale:
    """Real-ESRGAN (or any spandrel model in ComfyUI/models/upscale_models),
    then restrained sharpening. Lanczos if no model is selected."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            models = ["none (lanczos)"] + folder_paths.get_filename_list("upscale_models")
        except Exception:
            models = ["none (lanczos)"]
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": (models,),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25}),
                "sharpen": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.6, "step": 0.05, "tooltip": "capped: AI frames over-sharpen easily"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, model, scale, sharpen):
        from genjutsu.restoration import restore

        from ._util import get_config

        path = None
        if not model.startswith("none"):
            import folder_paths

            path = folder_paths.get_full_path("upscale_models", model)
        return (to_image(restore(to_np(frames), scale, path, sharpen, get_config().get("device"))),)


NODE_CLASS_MAPPINGS = {"GenjutsuCompositor": GenjutsuCompositor, "GenjutsuColorMatch": GenjutsuColorMatch, "GenjutsuUpscale": GenjutsuUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuCompositor": "Genjutsu Compositor", "GenjutsuColorMatch": "Genjutsu Color / Lighting Match", "GenjutsuUpscale": "Genjutsu Upscale / Restore"}
