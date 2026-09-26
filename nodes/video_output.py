from __future__ import annotations

import json
import os
import time

from ._util import CATEGORY, CONTROLS, REFS, VIDEO, mask_seq, output_dir, to_np


class GenjutsuVideoOutput:
    """Encode with FFmpeg (H.264 / H.265 / ProRes / VP9 / PNG sequence),
    keeping the original FPS and audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "genjutsu/out"}),
                "codec": (["h264", "h265", "prores", "vp9", "png"], {"default": "h264"}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 51}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "tooltip": "0 = source fps"}),
                "keep_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {"video_info": (VIDEO,), "audio_path": ("STRING", {"default": "", "tooltip": "replace audio with this file"})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, frames, filename_prefix, codec, crf, fps, keep_audio, video_info=None, audio_path=""):
        from genjutsu.video import CODECS, write_video

        info = video_info or {}
        fps = fps or float(info.get("fps", 24.0))
        ext = CODECS[codec][1]
        base = os.path.join(output_dir(), f"{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
        path = base if codec == "png" else f"{base}.{ext}"
        audio = audio_path.strip() or (info.get("path") if keep_audio else None)
        write_video(to_np(frames), path, fps, codec, crf, audio_from=audio, audio_offset_s=float(info.get("audio_offset_s", 0.0)) if not audio_path.strip() else 0.0)
        rel = os.path.relpath(path, output_dir())
        ui = {"text": [rel]}
        if codec != "png":
            sub, name = os.path.split(rel)
            ui["gifs"] = [{"filename": name, "subfolder": sub, "type": "output", "format": f"video/{ext}"}]
        return {"ui": ui, "result": (path,)}


class GenjutsuSaveProject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"controls": (CONTROLS,), "name": ("STRING", {"default": "genjutsu/project"}), "embed_source": ("BOOLEAN", {"default": False})},
            "optional": {"output_path": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, controls, name, embed_source, output_path=None):
        from genjutsu.project import save_project

        from ._util import get_config

        p = save_project(os.path.join(output_dir(), name), controls, settings=get_config().data, embed_source=embed_source, output_path=output_path)
        return {"ui": {"text": [p]}, "result": (p,)}


class GenjutsuLoadProject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"project_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = (CONTROLS, "IMAGE", REFS, "MASK", "STRING")
    RETURN_NAMES = ("controls", "frames", "references", "edit_mask", "manifest")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, project_path):
        import numpy as np

        from genjutsu.project import load_project

        from ._util import to_image, to_mask

        p = project_path if os.path.isabs(project_path) else os.path.join(output_dir(), project_path)
        cp, man = load_project(p)
        em = cp.edit_mask if cp.edit_mask is not None else np.zeros(cp.frames.shape[:3], np.float32)
        return (cp, to_image(cp.frames), cp.references, to_mask(em), json.dumps(man, indent=2))


class GenjutsuEvaluate:
    """GenjutsuScore: identity, motion, camera, background, object, temporal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": ("IMAGE",), "output": ("IMAGE",), "max_side": ("INT", {"default": 480, "min": 128, "max": 2048})},
            "optional": {"controls": (CONTROLS,), "references": (REFS,)},
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("score_table", "overall")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, source, output, max_side, controls=None, references=None):
        from genjutsu.controls import RefCategory
        from genjutsu.metrics import evaluate

        s, o = to_np(source), to_np(output)
        if o.shape[1:3] != s.shape[1:3]:
            from genjutsu.video import resize_frames

            o = resize_frames(o, s.shape[2], s.shape[1])
        refs = references or (controls.references if controls else None)
        obj = refs.primary(RefCategory.PRODUCT, RefCategory.OBJECT, RefCategory.PROP) if refs else None
        ch = refs.primary(RefCategory.CHARACTER) if refs else None
        emb = ch.embedding if ch is not None and ch.embedding is not None and ch.embedding.shape[-1] == 512 else None
        sc = evaluate(s, o, getattr(controls, "preserve_mask", None), getattr(controls, "edit_mask", None),
                      obj.image if obj else None, obj.mask if obj else None, emb, max_side, object_mask=getattr(controls, "subject_mask", None))
        table = sc.table()
        return {"ui": {"text": [table]}, "result": (table, float(sc.overall() or 0.0))}


NODE_CLASS_MAPPINGS = {
    "GenjutsuVideoOutput": GenjutsuVideoOutput,
    "GenjutsuSaveProject": GenjutsuSaveProject,
    "GenjutsuLoadProject": GenjutsuLoadProject,
    "GenjutsuEvaluate": GenjutsuEvaluate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenjutsuVideoOutput": "Genjutsu Video Output",
    "GenjutsuSaveProject": "Genjutsu Save Project (.genjutsu)",
    "GenjutsuLoadProject": "Genjutsu Load Project (.genjutsu)",
    "GenjutsuEvaluate": "Genjutsu Score",
}
