from __future__ import annotations

import json

import numpy as np

from ._util import CATEGORY, to_np


class GenjutsuSceneAnalyzer:
    """Scene decomposition: hard cuts, shot list with per-shot statistics,
    and (if Grounding DINO is installed) the subjects/objects present."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "cut_threshold": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.01}),
                "detect": ("STRING", {"default": "person, face, hand, product, car, animal", "tooltip": "comma-separated labels to look for (needs Grounding DINO)"}),
                "sample_every": ("INT", {"default": 16, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("scene_json", "scene_cuts", "num_shots")
    FUNCTION = "analyze"
    CATEGORY = CATEGORY

    def analyze(self, frames, cut_threshold, detect, sample_every):
        from genjutsu.video import detect_scene_cuts

        f = to_np(frames)
        cuts = detect_scene_cuts(f, cut_threshold)
        bounds = [0] + cuts + [len(f)]
        shots = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = f[a:b]
            shots.append({"start": a, "end": b, "mean_luma": float(seg.mean()), "motion": float(np.abs(np.diff(seg, axis=0)).mean()) if b - a > 1 else 0.0})
        subjects: dict = {}
        labels = [l.strip() for l in detect.split(",") if l.strip()]
        if labels:
            try:
                from genjutsu.segmentation import GroundingDetector

                det = GroundingDetector()
                for t in range(0, len(f), sample_every):
                    for lab in labels:
                        hits = det.detect(f[t], lab)
                        if hits:
                            subjects.setdefault(lab, []).append({"frame": t, "score": round(hits[0][0], 3), "box": [round(v, 1) for v in hits[0][1]]})
            except Exception as e:
                subjects = {"unavailable": f"object detection needs transformers + Grounding DINO ({type(e).__name__})"}
        out = {"frames": len(f), "height": int(f.shape[1]), "width": int(f.shape[2]), "cuts": cuts, "shots": shots, "subjects": subjects}
        return (json.dumps(out, indent=2), ",".join(map(str, cuts)), len(shots))


class GenjutsuOpticalFlow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": (["raft", "dis"], {"default": "raft"}),
                "max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 32, "tooltip": "compute at this size, upsample result"}),
            }
        }

    RETURN_TYPES = ("GENJUTSU_FLOW", "IMAGE")
    RETURN_NAMES = ("flow", "flow_preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, model, max_side):
        from genjutsu.motion import compute_flow, flow_to_rgb, make_flow_estimator

        from ._util import to_image

        f = to_np(frames)
        fw, bw = compute_flow(f, make_flow_estimator(model), max_side=max_side)
        prev = np.stack([flow_to_rgb(x) for x in fw] + ([flow_to_rgb(fw[-1])] if len(fw) else [np.zeros_like(f[0])]))
        return ({"fw": fw, "bw": bw}, to_image(prev))


class GenjutsuCameraAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",)}, "optional": {"exclude_mask": ("MASK", {"tooltip": "moving subjects to ignore"})}}

    RETURN_TYPES = ("GENJUTSU_CAMERA", "STRING")
    RETURN_NAMES = ("camera", "summary")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, exclude_mask=None):
        from genjutsu.camera import estimate_camera

        from ._util import mask_seq

        f = to_np(frames)
        m = mask_seq(exclude_mask, *f.shape[:3])
        cam = estimate_camera(f, m)
        return (cam, json.dumps(cam["summary"], indent=2))


NODE_CLASS_MAPPINGS = {
    "GenjutsuSceneAnalyzer": GenjutsuSceneAnalyzer,
    "GenjutsuOpticalFlow": GenjutsuOpticalFlow,
    "GenjutsuCameraAnalyzer": GenjutsuCameraAnalyzer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenjutsuSceneAnalyzer": "Genjutsu Scene Analyzer",
    "GenjutsuOpticalFlow": "Genjutsu Optical Flow",
    "GenjutsuCameraAnalyzer": "Genjutsu Camera Analyzer",
}
