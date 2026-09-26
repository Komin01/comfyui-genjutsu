from __future__ import annotations

import json

import numpy as np

from ._util import CATEGORY, CONTROLS, REFS, mask_seq, to_image, to_np


class GenjutsuTemporalRepair:
    """Check generated frames against the *source* motion (optical flow) and
    repair shimmer/flicker without smearing real motion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "controls": (CONTROLS,),
                "repair_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smoothing_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "global exposure de-flicker"}),
                "inconsistency_threshold": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01}),
                "limit_to_edit_region": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("repaired", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, generated, controls, repair_strength, smoothing_strength, inconsistency_threshold, limit_to_edit_region):
        from genjutsu.motion import DISFlow, compute_flow
        from genjutsu.temporal import deflicker_luminance, detect_flicker_frames, flicker_score, temporal_repair
        from genjutsu.video import resize_frames

        g = to_np(generated)
        cp = controls
        if g.shape[1:3] != cp.frames.shape[1:3]:
            g = resize_frames(g, cp.frames.shape[2], cp.frames.shape[1])
        fw, bw = cp.flow_fw, cp.flow_bw
        if fw is None:
            fw, bw = compute_flow(cp.frames, DISFlow(), max_side=640)
        n = min(len(g), len(fw) + 1)
        g = g[:n]
        before = flicker_score(g, fw[: n - 1], bw[: n - 1])
        region = cp.soft_mask[:n] if (limit_to_edit_region and cp.soft_mask is not None) else None
        out = temporal_repair(g, fw[: n - 1], bw[: n - 1], repair_strength, inconsistency_threshold, region)
        if smoothing_strength > 0 and region is None:
            out = deflicker_luminance(out, strength=smoothing_strength)
        after = flicker_score(out, fw[: n - 1], bw[: n - 1])
        rep = {"flicker_score_before": round(before, 4), "flicker_score_after": round(after, 4),
               "outlier_frames": detect_flicker_frames(out, fw[: n - 1], bw[: n - 1])}
        return (to_image(out), json.dumps(rep, indent=2))


class GenjutsuIdentityLock:
    """Measure identity against the CHARACTER reference on every frame and
    report frames that fall below threshold (as regeneration segments)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "references": (REFS,),
                "threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stride": ("INT", {"default": 2, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("mean_identity", "report", "regenerate_segments")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, generated, references, threshold, stride):
        from genjutsu.controls import RefCategory
        from genjutsu.identity import face_embedding, identity_scores, segments_to_regenerate

        from ._util import get_config

        dev = get_config().get("device")
        ref = references.primary(RefCategory.CHARACTER)
        if ref is None:
            raise ValueError("Identity Lock needs a CHARACTER reference")
        emb = ref.embedding if ref.embedding is not None and ref.embedding.shape[-1] == 512 else face_embedding(ref.image, dev)
        if emb is None:
            raise ValueError("No face found in the character reference")
        g = to_np(generated)
        rep = identity_scores(g, emb, threshold, stride, dev)
        segs = segments_to_regenerate(rep.below_threshold, len(g))
        return (float(rep.mean), json.dumps(rep.as_dict()), json.dumps(segs))


class GenjutsuFrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "factor": ("INT", {"default": 2, "min": 2, "max": 8})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, factor):
        from genjutsu.motion import DISFlow, compute_flow
        from genjutsu.temporal import interpolate_frames

        f = to_np(frames)
        fw, bw = compute_flow(f, DISFlow())
        return (to_image(interpolate_frames(f, fw, bw, factor)),)


NODE_CLASS_MAPPINGS = {
    "GenjutsuTemporalRepair": GenjutsuTemporalRepair,
    "GenjutsuIdentityLock": GenjutsuIdentityLock,
    "GenjutsuFrameInterpolate": GenjutsuFrameInterpolate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenjutsuTemporalRepair": "Genjutsu Temporal Repair",
    "GenjutsuIdentityLock": "Genjutsu Identity Lock",
    "GenjutsuFrameInterpolate": "Genjutsu Frame Interpolate",
}
