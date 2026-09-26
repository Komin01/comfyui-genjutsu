from __future__ import annotations

import json

import numpy as np

from ._util import CATEGORY, FLOW, mask_seq, parse_box, parse_points, to_mask, to_np

TARGETS = ["person", "face", "hands", "clothing", "object", "product", "foreground", "background", "custom"]


class GenjutsuSegmenter:
    """Temporally stable video masks. SAM 2 + Grounding DINO when installed;
    GrabCut + flow tracking fallback when only a box is given."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "target": (TARGETS, {"default": "person"}),
                "text": ("STRING", {"default": "", "tooltip": "used when target=custom, or to refine (e.g. 'wrist watch')"}),
                "box": ("STRING", {"default": "", "tooltip": "x0,y0,x1,y1 on prompt_frame"}),
                "points": ("STRING", {"default": "", "tooltip": "x,y; x,y; -x,y (minus = negative point)"}),
                "prompt_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "model": (["sam2", "grabcut"], {"default": "sam2"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.95, "step": 0.05}),
                "fill_holes": ("BOOLEAN", {"default": True}),
            },
            "optional": {"flow": (FLOW,)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "segment"
    CATEGORY = CATEGORY

    def segment(self, frames, target, text, box, points, prompt_frame, model, temporal_smoothing, fill_holes, flow=None):
        from genjutsu.segmentation import TARGET_PROMPTS, fill_holes as _fill, remove_small, segment_video, temporal_smooth_masks

        from ._util import get_config

        f = to_np(frames)
        cfg = get_config()
        invert = target == "background"
        q = text.strip() or TARGET_PROMPTS.get("person" if invert else target, target)
        pts, labels = parse_points(points)
        fw = flow["fw"] if flow else None
        bw = flow["bw"] if flow else None
        masks = segment_video(f, q, parse_box(box), pts, labels, min(prompt_frame, len(f) - 1), model,
                              cfg.get("segmentation.checkpoint"), cfg.get("device"), bw, fw)
        masks = remove_small(masks)
        if fill_holes:
            masks = _fill(masks)
        masks = temporal_smooth_masks(masks, bw, temporal_smoothing)
        masks = (masks > 0.5).astype(np.float32)
        if invert:
            masks = 1.0 - masks
        return (to_mask(masks),)


class GenjutsuTracker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "initial_mask": ("MASK", {"tooltip": "single mask for start_frame (or a sequence; start_frame is used)"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "appearance_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "reid_score": ("FLOAT", {"default": 0.55, "min": 0.1, "max": 0.99, "step": 0.01, "tooltip": "template-match score to re-acquire a lost object"}),
            },
            "optional": {"flow": (FLOW,)},
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("tracked_masks", "track_report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, initial_mask, start_frame, appearance_weight, reid_score, flow=None):
        from genjutsu.tracking import track

        f = to_np(frames)
        T, H, W = f.shape[:3]
        m = to_np(initial_mask)
        m0 = mask_seq(m[min(start_frame, len(m) - 1)] if m.ndim == 3 else m, 1, H, W)[0]
        res = track(f, m0, start=min(start_frame, T - 1), flow_fw=flow["fw"] if flow else None, flow_bw=flow["bw"] if flow else None,
                    reid_score=reid_score, appearance_weight=appearance_weight)
        rep = {"visible_frames": int(sum(res.visible)), "total": T, "reidentified_at": res.reidentified,
               "lost_frames": [t for t, v in enumerate(res.visible) if not v]}
        return (to_mask(res.masks), json.dumps(rep))


class GenjutsuPreservationMasks:
    """EDIT -> dilation -> feather -> SOFT, with protected regions (e.g. hands
    that pass in front of the edited object) and optional depth occlusion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_mask": ("MASK",),
                "dilate_px": ("INT", {"default": 6, "min": 0, "max": 128}),
                "feather_px": ("INT", {"default": 12, "min": 0, "max": 256}),
                "depth_occlusion": ("BOOLEAN", {"default": True}),
            },
            "optional": {"protect_mask": ("MASK",), "depth": ("GENJUTSU_DEPTH",)},
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("edit_mask", "preserve_mask", "soft_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, edit_mask, dilate_px, feather_px, depth_occlusion, protect_mask=None, depth=None):
        from genjutsu.compositing import depth_occlusion_from_scene, occlusion_protect
        from genjutsu.segmentation import build_preservation_masks

        e = to_np(edit_mask)
        if e.ndim == 2:
            e = e[None]
        T, H, W = e.shape
        prot = mask_seq(protect_mask, T, H, W)
        d = None
        if depth is not None:
            from genjutsu.video import resize_masks

            d = resize_masks(depth, W, H)[:T]
        if prot is not None and d is not None:
            prot = occlusion_protect(d, e, [prot])
        elif prot is None and depth_occlusion and d is not None:
            prot = depth_occlusion_from_scene(d, e)
        a, b, c = build_preservation_masks(e, prot, dilate_px, feather_px)
        return (to_mask(a), to_mask(b), to_mask(c))


class GenjutsuMaskOps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("MASK",),
                "op": (["union", "subtract", "intersect", "invert_a", "dilate_a", "erode_a", "feather_a"],),
                "amount_px": ("INT", {"default": 8, "min": 0, "max": 256}),
            },
            "optional": {"b": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, a, op, amount_px, b=None):
        from genjutsu.segmentation import dilate, erode, feather

        A = to_np(a)
        A = A[None] if A.ndim == 2 else A
        B = mask_seq(b, *A.shape) if b is not None else np.zeros_like(A)
        out = {
            "union": lambda: np.maximum(A, B), "subtract": lambda: (A - B).clip(0, 1), "intersect": lambda: np.minimum(A, B),
            "invert_a": lambda: 1 - A, "dilate_a": lambda: dilate(A, amount_px), "erode_a": lambda: erode(A, amount_px),
            "feather_a": lambda: feather(A, amount_px),
        }[op]()
        return (to_mask(out.astype(np.float32)),)


NODE_CLASS_MAPPINGS = {
    "GenjutsuSegmenter": GenjutsuSegmenter,
    "GenjutsuTracker": GenjutsuTracker,
    "GenjutsuPreservationMasks": GenjutsuPreservationMasks,
    "GenjutsuMaskOps": GenjutsuMaskOps,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenjutsuSegmenter": "Genjutsu Segmenter",
    "GenjutsuTracker": "Genjutsu Tracker",
    "GenjutsuPreservationMasks": "Genjutsu Preservation Masks",
    "GenjutsuMaskOps": "Genjutsu Mask Ops",
}
