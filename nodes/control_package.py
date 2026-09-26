from __future__ import annotations

import json

import numpy as np

from ._util import CAMERA, CATEGORY, CONTROLS, DEPTH, FLOW, OPERATIONS, REFS, VIDEO, mask_seq, to_np


class GenjutsuControlPackage:
    """Collect everything the generator needs into one model-independent
    ControlPackage. Only frames + operation are required; every other signal
    is optional and used when present."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "operation": (OPERATIONS, {"default": "object_swap"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "identity_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "reference_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preservation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "VACE conditioning scale for region edits"}),
                "style_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "chunk_frames": ("INT", {"default": 81, "min": 5, "max": 241, "step": 4}),
                "overlap_frames": ("INT", {"default": 9, "min": 0, "max": 64}),
                "recast_control": (["depth", "pose", "edges"], {"default": "depth"}),
            },
            "optional": {
                "video_info": (VIDEO,),
                "references": (REFS,),
                "edit_mask": ("MASK",),
                "preserve_mask": ("MASK",),
                "soft_mask": ("MASK",),
                "subject_mask": ("MASK",),
                "pose_frames": ("IMAGE",),
                "face_frames": ("IMAGE",),
                "depth": (DEPTH,),
                "flow": (FLOW,),
                "camera": (CAMERA,),
                "scene_cuts": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (CONTROLS, "STRING")
    RETURN_NAMES = ("controls", "summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, frames, operation, seed, steps, guidance, identity_strength, motion_strength, reference_strength,
              preservation_strength, style_strength, chunk_frames, overlap_frames, recast_control, video_info=None, references=None,
              edit_mask=None, preserve_mask=None, soft_mask=None, subject_mask=None, pose_frames=None, face_frames=None,
              depth=None, flow=None, camera=None, scene_cuts=None):
        from genjutsu.controls import ControlPackage, Operation, ReferencePackage, Strengths, TemporalSettings
        from genjutsu.segmentation import build_preservation_masks
        from genjutsu.video import resize_frames, resize_masks

        f = to_np(frames)
        T, H, W = f.shape[:3]
        em, pm, sm, sub = (mask_seq(m, T, H, W) for m in (edit_mask, preserve_mask, soft_mask, subject_mask))
        if em is not None and (pm is None or sm is None):
            _, pm2, sm2 = build_preservation_masks(em, None, 6, 12)
            pm = pm if pm is not None else pm2
            sm = sm if sm is not None else sm2
        if sub is None and em is not None:
            sub = em
        pose = to_np(pose_frames)
        if pose is not None and pose.shape[1:3] != (H, W):
            pose = resize_frames(pose, W, H)
        d = None if depth is None else resize_masks(np.asarray(depth), W, H)[:T]
        cuts = [int(c) for c in (scene_cuts or "").split(",") if c.strip().isdigit()]
        cp = ControlPackage(
            frames=f, fps=float((video_info or {}).get("fps", 24.0)), operation=Operation(operation),
            source_path=(video_info or {}).get("path", ""), edit_mask=em, preserve_mask=pm, soft_mask=sm, subject_mask=sub,
            pose_frames=pose, face_frames=to_np(face_frames), depth=d,
            flow_fw=flow["fw"] if flow else None, flow_bw=flow["bw"] if flow else None, camera=camera, scene_cuts=cuts,
            references=references or ReferencePackage(),
            temporal=TemporalSettings(chunk_frames=chunk_frames, overlap_frames=overlap_frames),
            strengths=Strengths(identity_strength, motion_strength, reference_strength, preservation_strength, style_strength),
            seed=seed, steps=steps, guidance=guidance, extras={"recast_control": recast_control, "video_info": video_info or {}},
        )
        return (cp, json.dumps(cp.describe(), indent=2))


class GenjutsuPromptCompiler:
    """High-level intent -> model prompts. Writes the prompts into the
    ControlPackage and also outputs them as strings for inspection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controls": (CONTROLS,),
                "target": ("STRING", {"default": "", "tooltip": "what is being replaced, e.g. 'wrist watch'"}),
                "motion": (["original", "free"], {"default": "original"}),
                "preserve_person": ("BOOLEAN", {"default": True}),
                "preserve_hands": ("BOOLEAN", {"default": True}),
                "preserve_background": ("BOOLEAN", {"default": True}),
                "preserve_camera": ("BOOLEAN", {"default": True}),
                "preserve_lighting": ("BOOLEAN", {"default": True}),
                "preserve_expression": ("BOOLEAN", {"default": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "environment": ("STRING", {"default": ""}),
                "style": ("STRING", {"default": ""}),
                "extra_negative": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (CONTROLS, "STRING", "STRING")
    RETURN_NAMES = ("controls", "positive", "negative")
    FUNCTION = "compile"
    CATEGORY = CATEGORY

    def compile(self, controls, target, motion, preserve_person, preserve_hands, preserve_background, preserve_camera,
                preserve_lighting, preserve_expression, user_prompt, environment, style, extra_negative):
        import copy

        from genjutsu.prompt import PromptSpec, compile_prompt

        cp = copy.copy(controls)
        flags = {"person": preserve_person, "hand": preserve_hands, "background": preserve_background, "camera": preserve_camera,
                 "lighting": preserve_lighting, "expression": preserve_expression}
        preserve = [k for k, v in flags.items() if v]
        descs = {r.category.value: r.metadata["description"] for r in cp.references.references if r.metadata.get("description")}
        spec = PromptSpec(cp.operation, target, preserve, motion, user_prompt, descs, style, environment, extra_negative)
        cp.prompt, cp.negative_prompt = compile_prompt(spec, cp.references)
        cp.invariants = sorted(set(cp.invariants) | {p for p in preserve if p in ("background", "camera", "lighting")} | ({"motion"} if motion == "original" else set()))
        return (cp, cp.prompt, cp.negative_prompt)


NODE_CLASS_MAPPINGS = {"GenjutsuControlPackage": GenjutsuControlPackage, "GenjutsuPromptCompiler": GenjutsuPromptCompiler}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuControlPackage": "Genjutsu Control Package", "GenjutsuPromptCompiler": "Genjutsu Prompt Compiler"}
