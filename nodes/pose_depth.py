from __future__ import annotations

import numpy as np

from ._util import CATEGORY, DEPTH, FLOW, POSE, to_image, to_np


class GenjutsuPoseExtractor:
    """DWPose whole-body pose (body, hands, face) as rendered skeleton frames
    plus 512px face crops for Wan-Animate. You can instead feed pose frames
    from comfyui_controlnet_aux straight into the Control Package node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "face_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05, "tooltip": "keypoint jitter reduction"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", POSE)
    RETURN_NAMES = ("pose_frames", "face_frames", "pose")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, face_size, smoothing):
        from genjutsu.pose import extract_pose, face_crops, render_skeleton, smooth_keypoints

        from ._util import get_config

        f = to_np(frames)
        pr = extract_pose(f, get_config().get("device"), True, face_size)
        if smoothing > 0:
            kps = smooth_keypoints(pr.keypoints, pr.scores, smoothing)
            pr.rendered = np.stack([render_skeleton(f.shape[1:3], k, s) for k, s in zip(kps, pr.scores)])
            pr.face_crops, pr.face_boxes = face_crops(f, kps, pr.scores, face_size)
            pr.keypoints = kps
        return (to_image(pr.rendered), to_image(pr.face_crops), {"keypoints": pr.keypoints, "scores": pr.scores, "face_boxes": pr.face_boxes})


class GenjutsuDepthExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": (["video_depth_anything", "depth_anything_v2"], {"default": "video_depth_anything"}),
                "encoder": (["vits", "vitb", "vitl"], {"default": "vits", "tooltip": "vits = Apache-2.0; vitb/vitl weights are CC-BY-NC-4.0 (non-commercial)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
            },
            "optional": {"flow": (FLOW, {"tooltip": "stabilizes per-frame depth models"})},
        }

    RETURN_TYPES = ("IMAGE", DEPTH)
    RETURN_NAMES = ("depth_preview", "depth")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, model, encoder, fps, flow=None):
        from genjutsu.depth import estimate_depth

        from ._util import get_config

        d = estimate_depth(to_np(frames), model, encoder, fps, get_config().get("device"), flow["bw"] if flow else None)
        return (to_image(np.repeat(d[..., None], 3, -1)), d)


NODE_CLASS_MAPPINGS = {"GenjutsuPoseExtractor": GenjutsuPoseExtractor, "GenjutsuDepthExtractor": GenjutsuDepthExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuPoseExtractor": "Genjutsu Pose Extractor", "GenjutsuDepthExtractor": "Genjutsu Depth Extractor"}
