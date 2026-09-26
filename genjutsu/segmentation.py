"""Segmentation: SAM 2 video predictor (+ Grounding DINO for text prompts),
a GrabCut fallback for box prompts, and the mask algebra the preservation
system is built on (edit / preserve / soft-boundary masks)."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Sequence

import cv2
import numpy as np

from .video import to_uint8

log = logging.getLogger("genjutsu.segmentation")

# Natural-language targets per category the Segmenter node exposes.
TARGET_PROMPTS = {
    "person": "person",
    "face": "human face",
    "hands": "hand",
    "clothing": "clothing",
    "object": "object",
    "product": "product",
    "foreground": "person",
}


# --------------------------------------------------------------------------
# Mask algebra
# --------------------------------------------------------------------------

def _kernel(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def dilate(masks: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return masks
    k = _kernel(px)
    return np.stack([cv2.dilate(m, k) for m in masks]).astype(np.float32)


def erode(masks: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return masks
    k = _kernel(px)
    return np.stack([cv2.erode(m, k) for m in masks]).astype(np.float32)


def feather(masks: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return masks
    s = max(px / 2.0, 0.5)
    return np.stack([cv2.GaussianBlur(m, (0, 0), s) for m in masks]).clip(0, 1).astype(np.float32)


def fill_holes(masks: np.ndarray, max_hole_frac: float = 0.02) -> np.ndarray:
    out = []
    for m in masks:
        b = (m > 0.5).astype(np.uint8)
        inv = 1 - b
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        limit = max_hole_frac * m.size
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            touches = x == 0 or y == 0 or x + w == m.shape[1] or y + h == m.shape[0]
            if not touches and area < limit:
                b[lab == i] = 1
        out.append(np.maximum(m, b.astype(np.float32)))
    return np.stack(out)


def remove_small(masks: np.ndarray, min_frac: float = 0.001) -> np.ndarray:
    out = []
    for m in masks:
        b = (m > 0.5).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        keep = np.zeros_like(b)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_frac * m.size:
                keep[lab == i] = 1
        out.append(m * keep)
    return np.stack(out).astype(np.float32)


def temporal_smooth_masks(masks: np.ndarray, flow_bw: Optional[np.ndarray] = None, strength: float = 0.5) -> np.ndarray:
    """Reduce mask flicker: blend each mask with the flow-warped previous
    result (or plain EMA if no flow), then re-sharpen."""
    if len(masks) < 2 or strength <= 0:
        return masks
    from .motion import warp

    out = np.empty_like(masks)
    out[0] = masks[0]
    for t in range(1, len(masks)):
        prev = warp(out[t - 1], flow_bw[t - 1]) if flow_bw is not None else out[t - 1]
        out[t] = (1 - strength) * masks[t] + strength * prev
    # also run backwards so the smoothing is symmetric
    back = np.empty_like(out)
    back[-1] = out[-1]
    for t in range(len(masks) - 2, -1, -1):
        back[t] = (1 - strength * 0.5) * out[t] + strength * 0.5 * back[t + 1]
    return back.clip(0, 1)


def build_preservation_masks(
    edit: np.ndarray,
    protect: Optional[np.ndarray] = None,
    dilate_px: int = 6,
    feather_px: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EDIT -> dilation -> feather -> SOFT (section 17).

    Returns (edit_mask, preserve_mask, soft_mask).
    `protect` marks regions that must never change even if they fall inside
    the dilated edit area (e.g. a hand in front of a swapped watch)."""
    e = dilate((edit > 0.5).astype(np.float32), dilate_px)
    if protect is not None:
        e = e * (1.0 - (protect > 0.5))
    core = (edit > 0.5).astype(np.float32)
    if protect is not None:
        core = core * (1.0 - (protect > 0.5))
    # feathered transition outside, but the object itself is always fully generated
    soft = np.maximum(feather(e, feather_px), core)
    if protect is not None:
        soft = soft * (1.0 - protect.clip(0, 1))
    preserve = 1.0 - e
    return e.astype(np.float32), preserve.astype(np.float32), soft.astype(np.float32)


def mask_bbox(m: np.ndarray, thresh: float = 0.5) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(m > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_mask(shape_hw: tuple[int, int], box: Sequence[float]) -> np.ndarray:
    m = np.zeros(shape_hw, np.float32)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    m[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = 1.0
    return m


# --------------------------------------------------------------------------
# Segmentation backends
# --------------------------------------------------------------------------

def grabcut(frame: np.ndarray, box: Sequence[float], iters: int = 5) -> np.ndarray:
    """Classical single-frame segmentation from a box. Model-free fallback."""
    img = cv2.cvtColor(to_uint8(frame), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)


class GroundingDetector:
    """Text -> boxes using Grounding DINO through transformers."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, frame: np.ndarray, text: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        from PIL import Image

        img = Image.fromarray(to_uint8(frame))
        text = text.strip().lower()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[img.size[::-1]]
        )[0]
        boxes = res["boxes"].cpu().numpy().tolist()
        scores = res["scores"].cpu().numpy().tolist()
        return sorted(zip(scores, boxes), key=lambda s: -s[0])


class SAM2VideoSegmenter:
    """SAM 2 / 2.1 video predictor (github.com/facebookresearch/sam2)."""

    def __init__(self, checkpoint: str = "facebook/sam2.1-hiera-large", device: str = "cuda"):
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2VideoPredictor.from_pretrained(checkpoint, device=self.device)

    def segment(
        self,
        frames: np.ndarray,
        prompts: list[dict],
    ) -> np.ndarray:
        """prompts: [{"frame": i, "obj_id": k, "box": [x0,y0,x1,y1]} or
        {"frame": i, "obj_id": k, "points": [[x,y],...], "labels": [1,0,...]}].
        Returns union mask (T,H,W)."""
        T, H, W = frames.shape[:3]
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(to_uint8(frames)):
                cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
            with self.torch.inference_mode(), self.torch.autocast(self.device, dtype=dtype, enabled=self.device == "cuda"):
                state = self.predictor.init_state(video_path=d, offload_video_to_cpu=True)
                for p in prompts:
                    kw = {}
                    if "box" in p:
                        kw["box"] = np.asarray(p["box"], np.float32)
                    if "points" in p:
                        kw["points"] = np.asarray(p["points"], np.float32)
                        kw["labels"] = np.asarray(p.get("labels", [1] * len(p["points"])), np.int32)
                    self.predictor.add_new_points_or_box(state, frame_idx=p.get("frame", 0), obj_id=p.get("obj_id", 1), **kw)
                out = np.zeros((T, H, W), np.float32)
                for fidx, _obj_ids, logits in self.predictor.propagate_in_video(state):
                    m = (logits > 0.0).float().amax(0)[0].cpu().numpy()
                    out[fidx] = np.maximum(out[fidx], m)
                # propagate backwards too when prompts start mid-video
                if min(p.get("frame", 0) for p in prompts) > 0:
                    for fidx, _obj_ids, logits in self.predictor.propagate_in_video(state, reverse=True):
                        m = (logits > 0.0).float().amax(0)[0].cpu().numpy()
                        out[fidx] = np.maximum(out[fidx], m)
                self.predictor.reset_state(state)
        return out


def segment_video(
    frames: np.ndarray,
    text: str = "",
    box: Optional[Sequence[float]] = None,
    points: Optional[list] = None,
    point_labels: Optional[list] = None,
    prompt_frame: int = 0,
    model: str = "sam2",
    checkpoint: str = "facebook/sam2.1-hiera-large",
    device: str = "cuda",
    flow_bw: Optional[np.ndarray] = None,
    flow_fw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """High-level entry used by the Segmenter node.

    Resolution order: SAM2 (+Grounding DINO for text) -> GrabCut on the
    prompt frame + flow tracking. Raises if nothing usable was given."""
    T, H, W = frames.shape[:3]
    if model == "sam2":
        try:
            if box is None and not points and text:
                det = GroundingDetector(device=device).detect(frames[prompt_frame], text)
                if not det:
                    raise RuntimeError(f"Grounding DINO found no '{text}' on frame {prompt_frame}")
                box = det[0][1]
            prompts = []
            if box is not None:
                prompts.append({"frame": prompt_frame, "obj_id": 1, "box": list(box)})
            if points:
                prompts.append({"frame": prompt_frame, "obj_id": 1, "points": points, "labels": point_labels or [1] * len(points)})
            if not prompts:
                raise ValueError("Segmenter needs a text prompt, a box, or points")
            return SAM2VideoSegmenter(checkpoint, device).segment(frames, prompts)
        except (ImportError, ModuleNotFoundError, OSError) as e:
            log.warning("SAM2 unavailable (%s); using GrabCut + flow tracking fallback", e)
    if box is None and points:
        pts = np.asarray(points, np.float32)
        pad = 0.15 * max(W, H)
        box = [pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad]
    if box is None:
        raise ValueError(
            "No segmentation model is installed and no box/points were given. Install SAM2 "
            "(see README) or provide a box prompt."
        )
    from .tracking import track_mask

    init = grabcut(frames[prompt_frame], box)
    if init.sum() < 16:
        init = box_mask((H, W), box)
    return track_mask(frames, init, start=prompt_frame, flow_fw=flow_fw, flow_bw=flow_bw)
