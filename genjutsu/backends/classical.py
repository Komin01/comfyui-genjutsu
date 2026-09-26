"""A model-free backend. It is a real (if simple) implementation of every
operation, used for dry-runs, CI, and as a template for new backends:

- region ops: fit the reference cut-out into the tracked mask, frame by frame,
  following the mask's position, scale and aspect;
- character ops: returns the source (no generative model -> no new identity);
- recasts: Reinhard color/style transfer from the location/style reference.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..controls import ControlPackage, Operation, RefCategory
from ..compositing import color_match
from ..segmentation import mask_bbox
from .base import CHARACTER_OPS, RECAST_OPS, REGION_OPS, VideoBackend, register_backend


@register_backend("classical")
class ClassicalBackend(VideoBackend):
    supports = REGION_OPS | CHARACTER_OPS | RECAST_OPS

    def generate_region(self, cp: ControlPackage) -> np.ndarray:
        ref = cp.references.primary(RefCategory.PRODUCT, RefCategory.OBJECT, RefCategory.PROP, RefCategory.OUTFIT, RefCategory.LOGO)
        if ref is None or cp.edit_mask is None:
            return cp.frames.copy()
        img = ref.image
        rmask = ref.mask if ref.mask is not None else np.ones(img.shape[:2], np.float32)
        rb = mask_bbox(rmask) or (0, 0, img.shape[1], img.shape[0])
        img_c = img[rb[1] : rb[3], rb[0] : rb[2]]
        m_c = rmask[rb[1] : rb[3], rb[0] : rb[2]]
        src_mask = cp.subject_mask if cp.subject_mask is not None else cp.edit_mask
        out = cp.frames.copy()
        H, W = out.shape[1:3]
        for t in range(len(out)):
            bb = mask_bbox(src_mask[t])
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            bw, bh = x1 - x0, y1 - y0
            ih, iw = img_c.shape[:2]
            s = min(bw / iw, bh / ih)  # fit, keep aspect
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            px0, py0 = cx - nw // 2, cy - nh // 2
            pi = cv2.resize(img_c, (nw, nh), interpolation=cv2.INTER_AREA)
            pm = cv2.resize(m_c, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ax0, ay0, ax1, ay1 = max(px0, 0), max(py0, 0), min(px0 + nw, W), min(py0 + nh, H)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            sl = (slice(ay0 - py0, ay1 - py0), slice(ax0 - px0, ax1 - px0))
            a = pm[sl][..., None]
            # 1) remove the old object: inpaint its (slightly dilated) footprint
            #    from a padded window so there is real background to draw from
            pad = max(8, int(0.5 * max(bw, bh)))
            wx0, wy0, wx1, wy1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
            hole = (src_mask[t, wy0:wy1, wx0:wx1] > 0.5).astype(np.uint8)
            if hole.any():
                hole = cv2.dilate(hole, np.ones((9, 9), np.uint8))  # mask can lag a fast edge by a few px
                win = (np.clip(out[t, wy0:wy1, wx0:wx1], 0, 1) * 255).astype(np.uint8)
                out[t, wy0:wy1, wx0:wx1] = cv2.inpaint(win, hole, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
            # 2) place the reference cut-out
            region = out[t, ay0:ay1, ax0:ax1]
            out[t, ay0:ay1, ax0:ax1] = region * (1 - a) + pi[sl] * a
        return out

    def generate_character(self, cp: ControlPackage) -> np.ndarray:
        return cp.frames.copy()

    def generate_recast(self, cp: ControlPackage) -> np.ndarray:
        ref = cp.references.primary(RefCategory.LOCATION, RefCategory.STYLE, RefCategory.TEXTURE)
        if ref is None:
            return cp.frames.copy()
        ref_video = np.repeat(ref.image[None], len(cp.frames), 0) if ref.image.shape == cp.frames.shape[1:] else np.stack(
            [cv2.resize(ref.image, (cp.frames.shape[2], cp.frames.shape[1]))] * len(cp.frames)
        )
        region = None
        if Operation(cp.operation) == Operation.ENVIRONMENT_RECAST and cp.subject_mask is not None:
            region = 1.0 - cp.subject_mask
        out = color_match(cp.frames, ref_video, gen_region=region, strength=cp.strengths.style if cp.strengths else 0.8)
        if region is not None:
            a = region[..., None]
            out = cp.frames * (1 - a) + out * a
        return out.astype(np.float32)
