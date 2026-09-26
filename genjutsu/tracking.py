"""Mask tracking through a video: flow propagation with drift control,
occlusion handling, disappearance detection and template re-identification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .motion import DISFlow, compute_flow, warp
from .segmentation import mask_bbox
from .video import to_uint8


@dataclass
class TrackResult:
    masks: np.ndarray  # (T,H,W)
    boxes: list  # per frame [x0,y0,x1,y1] or None
    visible: list  # per frame bool
    reidentified: list = field(default_factory=list)  # frames where re-id snapped the track back


def _crop(frame8: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame8[y0:y1, x0:x1]


def _reidentify(frame8: np.ndarray, template: np.ndarray, min_score: float, scales=(0.8, 0.9, 1.0, 1.1, 1.25)):
    best = (-1.0, None, 1.0)
    H, W = frame8.shape[:2]
    for s in scales:
        th, tw = int(template.shape[0] * s), int(template.shape[1] * s)
        if th < 8 or tw < 8 or th >= H or tw >= W:
            continue
        tpl = cv2.resize(template, (tw, th))
        res = cv2.matchTemplate(frame8, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, (loc[0], loc[1], loc[0] + tw, loc[1] + th), s)
    return best if best[0] >= min_score else (best[0], None, 1.0)


class AppearanceModel:
    """Foreground/background color likelihood (HSV histograms) learned on the
    prompt frame. Used to snap flow-propagated masks back onto the object,
    which matters for flat, textureless objects where flow is weak."""

    def __init__(self, frame8: np.ndarray, mask: np.ndarray, bins: int = 16, ring_px: int = 15):
        hsv = cv2.cvtColor(frame8, cv2.COLOR_RGB2HSV)
        fg = (mask > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
        bg = (cv2.dilate(fg, k) - fg).astype(np.uint8)
        rng = [0, 180, 0, 256, 0, 256]
        self.bins = bins
        fh = cv2.calcHist([hsv], [0, 1, 2], fg * 255, [bins] * 3, rng)
        bh = cv2.calcHist([hsv], [0, 1, 2], bg * 255, [bins] * 3, rng)
        # asymmetric priors: a color never seen on the object is far more likely
        # to belong to something else (an occluder, new background) than to it
        self.fg = fh / max(fh.sum(), 1.0) + 1e-5
        self.bg = bh / max(bh.sum(), 1.0) + 1e-3

    def prob(self, frame8: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame8, cv2.COLOR_RGB2HSV).astype(np.int32)
        idx = (hsv[..., 0] * self.bins // 180, hsv[..., 1] * self.bins // 256, hsv[..., 2] * self.bins // 256)
        f, b = self.fg[idx], self.bg[idx]
        return (f / (f + b)).astype(np.float32)


def _refine(m: np.ndarray, frame8: np.ndarray, app: "AppearanceModel", weight: float) -> np.ndarray:
    area = m.sum()
    if area < 4 or weight <= 0:
        return m
    r = int(max(4, 0.25 * np.sqrt(area)))
    band = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)).astype(np.float32)
    p = app.prob(frame8)
    p = cv2.GaussianBlur(p, (0, 0), 1.0)
    score = (1 - weight) * cv2.GaussianBlur(m, (0, 0), 1.5) + weight * p
    # threshold > 0.5: flow alone can no longer keep pixels whose color says "not the object"
    return ((score > 0.55) & (band > 0)).astype(np.float32)


def _reacquire(frame8: np.ndarray, app: "AppearanceModel", ref_area: float, near, frames_missing: int = 1,
               min_mean_p: float = 0.7):
    """Find the object again by color likelihood: connected blobs of
    object-colored pixels with a plausible size, within a search radius of
    the last known position that widens the longer the object is missing."""
    radius = np.sqrt(ref_area) * (2.5 + 0.75 * max(frames_missing, 1))
    p = cv2.GaussianBlur(app.prob(frame8), (0, 0), 1.0)
    cand = (p > 0.5).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best, best_d = None, 1e18
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (0.35 * ref_area <= area <= 3.0 * ref_area):
            continue
        blob = lab == i
        if p[blob].mean() < min_mean_p:
            continue
        d = (cents[i][0] - near[0]) ** 2 + (cents[i][1] - near[1]) ** 2
        if d > radius**2:
            continue
        if d < best_d:
            best, best_d = blob, d
    if best is None:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(best.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)


def _place_mask(mask_src: np.ndarray, box_src, box_dst, shape) -> np.ndarray:
    x0, y0, x1, y1 = box_src
    patch = mask_src[y0:y1, x0:x1]
    dx0, dy0, dx1, dy1 = box_dst
    patch = cv2.resize(patch, (max(1, dx1 - dx0), max(1, dy1 - dy0)), interpolation=cv2.INTER_LINEAR)
    out = np.zeros(shape, np.float32)
    H, W = shape
    cx0, cy0 = max(dx0, 0), max(dy0, 0)
    cx1, cy1 = min(dx1, W), min(dy1, H)
    out[cy0:cy1, cx0:cx1] = patch[cy0 - dy0 : cy1 - dy0, cx0 - dx0 : cx1 - dx0]
    return out


def track(
    frames: np.ndarray,
    init_mask: np.ndarray,
    start: int = 0,
    flow_fw: Optional[np.ndarray] = None,
    flow_bw: Optional[np.ndarray] = None,
    lost_area_frac: float = 0.15,
    reid_score: float = 0.55,
    cleanup_px: int = 2,
    appearance_weight: float = 0.5,
) -> TrackResult:
    T, H, W = frames.shape[:3]
    if flow_fw is None or flow_bw is None:
        flow_fw, flow_bw = compute_flow(frames, DISFlow(), both=True, max_side=640)
    f8 = to_uint8(frames)
    masks = np.zeros((T, H, W), np.float32)
    masks[start] = (init_mask > 0.5).astype(np.float32)
    ref_area = max(masks[start].sum(), 1.0)
    b0 = mask_bbox(masks[start])
    template = _crop(f8[start], b0) if b0 else None
    tpl_mask_frame, tpl_box = masks[start].copy(), b0
    app = AppearanceModel(f8[start], masks[start]) if b0 else None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cleanup_px + 1,) * 2)
    visible = [False] * T
    visible[start] = b0 is not None
    last_center = [((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2) if b0 else (W / 2, H / 2), start]
    reids: list[int] = []

    def step(src_t: int, dst_t: int, fl: np.ndarray, last_good: int):
        nonlocal template, tpl_mask_frame, tpl_box
        m = warp(masks[src_t], fl, border=cv2.BORDER_CONSTANT)
        m = (m > 0.5).astype(np.float32)
        if app is not None:
            m = _refine(m, f8[dst_t], app, appearance_weight)
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k).astype(np.float32)
        area = m.sum()
        if area >= lost_area_frac * ref_area:
            masks[dst_t] = m
            visible[dst_t] = True
            ys, xs = np.nonzero(m)
            last_center[0] = (xs.mean(), ys.mean())
            last_center[1] = dst_t
            # refresh template occasionally when the track is healthy
            if abs(dst_t - last_good) >= 12 and 0.6 * ref_area < area < 1.6 * ref_area:
                bb = mask_bbox(m)
                if bb:
                    template, tpl_mask_frame, tpl_box = _crop(f8[dst_t], bb), m.copy(), bb
                    return dst_t
            return last_good
        # lost or occluded: try to re-identify
        found = _reacquire(f8[dst_t], app, ref_area, last_center[0], abs(dst_t - last_center[1])) if app is not None else None
        if found is None and template is not None and template.size and template.std() > 6:
            # template matching only makes sense for textured objects
            score, box, _ = _reidentify(f8[dst_t], template, reid_score)
            if box is not None:
                cand = _place_mask(tpl_mask_frame, tpl_box, box, (H, W))
                gap = max(abs(dst_t - last_center[1]), 1)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                near = (cx - last_center[0][0]) ** 2 + (cy - last_center[0][1]) ** 2 <= (np.sqrt(ref_area) * (2.5 + 0.75 * gap)) ** 2
                colour_ok = app is None or app.prob(f8[dst_t])[cand > 0.5].mean() > 0.5 if cand.any() else False
                if near and colour_ok:
                    found = cand
        if found is not None:
            masks[dst_t] = found
            visible[dst_t] = True
            reids.append(dst_t)
            ys, xs = np.nonzero(found > 0.5)
            if len(xs):
                last_center[:] = [(xs.mean(), ys.mean()), dst_t]
            return dst_t
        masks[dst_t] = m  # keep the partial (possibly empty) mask
        visible[dst_t] = area > 0
        return last_good

    lg = start
    for t in range(start, T - 1):
        lg = step(t, t + 1, flow_bw[t], lg)
    lg = start
    last_center[:] = [((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2) if b0 else (W / 2, H / 2), start]
    for t in range(start, 0, -1):
        lg = step(t, t - 1, flow_fw[t - 1], lg)
    boxes = [mask_bbox(m) for m in masks]
    return TrackResult(masks=masks, boxes=boxes, visible=visible, reidentified=reids)


def track_mask(frames, init_mask, start=0, flow_fw=None, flow_bw=None) -> np.ndarray:
    return track(frames, init_mask, start=start, flow_fw=flow_fw, flow_bw=flow_bw).masks
