"""Compositing: mask-driven blending (alpha, Laplacian multiband, Poisson),
depth-aware occlusion protection, and temporally-smoothed color/lighting
matching of generated content to the original plate."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .segmentation import dilate
from .video import to_uint8


# ----------------------------------------------------------------- color

def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32).clip(0, 1), cv2.COLOR_RGB2LAB)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB).clip(0, 1)


def _stats(lab: np.ndarray, m: Optional[np.ndarray]):
    px = lab.reshape(-1, 3) if m is None or not (m > 0.5).any() else lab[m > 0.5]
    return px.mean(0), px.std(0) + 1e-4


def color_match(
    generated: np.ndarray,
    original: np.ndarray,
    gen_region: Optional[np.ndarray] = None,
    ref_region: Optional[np.ndarray] = None,
    strength: float = 1.0,
    temporal_window: int = 9,
    match_contrast: bool = True,
    ab_mode: str = "reinhard",
) -> np.ndarray:
    """Reinhard LAB transfer: matches exposure (L mean), contrast (L std) and
    white balance / color temperature (a,b). Transfer parameters are
    smoothed over time so the correction itself cannot flicker.

    For region edits pass gen_region=edit ring and ref_region=same ring on
    the original so the generated object inherits the local lighting."""
    T = len(generated)
    params = []
    for t in range(T):
        gm, gs = _stats(_lab(generated[t]), None if gen_region is None else gen_region[t])
        om, os_ = _stats(_lab(original[t]), None if ref_region is None else ref_region[t])
        params.append(np.concatenate([gm, gs, om, os_]))
    params = np.asarray(params)
    if T >= 3 and temporal_window > 1:
        k = temporal_window | 1
        pad = k // 2
        params = np.stack([np.convolve(np.pad(params[:, i], pad, mode="edge"), np.ones(k) / k, mode="valid") for i in range(12)], 1)
    out = np.empty_like(generated)
    for t in range(T):
        gm, gs, om, os_ = params[t, :3], params[t, 3:6], params[t, 6:9], params[t, 9:12]
        lab = _lab(generated[t])
        ratio = os_ / gs if match_contrast else np.ones(3)
        ratio = np.clip(ratio, 0.5, 2.0)
        if ab_mode == "offset":
            # only shift white balance by the surroundings' difference; never
            # rescale chroma (keeps product / brand colors intact)
            ratio = np.array([ratio[0], 1.0, 1.0])
        new = (lab - gm) * ratio + om
        out[t] = _rgb((1 - strength) * lab + strength * new)
    return out


def boundary_ring(mask: np.ndarray, inner_px: int = 4, outer_px: int = 24) -> np.ndarray:
    """Band just outside the edit mask: where local lighting is sampled."""
    outer = dilate(mask, outer_px)
    inner = dilate(mask, inner_px)
    return (outer - inner).clip(0, 1)


def match_local_luminance(generated: np.ndarray, original: np.ndarray, soft: np.ndarray, sigma: float = 25.0, strength: float = 0.6) -> np.ndarray:
    """Match low-frequency shading (shadows/highlights) of the original inside
    the edit region: out = gen * blur(orig_L)/blur(gen_L)."""
    out = np.empty_like(generated)
    for t in range(len(generated)):
        gL = cv2.GaussianBlur(_lab(generated[t])[..., 0], (0, 0), sigma) + 1.0
        oL = cv2.GaussianBlur(_lab(original[t])[..., 0], (0, 0), sigma) + 1.0
        gain = np.clip(oL / gL, 0.6, 1.6)
        gain = 1 + (gain - 1) * strength * soft[t]
        lab = _lab(generated[t])
        lab[..., 0] = np.clip(lab[..., 0] * gain, 0, 100)
        out[t] = _rgb(lab)
    return out


# ----------------------------------------------------------------- blending

def alpha_composite(original: np.ndarray, generated: np.ndarray, soft: np.ndarray) -> np.ndarray:
    a = soft[..., None].astype(np.float32)
    return (original * (1 - a) + generated * a).astype(np.float32)


def _lap_pyr(img, levels):
    g = [img]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lp.append(g[i] - up)
    lp.append(g[-1])
    return lp


def multiband_blend(original: np.ndarray, generated: np.ndarray, soft: np.ndarray, levels: int = 5) -> np.ndarray:
    """Laplacian pyramid blend per frame: hides seams in both low and high
    frequencies better than a single feathered alpha."""
    out = np.empty_like(original)
    for t in range(len(original)):
        H, W = soft[t].shape
        lv = int(min(levels, np.log2(max(min(H, W), 2)) - 3))
        lv = max(lv, 1)
        la, lb = _lap_pyr(original[t].astype(np.float32), lv), _lap_pyr(generated[t].astype(np.float32), lv)
        gm = [soft[t].astype(np.float32)]
        for _ in range(lv):
            gm.append(cv2.pyrDown(gm[-1]))
        blended = [a * (1 - m[..., None]) + b * m[..., None] for a, b, m in zip(la, lb, gm)]
        img = blended[-1]
        for i in range(lv - 1, -1, -1):
            img = cv2.pyrUp(img, dstsize=(blended[i].shape[1], blended[i].shape[0])) + blended[i]
        # inside the solid core use the generated pixels exactly, outside the original exactly
        core = (soft[t] >= 0.999)[..., None]
        bg = (soft[t] <= 0.001)[..., None]
        out[t] = np.where(core, generated[t], np.where(bg, original[t], img))
    return out.clip(0, 1)


def poisson_blend(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.empty_like(original)
    for t in range(len(original)):
        m = ((mask[t] > 0.5) * 255).astype(np.uint8)
        if m.sum() == 0:
            out[t] = original[t]
            continue
        x, y, w, h = cv2.boundingRect(m)
        H, W = m.shape
        # seamlessClone needs the ROI strictly inside the image
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            m[:1, :] = m[-1:, :] = 0
            m[:, :1] = m[:, -1:] = 0
            x, y, w, h = cv2.boundingRect(m)
            if w == 0 or h == 0:
                out[t] = original[t]
                continue
        center = (x + w // 2, y + h // 2)
        src = cv2.cvtColor(to_uint8(generated[t]), cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(to_uint8(original[t]), cv2.COLOR_RGB2BGR)
        res = cv2.seamlessClone(src, dst, m, center, cv2.NORMAL_CLONE)
        out[t] = cv2.cvtColor(res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return out


# ----------------------------------------------------------------- occlusion

def occlusion_protect(depth: np.ndarray, edit_mask: np.ndarray, occluder_masks: list[np.ndarray], margin: float = 0.03) -> np.ndarray:
    """Pixels of other layers (hands, people, props) that are nearer than the
    edited object where they overlap its dilated footprint. These must come
    from the original plate so the new object passes *behind* them."""
    protect = np.zeros_like(edit_mask)
    near_edit = dilate(edit_mask, 8)
    for t in range(len(edit_mask)):
        e = edit_mask[t] > 0.5
        if not e.any():
            continue
        de = np.median(depth[t][e])
        for occ in occluder_masks:
            o = (occ[t] > 0.5) & (near_edit[t] > 0.5)
            if not o.any():
                continue
            # per-pixel test, robust to depth noise by comparing to the object's median depth
            protect[t] = np.maximum(protect[t], (o & (depth[t] > de + margin)).astype(np.float32))
    return protect


def depth_occlusion_from_scene(depth: np.ndarray, edit_mask: np.ndarray, margin: float = 0.05, band_px: int = 12) -> np.ndarray:
    """Without explicit occluder masks: anything inside the edit mask's
    dilated band that is clearly nearer than the object is treated as an
    occluder."""
    band = dilate(edit_mask, band_px)
    protect = np.zeros_like(edit_mask)
    for t in range(len(edit_mask)):
        e = edit_mask[t] > 0.5
        if not e.any():
            continue
        de = np.percentile(depth[t][e], 75)
        protect[t] = ((band[t] > 0.5) & (depth[t] > de + margin)).astype(np.float32)
    return protect


# ----------------------------------------------------------------- top level

def composite(
    original: np.ndarray,
    generated: np.ndarray,
    soft_mask: np.ndarray,
    preserve_mask: Optional[np.ndarray] = None,
    edit_mask: Optional[np.ndarray] = None,
    mode: str = "multiband",
    color_match_strength: float = 1.0,
    local_luminance: float = 0.5,
) -> np.ndarray:
    """output = original*preserve + generated*edit + blended boundary (section 26)."""
    if generated.shape != original.shape:
        from .video import resize_frames

        generated = resize_frames(generated, original.shape[2], original.shape[1], cv2.INTER_LANCZOS4)
    gen = generated
    if color_match_strength > 0 and edit_mask is not None:
        ring = boundary_ring(edit_mask)
        gen = color_match(gen, original, gen_region=ring, ref_region=ring, strength=color_match_strength, ab_mode="offset")
    if local_luminance > 0:
        gen = match_local_luminance(gen, original, soft_mask, strength=local_luminance)
    soft = soft_mask
    if preserve_mask is not None and edit_mask is not None:
        # any pixel that is hard-preserved and not part of the edit is untouched
        soft = soft * (1 - (preserve_mask > 0.999) * (edit_mask < 0.5))
    if mode == "alpha":
        out = alpha_composite(original, gen, soft)
    elif mode == "poisson":
        out = alpha_composite(original, poisson_blend(original, gen, soft > 0.5), soft)
    else:
        out = multiband_blend(original, gen, soft)
    return out.clip(0, 1).astype(np.float32)


def mask_leakage(original: np.ndarray, output: np.ndarray, preserve_mask: np.ndarray, tol: float = 2 / 255) -> float:
    """Fraction of preserved pixels that changed by more than `tol`."""
    diff = np.abs(original - output).max(-1)
    p = preserve_mask > 0.999
    return float(((diff > tol) & p).sum() / max(p.sum(), 1))
