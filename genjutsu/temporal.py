"""Temporal consistency engine: flicker measurement, flow-guided repair,
luminance de-flicker, chunk planning with overlap, and chunk merging.

Key idea: in a transformation (not generation) the *source* video's motion is
an invariant, so the source's optical flow is the reference against which
the generated frames' temporal behaviour is checked and repaired."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .motion import fb_consistency, warp


# ----------------------------------------------------------------- measurement

def warp_error(frames: np.ndarray, flow_fw: np.ndarray, flow_bw: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-transition mean abs error between frame t+1 and warped frame t,
    ignoring occluded pixels. (T-1,) array; lower = more consistent."""
    errs = []
    for t in range(len(frames) - 1):
        w = warp(frames[t], flow_bw[t])
        valid = 1.0 - fb_consistency(flow_fw[t], flow_bw[t])
        if mask is not None:
            valid = valid * mask[t + 1]
        d = np.abs(frames[t + 1] - w).mean(-1)
        errs.append(float((d * valid).sum() / max(valid.sum(), 1.0)))
    return np.asarray(errs, np.float32)


def flicker_score(frames: np.ndarray, flow_fw=None, flow_bw=None, mask=None) -> float:
    """0..1, 1 = no flicker. With flow: warp error; without: luminance jitter."""
    if len(frames) < 2:
        return 1.0
    if flow_fw is not None and flow_bw is not None:
        e = float(np.mean(warp_error(frames, flow_fw, flow_bw, mask)))
        return float(np.exp(-e / 0.03))
    lum = frames.mean((1, 2, 3))
    jitter = np.abs(np.diff(lum, 2)).mean() if len(lum) > 2 else 0.0
    return float(np.exp(-jitter / 0.01))


def detect_flicker_frames(frames, flow_fw, flow_bw, z: float = 2.5) -> list[int]:
    """Frames whose warp error is an outlier relative to the clip."""
    e = warp_error(frames, flow_fw, flow_bw)
    if len(e) < 3:
        return []
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-6
    return [int(t + 1) for t in np.nonzero((e - med) / (1.4826 * mad) > z)[0]]


# ----------------------------------------------------------------- repair

def deflicker_luminance(frames: np.ndarray, window: int = 9, strength: float = 1.0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove global exposure pumping: match each frame's mean/std (in LAB L)
    to a temporally smoothed curve."""
    T = len(frames)
    if T < 3 or strength <= 0:
        return frames
    labs = [cv2.cvtColor(f.astype(np.float32), cv2.COLOR_RGB2LAB) for f in frames]
    stats = []
    for t, lab in enumerate(labs):
        L = lab[..., 0]
        sel = L[mask[t] > 0.5] if mask is not None and (mask[t] > 0.5).any() else L
        stats.append((sel.mean(), sel.std() + 1e-6))
    stats = np.asarray(stats)
    k = max(3, window | 1)
    pad = k // 2
    sm = np.stack([np.convolve(np.pad(stats[:, i], pad, mode="edge"), np.ones(k) / k, mode="valid") for i in range(2)], 1)
    out = np.empty_like(frames)
    for t, lab in enumerate(labs):
        m, s = stats[t]
        tm, ts = sm[t]
        L = (lab[..., 0] - m) * (ts / s) + tm
        lab2 = lab.copy()
        lab2[..., 0] = (1 - strength) * lab[..., 0] + strength * L
        out[t] = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out.clip(0, 1)


def temporal_repair(
    generated: np.ndarray,
    flow_fw: np.ndarray,
    flow_bw: np.ndarray,
    strength: float = 0.6,
    threshold: float = 0.08,
    region: Optional[np.ndarray] = None,
    bidirectional: bool = True,
) -> np.ndarray:
    """Pull each generated frame toward its flow-warped neighbours wherever
    they nearly agree (shimmer / texture boiling), leave genuine changes and
    occlusions alone. `region` limits repair to e.g. the edit mask."""
    if len(generated) < 2 or strength <= 0:
        return generated

    def one_pass(frames, fws, bws, reverse=False):
        out = frames.copy()
        idx = range(1, len(frames)) if not reverse else range(len(frames) - 2, -1, -1)
        for t in idx:
            s = t - 1 if not reverse else t + 1
            if not reverse:
                fl, occ = bws[t - 1], fb_consistency(fws[t - 1], bws[t - 1])
            else:
                fl, occ = fws[t], fb_consistency(bws[t], fws[t])
            w = warp(out[s], fl)
            err = cv2.GaussianBlur(np.abs(out[t] - w).mean(-1), (0, 0), 1.5)
            wt = strength * (1 - occ) * np.exp(-((err / threshold) ** 2))
            if region is not None:
                wt = wt * region[t]
            out[t] = out[t] * (1 - wt[..., None]) + w * wt[..., None]
        return out

    fwd = one_pass(generated, flow_fw, flow_bw)
    if not bidirectional:
        return fwd
    bwd = one_pass(generated, flow_fw, flow_bw, reverse=True)
    return ((fwd + bwd) * 0.5).clip(0, 1)


def interpolate_frames(frames: np.ndarray, flow_fw: np.ndarray, flow_bw: np.ndarray, factor: int = 2) -> np.ndarray:
    """Flow-based frame interpolation (RIFE can replace this via the
    restoration adapter). Inserts factor-1 frames between each pair."""
    out = []
    for t in range(len(frames) - 1):
        out.append(frames[t])
        for k in range(1, factor):
            a = k / factor
            # approximate backward flows from the intermediate time
            fa = warp(frames[t], flow_bw[t] * a)
            fb = warp(frames[t + 1], flow_fw[t] * (1 - a))
            out.append((1 - a) * fa + a * fb)
    out.append(frames[-1])
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------- chunking

@dataclass
class Chunk:
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def wan_frame_count(n: int) -> int:
    """Wan models want 4k+1 frames. Largest valid count <= n (min 1)."""
    return max(1, ((n - 1) // 4) * 4 + 1)


def plan_chunks(total: int, chunk_frames: int = 81, overlap: int = 9, align_4n1: bool = True, cuts: Optional[list] = None) -> list[Chunk]:
    """Overlapping windows covering [0,total). Scene cuts start a new chunk
    (no blending across a hard cut)."""
    if align_4n1:
        chunk_frames = wan_frame_count(chunk_frames)
    overlap = min(overlap, chunk_frames // 2)
    segments = []
    bounds = [0] + sorted(c for c in (cuts or []) if 0 < c < total) + [total]
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = a
        while True:
            e = min(s + chunk_frames, b)
            segments.append(Chunk(s, e))
            if e >= b:
                break
            s = e - overlap
    return segments


def merge_chunks(outputs: list[np.ndarray], chunks: list[Chunk], total: int) -> np.ndarray:
    """Linear cross-fade in overlaps; hard join where chunks do not overlap."""
    first = outputs[0]
    acc = np.zeros((total,) + first.shape[1:], np.float32)
    wsum = np.zeros((total,) + (1,) * (first.ndim - 1), np.float32)
    for i, (out, ch) in enumerate(zip(outputs, chunks)):
        n = min(len(out), ch.length)
        w = np.ones(n, np.float32)
        if i > 0 and chunks[i - 1].end > ch.start:
            ov = chunks[i - 1].end - ch.start
            w[:ov] = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]
        if i + 1 < len(chunks) and chunks[i + 1].start < ch.end:
            ov = ch.end - chunks[i + 1].start
            w[n - ov :] = np.minimum(w[n - ov :], np.linspace(1, 0, ov + 2, dtype=np.float32)[1:-1])
        shape = (n,) + (1,) * (first.ndim - 1)
        acc[ch.start : ch.start + n] += out[:n] * w.reshape(shape)
        wsum[ch.start : ch.start + n] += w.reshape(shape)
    return acc / np.maximum(wsum, 1e-6)
