"""Temporal engine, compositing, color, restoration."""
import numpy as np

from synth import make_frames


def _flicker(frames, amp=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return (frames + rng.uniform(-amp, amp, (len(frames), 1, 1, 1))).clip(0, 1).astype(np.float32)


def test_flicker_score_detects_flicker(clip):
    from genjutsu.motion import compute_flow
    from genjutsu.temporal import flicker_score

    frames, _, _ = clip
    fw, bw = compute_flow(frames)
    assert flicker_score(frames, fw, bw) > flicker_score(_flicker(frames), fw, bw)


def test_deflicker_luminance(clip):
    from genjutsu.temporal import deflicker_luminance

    frames, _, _ = clip
    bad = _flicker(frames, 0.1)
    fixed = deflicker_luminance(bad, window=9, strength=1.0)
    jitter = lambda x: np.abs(np.diff(x.mean((1, 2, 3)))).mean()
    assert jitter(fixed) < 0.5 * jitter(bad)


def test_temporal_repair_reduces_texture_boiling(clip):
    """Simulate diffusion 'boiling': independent small noise per frame."""
    from genjutsu.motion import compute_flow
    from genjutsu.temporal import temporal_repair, warp_error

    frames, _, _ = clip
    fw, bw = compute_flow(frames)
    rng = np.random.default_rng(1)
    boiled = (frames + rng.normal(0, 0.03, frames.shape)).clip(0, 1).astype(np.float32)
    fixed = temporal_repair(boiled, fw, bw, strength=0.7, threshold=0.08)
    assert warp_error(fixed, fw, bw).mean() < 0.8 * warp_error(boiled, fw, bw).mean()
    # must not destroy the content
    assert np.abs(fixed - frames).mean() < np.abs(boiled - frames).mean()


def test_detect_flicker_frames(clip):
    from genjutsu.motion import compute_flow
    from genjutsu.temporal import detect_flicker_frames

    frames, _, _ = clip
    fw, bw = compute_flow(frames)
    bad = frames.copy()
    bad[15] = (bad[15] * 0.6).astype(np.float32)
    assert 15 in detect_flicker_frames(bad, fw, bw)


def test_chunk_plan_covers_video_with_overlap_and_cuts():
    from genjutsu.temporal import plan_chunks

    ch = plan_chunks(240, 81, 9)
    assert ch[0].start == 0 and ch[-1].end == 240
    for a, b in zip(ch, ch[1:]):
        assert b.start == a.end - 9
    assert all((c.length - 1) % 4 == 0 for c in ch[:-1])  # Wan 4n+1
    cut = plan_chunks(200, 81, 9, cuts=[100])
    assert any(c.start == 100 for c in cut)
    assert not any(c.start < 100 < c.end for c in cut)  # never blend across a cut


def test_merge_chunks_reconstructs_identity():
    from genjutsu.temporal import merge_chunks, plan_chunks

    T = 100
    video = np.random.default_rng(0).random((T, 4, 4, 3)).astype(np.float32)
    ch = plan_chunks(T, 33, 8)
    outs = [video[c.start : c.end] for c in ch]
    assert np.allclose(merge_chunks(outs, ch, T), video, atol=1e-5)


def test_interpolation_doubles_frames(clip):
    from genjutsu.motion import compute_flow
    from genjutsu.temporal import interpolate_frames

    frames, _, _ = clip
    fw, bw = compute_flow(frames)
    out = interpolate_frames(frames, fw, bw, 2)
    assert len(out) == 2 * len(frames) - 1
    assert np.allclose(out[::2], frames)


def test_composite_no_leakage_and_smooth_boundary(clip):
    from genjutsu.compositing import composite, mask_leakage
    from genjutsu.segmentation import build_preservation_masks

    frames, masks, _ = clip
    gen = np.zeros_like(frames)
    gen[..., 1] = 1.0  # bright green generation everywhere
    edit, preserve, soft = build_preservation_masks(masks, None, 6, 12)
    out = composite(frames, gen, soft, preserve, edit, mode="multiband", color_match_strength=0.0, local_luminance=0.0)
    assert mask_leakage(frames, out, preserve) == 0.0
    # inside the object: generated content
    assert out[masks > 0.5][:, 1].mean() > 0.95
    # boundary artifacts: no ringing values outside [min,max] of the two sources
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_alpha_and_poisson_modes_run(clip):
    from genjutsu.compositing import composite
    from genjutsu.segmentation import build_preservation_masks

    frames, masks, _ = clip
    gen = frames[:, :, ::-1].copy()
    e, p, s = build_preservation_masks(masks, None)
    for mode in ("alpha", "poisson"):
        out = composite(frames[:4], gen[:4], s[:4], p[:4], e[:4], mode=mode)
        assert out.shape == frames[:4].shape


def test_color_match_offset_preserves_object_chroma():
    import cv2

    from genjutsu.compositing import color_match

    T, H, W = 3, 32, 32
    orig = np.full((T, H, W, 3), (0.8, 0.7, 0.5), np.float32)  # warm scene
    gen = np.full((T, H, W, 3), (0.6, 0.6, 0.6), np.float32)
    gen[:, 8:24, 8:24] = (0.1, 0.2, 0.9)  # saturated blue product
    ring = np.ones((T, H, W), np.float32)
    ring[:, 4:28, 4:28] = 0
    out = color_match(gen, orig, ring, ring, 1.0, ab_mode="offset")
    lab = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    # product stays blue (b channel strongly negative), surroundings become warm
    assert lab(out[0])[16, 16, 2] < -40
    assert lab(out[0])[2, 2, 2] > lab(gen[0])[2, 2, 2]


def test_occlusion_protect_keeps_nearer_hand():
    from genjutsu.compositing import occlusion_protect

    T, H, W = 1, 40, 40
    watch = np.zeros((T, H, W), np.float32)
    watch[:, 15:25, 15:25] = 1
    hand = np.zeros_like(watch)
    hand[:, 18:22, 5:35] = 1
    depth = np.full((T, H, W), 0.2, np.float32)
    depth[watch > 0] = 0.5
    depth[hand > 0] = 0.9
    prot = occlusion_protect(depth, watch, [hand])
    assert prot[0, 20, 20] == 1.0 and prot[0, 5, 5] == 0.0


def test_restoration_upscale_and_sharpen_limits(clip):
    from genjutsu.restoration import restore, sharpen

    frames, _, _ = clip
    up = restore(frames[:2], scale=2.0)
    assert up.shape[1:3] == (frames.shape[1] * 2, frames.shape[2] * 2)
    # amount is capped: asking for absurd sharpening can't blow up the image
    s = sharpen(frames[:2], amount=5.0)
    assert np.abs(s - frames[:2]).max() < 0.5
