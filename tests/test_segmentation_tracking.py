"""Segmentation, preservation masks and tracking."""
import numpy as np

from conftest import iou
from synth import make_frames


def test_preservation_masks_core_and_complement(clip):
    from genjutsu.segmentation import build_preservation_masks

    _, masks, _ = clip
    edit, preserve, soft = build_preservation_masks(masks, None, dilate_px=4, feather_px=8)
    assert np.allclose(edit + preserve, 1.0)
    # the object itself is always fully generated
    assert soft[masks > 0.5].min() == 1.0
    # nothing outside the dilated edit is generated
    assert soft[preserve > 0.999].max() < 1e-6 or soft[edit < 0.5].max() < 0.5
    assert (edit >= masks).all()


def test_protect_mask_is_never_generated(clip):
    from genjutsu.segmentation import build_preservation_masks

    _, masks, _ = clip
    protect = np.zeros_like(masks)
    protect[:, 40:48, :] = 1  # a 'hand' crossing the object
    _, _, soft = build_preservation_masks(masks, protect, 4, 8)
    assert soft[protect > 0.5].max() == 0.0


def test_grabcut_accuracy(clip):
    from genjutsu.segmentation import grabcut

    frames, masks, boxes = clip
    x0, y0, x1, y1 = boxes[0]
    m = grabcut(frames[0], [x0 - 4, y0 - 4, x1 + 4, y1 + 4])
    assert iou(m, masks[0]) > 0.8


def test_segment_video_fallback_is_temporally_stable(clip):
    from genjutsu.segmentation import segment_video

    frames, masks, boxes = clip
    x0, y0, x1, y1 = boxes[0]
    out = segment_video(frames, box=[x0 - 3, y0 - 3, x1 + 3, y1 + 3], model="grabcut")
    per_frame = [iou(a, b) for a, b in zip(out, masks)]
    assert min(per_frame) > 0.7, per_frame
    # temporal stability: area should not jump between frames
    area = out.sum((1, 2))
    assert np.abs(np.diff(area)).max() / area.mean() < 0.25


def test_temporal_smoothing_reduces_mask_flicker(clip):
    from genjutsu.segmentation import temporal_smooth_masks

    _, masks, _ = clip
    rng = np.random.default_rng(0)
    noisy = masks.copy()
    for t in range(len(noisy)):  # random speckle per frame
        noisy[t][rng.random(noisy[t].shape) < 0.02] = 1
    smooth = (temporal_smooth_masks(noisy, None, 0.6) > 0.5).astype(np.float32)
    flick = lambda m: np.abs(np.diff(m, axis=0)).mean()
    assert flick(smooth) < flick(noisy)


def test_tracking_follows_moving_object(clip):
    from genjutsu.tracking import track

    frames, masks, _ = clip
    res = track(frames, masks[0])
    ious = [iou(a, b) for a, b in zip(res.masks, masks)]
    assert min(ious) > 0.7, ious
    assert all(res.visible)


def test_tracking_starts_mid_video_both_directions(clip):
    from genjutsu.tracking import track

    frames, masks, _ = clip
    res = track(frames, masks[12], start=12)
    assert iou(res.masks[0], masks[0]) > 0.6
    assert iou(res.masks[-1], masks[-1]) > 0.6


def test_tracking_disappearance_and_reidentification():
    """Object is hidden for 4 frames (occluder), then reappears elsewhere."""
    from genjutsu.tracking import track

    frames, masks, boxes = make_frames(T=24, H=96, W=128, obj_speed=1.5)
    frames = frames.copy()
    gt = masks.copy()
    # occlude frames 10..13 with a gray card
    for t in range(10, 14):
        x0, y0, x1, y1 = boxes[t]
        frames[t, y0 - 2 : y1 + 2, x0 - 2 : x1 + 2] = 0.5
        gt[t] = 0
    res = track(frames, masks[0])
    assert not any(res.visible[11:13])  # correctly reports it as gone
    assert iou(res.masks[20], gt[20]) > 0.5  # re-acquired after reappearance
    assert res.reidentified, "re-identification should have fired"
