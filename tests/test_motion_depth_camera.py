"""Optical flow, depth, camera."""
import cv2
import numpy as np

from synth import make_frames


def _shifted_pair(dx=3, dy=2, H=96, W=128, seed=1):
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.random((H + 20, W + 20, 3)).astype(np.float32), (0, 0), 2.0)
    big = (big - big.min()) / (big.max() - big.min())
    a = big[10 : 10 + H, 10 : 10 + W]
    b = big[10 - dy : 10 - dy + H, 10 - dx : 10 - dx + W]  # content moves by (+dx,+dy)
    return a, b


def test_flow_accuracy_global_translation():
    from genjutsu.motion import DISFlow

    a, b = _shifted_pair(3, 2)
    fl = DISFlow()(a, b)[10:-10, 10:-10]
    epe = np.linalg.norm(fl - np.array([3, 2]), axis=-1).mean()
    assert epe < 0.5, epe


def test_warp_aligns_frames():
    from genjutsu.motion import compute_flow, warp

    a, b = _shifted_pair(4, 0)
    fw, bw = compute_flow(np.stack([a, b]))
    w = warp(a, bw[0])
    err_before = np.abs(a - b)[8:-8, 8:-8].mean()
    err_after = np.abs(w - b)[8:-8, 8:-8].mean()
    assert err_after < 0.35 * err_before


def test_forward_backward_consistency_low_on_rigid_motion():
    from genjutsu.motion import compute_flow, fb_consistency

    a, b = _shifted_pair(2, 1)
    fw, bw = compute_flow(np.stack([a, b]))
    occ = fb_consistency(fw[0], bw[0])[10:-10, 10:-10]
    assert occ.mean() < 0.05


def test_depth_normalization_is_per_video():
    from genjutsu.depth import normalize_depth

    d = np.stack([np.full((8, 8), v, np.float32) for v in (1.0, 2.0, 3.0)])
    n = normalize_depth(d, 0, 100)
    assert n[0].max() < n[2].min()  # relative order across frames preserved


def test_depth_stabilization_removes_scale_flicker():
    from genjutsu.depth import stabilize_depth

    base = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (24, 1))
    rng = np.random.default_rng(0)
    seq = np.stack([base * s + o for s, o in zip(rng.uniform(0.6, 1.4, 16), rng.uniform(-0.2, 0.2, 16))])
    stab = stabilize_depth(seq, None, 0.5)
    spread = lambda x: x.mean((1, 2)).std()
    assert spread(stab) < 0.3 * spread(seq)


def test_layer_order_uses_depth():
    from genjutsu.depth import layer_order

    T, H, W = 2, 32, 32
    depth = np.zeros((T, H, W), np.float32)
    hand = np.zeros((T, H, W), np.float32)
    hand[:, 10:20, 5:25] = 1
    watch = np.zeros((T, H, W), np.float32)
    watch[:, 12:18, 12:18] = 1
    depth[0][hand[0] > 0] = 0.9  # frame 0: hand nearer
    depth[0][watch[0] > 0] = 0.3
    depth[1][hand[1] > 0] = 0.3  # frame 1: watch nearer
    depth[1][watch[1] > 0] = 0.9
    order = layer_order(depth, hand, watch)
    assert order[0].sum() > 0 and order[1].sum() == 0


def test_camera_detects_pan_and_static():
    from genjutsu.camera import estimate_camera

    frames, masks, _ = make_frames(T=24, pan=2.0, obj_speed=0.0)
    cam = estimate_camera(frames, masks)
    dx = np.array([p["dx"] for p in cam["per_frame"][1:]])
    assert abs(np.median(dx) + 2.0) < 0.3  # content moves left 2 px/frame
    assert "pan_right" in cam["summary"]["labels"]
    still, m2, _ = make_frames(T=12, pan=0.0, obj_speed=0.0)
    assert estimate_camera(still, m2)["summary"]["labels"] == ["static"]


def test_camera_similarity():
    from genjutsu.camera import camera_similarity, estimate_camera

    a, ma, _ = make_frames(T=16, pan=2.0, obj_speed=0.0)
    b, mb, _ = make_frames(T=16, pan=2.0, obj_speed=0.0, seed=5)  # different texture, same camera
    c, mc, _ = make_frames(T=16, pan=0.0, obj_speed=0.0)
    ca, cb, cc = estimate_camera(a, ma), estimate_camera(b, mb), estimate_camera(c, mc)
    assert camera_similarity(ca, cb) > 0.8
    assert camera_similarity(ca, cc) < camera_similarity(ca, cb)
