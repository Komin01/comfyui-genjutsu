"""Camera motion analysis from background feature tracks: per-frame
similarity/homography, cumulative trajectory, zoom, rotation and shake."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .video import to_uint8


def _features(gray: np.ndarray, exclude: Optional[np.ndarray]) -> Optional[np.ndarray]:
    mask = None
    if exclude is not None:
        mask = ((exclude < 0.5) * 255).astype(np.uint8)
    return cv2.goodFeaturesToTrack(gray, maxCorners=800, qualityLevel=0.01, minDistance=8, mask=mask, blockSize=7)


def estimate_camera(frames: np.ndarray, exclude_masks: Optional[np.ndarray] = None, smooth_window: int = 9) -> dict:
    """Estimate the camera from background motion (subject masks excluded).

    Returns a JSON-friendly dict:
      per_frame: list of {dx, dy, rotation_deg, scale, homography, inliers}
      trajectory: cumulative [x, y, rotation_deg, zoom] per frame
      shake: per-frame high-frequency residual magnitude (px)
      summary: dominant motion labels (pan/tilt/zoom/roll/static/handheld)
    """
    T = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in to_uint8(frames)]
    per = [{"dx": 0.0, "dy": 0.0, "rotation_deg": 0.0, "scale": 1.0, "homography": np.eye(3).tolist(), "inliers": 0}]
    for t in range(1, T):
        ex = exclude_masks[t - 1] if exclude_masks is not None else None
        p0 = _features(grays[t - 1], ex)
        rec = {"dx": 0.0, "dy": 0.0, "rotation_deg": 0.0, "scale": 1.0, "homography": np.eye(3).tolist(), "inliers": 0}
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[t - 1], grays[t], p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
            if len(a) >= 8:
                M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None:
                    s = float(np.hypot(M[0, 0], M[1, 0]))
                    rec.update(
                        dx=float(M[0, 2]),
                        dy=float(M[1, 2]),
                        rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                        scale=s,
                        inliers=int(inl.sum()) if inl is not None else 0,
                    )
                Hm, _ = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
                if Hm is not None:
                    rec["homography"] = Hm.tolist()
        per.append(rec)

    traj = np.zeros((T, 4), np.float64)
    traj[0, 3] = 1.0
    for t in range(1, T):
        traj[t, 0] = traj[t - 1, 0] + per[t]["dx"]
        traj[t, 1] = traj[t - 1, 1] + per[t]["dy"]
        traj[t, 2] = traj[t - 1, 2] + per[t]["rotation_deg"]
        traj[t, 3] = traj[t - 1, 3] * per[t]["scale"]

    k = max(1, smooth_window | 1)
    pad = k // 2
    smooth = np.stack(
        [np.convolve(np.pad(traj[:, i], pad, mode="edge"), np.ones(k) / k, mode="valid") for i in range(3)], 1
    )
    shake = np.linalg.norm(traj[:, :2] - smooth[:, :2], axis=1)

    H, W = frames.shape[1:3]
    diag = float(np.hypot(W, H))
    total = traj[-1]
    labels = []
    if abs(total[0]) > 0.05 * W:
        labels.append("pan_right" if total[0] < 0 else "pan_left")  # background moves opposite to camera
    if abs(total[1]) > 0.05 * H:
        labels.append("tilt_down" if total[1] < 0 else "tilt_up")
    if total[3] > 1.05:
        labels.append("zoom_in")
    elif total[3] < 0.95:
        labels.append("zoom_out")
    if abs(total[2]) > 3:
        labels.append("roll")
    if float(np.mean(shake)) > 0.004 * diag:
        labels.append("handheld")
    if not labels:
        labels.append("static")
    return {
        "per_frame": per,
        "trajectory": traj.tolist(),
        "shake": shake.tolist(),
        "summary": {"labels": labels, "mean_shake_px": float(np.mean(shake)), "total_zoom": float(total[3])},
    }


def camera_similarity(cam_a: dict, cam_b: dict) -> float:
    """1.0 = identical camera paths. Compares per-frame translation/rotation/scale."""
    n = min(len(cam_a["per_frame"]), len(cam_b["per_frame"]))
    if n < 2:
        return 1.0
    def vec(c):
        return np.array([[p["dx"], p["dy"], p["rotation_deg"], (p["scale"] - 1) * 100] for p in c["per_frame"][:n]])
    a, b = vec(cam_a), vec(cam_b)
    err = np.linalg.norm(a - b, axis=1).mean()
    scale = np.linalg.norm(a, axis=1).mean() + 1.0
    return float(np.exp(-err / scale))
