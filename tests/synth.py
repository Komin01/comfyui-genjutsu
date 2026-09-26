"""Synthetic test media: textured background with a camera pan, a moving
square 'object', and optional audio track."""
import subprocess
import cv2
import numpy as np


def make_frames(T=24, H=96, W=128, seed=0, pan=1.0, obj_speed=1.5, obj=16):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.random((H, W * 2, 3)).astype(np.float32), (0, 0), 2.0)
    bg = (bg - bg.min()) / (bg.max() - bg.min()) * 0.7 + 0.15
    frames, masks, boxes = [], [], []
    for t in range(T):
        x = int(t * pan)
        f = bg[:, x : x + W].copy()
        ox, oy = int(20 + t * obj_speed), 40
        f[oy : oy + obj, ox : ox + obj] = (0.9, 0.1, 0.1)
        m = np.zeros((H, W), np.float32)
        m[oy : oy + obj, ox : ox + obj] = 1
        frames.append(f); masks.append(m); boxes.append((ox, oy, ox + obj, oy + obj))
    return np.stack(frames), np.stack(masks), boxes


def reference_image(size=64):
    img = np.ones((size, size, 3), np.float32)
    cv2.circle(img, (size // 2, size // 2), size // 3, (0.1, 0.2, 0.9), -1)
    return img


def write_with_audio(frames, path, fps=12):
    from genjutsu.video import write_video
    tmp = str(path) + ".noaudio.mp4"
    write_video(frames, tmp, fps, "h264", crf=10)
    dur = len(frames) / fps
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
    return str(path)
