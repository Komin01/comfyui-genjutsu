"""Video I/O through FFmpeg: probing, frame extraction, scene cuts, encoding
with audio passthrough."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FFmpegMissing(RuntimeError):
    pass


def _bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FFmpegMissing(f"{name} not found on PATH. Install FFmpeg (https://ffmpeg.org/download.html).")
    return p


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None
    pix_fmt: Optional[str] = None
    rotation: int = 0
    raw: dict = field(default_factory=dict, repr=False)


def _parse_rate(r: str) -> float:
    if not r or r == "0/0":
        return 0.0
    if "/" in r:
        n, d = r.split("/")
        return float(n) / float(d) if float(d) else 0.0
    return float(r)


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(out.stdout)
    vs = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise ValueError(f"No video stream in {path}")
    aus = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    fps = _parse_rate(vs.get("avg_frame_rate")) or _parse_rate(vs.get("r_frame_rate")) or 24.0
    duration = float(vs.get("duration") or meta.get("format", {}).get("duration") or 0.0)
    n = int(vs.get("nb_frames") or 0) or int(round(duration * fps))
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    return VideoInfo(
        path=str(path),
        width=int(vs["width"]),
        height=int(vs["height"]),
        fps=fps,
        frame_count=n,
        duration=duration,
        has_audio=aus is not None,
        audio_codec=aus.get("codec_name") if aus else None,
        video_codec=vs.get("codec_name"),
        pix_fmt=vs.get("pix_fmt"),
        rotation=rot,
        raw=meta,
    )


def load_frames(
    path: str,
    start_frame: int = 0,
    max_frames: int = 0,
    stride: int = 1,
    max_side: int = 0,
) -> tuple[np.ndarray, VideoInfo]:
    """Decode frames as float32 RGB (T,H,W,3) in [0,1]. Uses ffmpeg so that
    rotation metadata and odd pixel formats are handled."""
    info = probe(path)
    w, h = info.width, info.height
    if abs(info.rotation) in (90, 270):
        w, h = h, w
    vf = []
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
        vf.append(f"scale={w}:{h}:flags=lanczos")
    sel = []
    if start_frame:
        sel.append(f"gte(n\\,{start_frame})")
    if stride > 1:
        sel.append(f"not(mod(n-{start_frame}\\,{stride}))")
    if sel:
        vf.insert(0, "select='" + "*".join(sel) + "'")
    cmd = [_bin("ffmpeg"), "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if sel:
        cmd += ["-vsync", "0"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8)
    n = arr.size // (w * h * 3)
    frames = arr[: n * w * h * 3].reshape(n, h, w, 3).astype(np.float32) / 255.0
    info.width, info.height = w, h
    if stride > 1:
        info.fps = info.fps / stride
    return frames, info


def to_uint8(frames: np.ndarray) -> np.ndarray:
    return (np.clip(frames, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


CODECS = {
    "h264": (["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow"], "mp4"),
    "h265": (["-c:v", "libx265", "-pix_fmt", "yuv420p", "-preset", "medium", "-tag:v", "hvc1"], "mp4"),
    "prores": (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "mov"),
    "vp9": (["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-row-mt", "1"], "webm"),
    "png": ([], "png"),
}


def write_video(
    frames: np.ndarray,
    out_path: str,
    fps: float,
    codec: str = "h264",
    crf: int = 16,
    audio_from: Optional[str] = None,
    audio_offset_s: float = 0.0,
) -> str:
    """Encode frames; optionally copy/transcode the audio track of another file.
    codec='png' writes an image sequence directory instead."""
    out_path = str(out_path)
    frames8 = to_uint8(frames)
    if frames8.ndim != 4:
        raise ValueError("frames must be (T,H,W,3)")
    T, H, W, _ = frames8.shape
    if codec == "png":
        d = Path(out_path)
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames8):
            cv2.imwrite(str(d / f"frame_{i:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        return str(d)
    if codec not in CODECS:
        raise ValueError(f"Unknown codec {codec}; choose from {list(CODECS)}")
    vargs, _ = CODECS[codec]
    if codec in ("h264", "h265", "vp9") and (W % 2 or H % 2):
        frames8 = frames8[:, : H - H % 2, : W - W % 2]
        T, H, W, _ = frames8.shape
    cmd = [_bin("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    has_audio = False
    if audio_from:
        try:
            has_audio = probe(audio_from).has_audio
        except Exception:
            has_audio = False
    if has_audio:
        if audio_offset_s:
            cmd += ["-ss", f"{audio_offset_s}"]
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
    cmd += vargs
    if codec in ("h264", "h265", "vp9"):
        cmd += ["-crf", str(crf)]
    if has_audio:
        acodec = {"vp9": ["-c:a", "libopus"], "prores": ["-c:a", "pcm_s16le"]}.get(codec, ["-c:a", "aac", "-b:a", "192k"])
        cmd += acodec
    cmd += [out_path]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, input=frames8.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return out_path


def detect_scene_cuts(frames: np.ndarray, threshold: float = 0.45, min_gap: int = 8, structure_threshold: float = 0.5) -> list[int]:
    """Indices t where a hard cut occurs between t-1 and t.

    Two cues: HSV histogram distance (Bhattacharyya) catches palette changes;
    a structural cue (1 - normalized cross-correlation of 32x32 gray
    thumbnails, required to spike well above the recent median) catches cuts
    between shots with a similar palette while ignoring exposure changes and
    ordinary motion."""
    cuts: list[int] = []
    prev_hist, prev_small = None, None
    recent: list[float] = []
    for t, f in enumerate(to_uint8(frames)):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        small = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if prev_hist is not None:
            dh = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            a, b = small - small.mean(), prev_small - prev_small.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
            dp = 1.0 - ncc
            base = float(np.median(recent[-12:])) if recent else 0.0
            is_cut = dh > threshold or (dp > structure_threshold and dp > 3 * base + 0.1)
            if is_cut and (not cuts or t - cuts[-1] >= min_gap):
                cuts.append(t)
                recent.clear()
            else:
                recent.append(dp)
        prev_hist, prev_small = hist, small
    return cuts


def resize_frames(frames: np.ndarray, width: int, height: int, interp=cv2.INTER_AREA) -> np.ndarray:
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    return np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in frames]).astype(frames.dtype)


def resize_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    if masks.shape[2] == width and masks.shape[1] == height:
        return masks
    return np.stack([cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR) for m in masks]).astype(np.float32)
