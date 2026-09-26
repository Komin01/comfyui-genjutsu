"""Dependency installer and checker for ComfyUI-Genjutsu.

    python install.py                 # core requirements + report
    python install.py --check         # report only, install nothing
    python install.py --sam2 --pose   # add optional groups
    python install.py --all

ComfyUI-Manager runs this file automatically on install (core only)."""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

GROUPS = {
    "sam2": ["git+https://github.com/facebookresearch/sam2.git"],
    "pose": ["rtmlib", "onnxruntime-gpu" if sys.platform != "darwin" else "onnxruntime"],
    "depth": ["git+https://github.com/DepthAnything/Video-Depth-Anything.git"],
    "identity": ["insightface"],
    "cutout": ["rembg"],
}

CHECKS = [
    # (import name, what it enables, group)
    ("numpy", "core", None),
    ("cv2", "core", None),
    ("yaml", "config.yaml", None),
    ("PIL", "core", None),
    ("torch", "all model-backed stages (provided by ComfyUI)", None),
    ("torchvision", "RAFT optical flow", None),
    ("diffusers", "Wan 2.2 generation", None),
    ("transformers", "Grounding DINO text prompts, CLIP, Depth Anything V2", None),
    ("sam2", "SAM 2 video segmentation", "sam2"),
    ("rtmlib", "DWPose pose extraction", "pose"),
    ("video_depth_anything", "Video Depth Anything", "depth"),
    ("insightface", "Identity Lock (non-commercial weights)", "identity"),
    ("rembg", "reference cut-outs (GrabCut fallback otherwise)", "cutout"),
    ("spandrel", "Real-ESRGAN upscaling", None),
]


def pip(*args: str) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "install", *args])


def report() -> bool:
    ok = True
    print("\nGenjutsu dependency report")
    print("-" * 60)
    for mod, what, group in CHECKS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"  [ok]      {mod:<22}{ver:<12}{what}")
        except Exception:
            tag = "optional" if group else "MISSING"
            if mod in ("numpy", "cv2", "yaml", "PIL"):
                ok = False
            hint = f"  (install.py --{group})" if group else ""
            print(f"  [{tag}]{' ' * (8 - len(tag))} {mod:<22}{'':<12}{what}{hint}")
    ff = shutil.which("ffmpeg")
    print(f"  [{'ok' if ff else 'MISSING'}]{' ' * (8 - (2 if ff else 7))} {'ffmpeg':<22}{'':<12}video I/O  {ff or '-> https://ffmpeg.org/download.html'}")
    try:
        import diffusers

        if not hasattr(diffusers, "WanAnimatePipeline"):
            print("  [warn]    diffusers is too old for Wan-Animate; pip install -U 'diffusers>=0.40'")
    except Exception:
        pass
    print("-" * 60)
    return ok and bool(ff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    for g in GROUPS:
        ap.add_argument(f"--{g}", action="store_true")
    a = ap.parse_args()
    if not a.check:
        pip("-r", str(HERE / "requirements.txt"))
        for g, pkgs in GROUPS.items():
            if a.all or getattr(a, g):
                print(f"installing optional group '{g}'")
                pip(*pkgs)
    sys.exit(0 if report() else 1)


if __name__ == "__main__":
    main()
