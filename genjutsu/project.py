"""The .genjutsu project file: a zip holding everything needed to reproduce
a result (source reference + checksum, references, masks, controls, prompt,
model settings, generation parameters, processing state)."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import __version__
from .controls import ControlPackage, Operation, RefCategory, Reference, ReferencePackage, Strengths, TemporalSettings
from .video import to_uint8

FORMAT_VERSION = 1


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _png(img: np.ndarray) -> bytes:
    if img.ndim == 3:
        img = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    else:
        img = to_uint8(img)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _unpng(b: bytes, color: bool = True) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if color:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr.astype(np.float32) / 255.0


def save_project(
    path: str,
    cp: ControlPackage,
    settings: Optional[dict] = None,
    state: Optional[dict] = None,
    embed_source: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Write a .genjutsu file. By default the source video is referenced by
    path + SHA-256 (videos are large); embed_source=True copies it in."""
    path = str(path)
    if not path.endswith(".genjutsu"):
        path += ".genjutsu"
    refs_meta = []
    manifest = {
        "format": FORMAT_VERSION,
        "genjutsu_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {
            "path": cp.source_path,
            "sha256": sha256_file(cp.source_path) if cp.source_path and Path(cp.source_path).is_file() else None,
            "embedded": bool(embed_source and cp.source_path),
            "fps": cp.fps,
            "frames": cp.num_frames,
            "size": list(cp.size),
        },
        "operation": Operation(cp.operation).value,
        "prompt": cp.prompt,
        "negative_prompt": cp.negative_prompt,
        "invariants": cp.invariants,
        "temporal": asdict(cp.temporal),
        "strengths": asdict(cp.strengths),
        "generation": {"seed": cp.seed, "steps": cp.steps, "guidance": cp.guidance},
        "camera_summary": (cp.camera or {}).get("summary"),
        "scene_cuts": cp.scene_cuts,
        "settings": settings or {},
        "state": state or {},
        "output": output_path,
        "references": refs_meta,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(cp.references.references):
            name = f"references/{i:02d}_{r.category.value}"
            z.writestr(name + ".png", _png(r.image))
            if r.mask is not None:
                z.writestr(name + "_mask.png", _png(r.mask))
            if r.embedding is not None:
                bio = io.BytesIO()
                np.save(bio, r.embedding)
                z.writestr(name + "_emb.npy", bio.getvalue())
            refs_meta.append(
                {"file": name + ".png", "category": r.category.value, "weight": r.weight, "name": r.name,
                 "has_mask": r.mask is not None, "has_embedding": r.embedding is not None,
                 "metadata": {k: v for k, v in r.metadata.items() if k != "color_signature"}}
            )
        masks = {}
        for k in ("edit_mask", "preserve_mask", "soft_mask", "subject_mask"):
            v = getattr(cp, k)
            if v is not None:
                masks[k] = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)
        if masks:
            bio = io.BytesIO()
            np.savez_compressed(bio, **masks)
            z.writestr("controls/masks.npz", bio.getvalue())
        if cp.depth is not None:
            bio = io.BytesIO()
            np.savez_compressed(bio, depth=(cp.depth * 65535).astype(np.uint16))
            z.writestr("controls/depth.npz", bio.getvalue())
        if cp.camera is not None:
            z.writestr("controls/camera.json", json.dumps(cp.camera))
        if embed_source and cp.source_path:
            z.write(cp.source_path, "source/" + Path(cp.source_path).name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return path


def load_project(path: str, frames: Optional[np.ndarray] = None, extract_dir: Optional[str] = None) -> tuple[ControlPackage, dict]:
    """Rebuild a ControlPackage. If `frames` is None the source video is
    decoded (from the embedded copy, else the recorded path, verified by hash)."""
    from .video import load_frames

    with zipfile.ZipFile(path) as z:
        man = json.loads(z.read("manifest.json"))
        src = man["source"]
        src_path = src["path"]
        if src.get("embedded"):
            names = [n for n in z.namelist() if n.startswith("source/")]
            out_dir = Path(extract_dir or Path(path).with_suffix(".src"))
            out_dir.mkdir(parents=True, exist_ok=True)
            z.extract(names[0], out_dir)
            src_path = str(out_dir / names[0])
        elif src.get("sha256") and Path(src_path).is_file() and sha256_file(src_path) != src["sha256"]:
            man.setdefault("warnings", []).append("source video changed since the project was saved")
        if frames is None:
            frames, _ = load_frames(src_path)
        refs = ReferencePackage()
        for r in man["references"]:
            base = r["file"][:-4]
            img = _unpng(z.read(r["file"]))
            mask = _unpng(z.read(base + "_mask.png"), color=False) if r["has_mask"] else None
            emb = np.load(io.BytesIO(z.read(base + "_emb.npy"))) if r["has_embedding"] else None
            refs = refs.add(Reference(img, RefCategory(r["category"]), r["weight"], r["name"], mask, emb, r.get("metadata", {})))
        masks = {}
        if "controls/masks.npz" in z.namelist():
            with np.load(io.BytesIO(z.read("controls/masks.npz"))) as m:
                masks = {k: m[k].astype(np.float32) / 255.0 for k in m.files}
        depth = None
        if "controls/depth.npz" in z.namelist():
            with np.load(io.BytesIO(z.read("controls/depth.npz"))) as d:
                depth = d["depth"].astype(np.float32) / 65535.0
        camera = json.loads(z.read("controls/camera.json")) if "controls/camera.json" in z.namelist() else None
    gen = man["generation"]
    cp = ControlPackage(
        frames=frames, fps=src["fps"], operation=Operation(man["operation"]), source_path=src_path,
        prompt=man["prompt"], negative_prompt=man["negative_prompt"], depth=depth, camera=camera,
        scene_cuts=man.get("scene_cuts", []), references=refs, invariants=man["invariants"],
        temporal=TemporalSettings(**man["temporal"]), strengths=Strengths(**man["strengths"]),
        seed=gen["seed"], steps=gen["steps"], guidance=gen["guidance"], **masks,
    )
    return cp, man
