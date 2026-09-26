"""Intermediate-result cache: every expensive stage is keyed by a hash of its
inputs + parameters and stored on disk, so a failed or re-run pipeline
resumes from the last completed stage (section 34)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

STAGES = ["frames", "masks", "tracking", "pose", "depth", "flow", "camera", "references", "generated", "temporal", "composite"]


def fingerprint(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(str(p.dtype).encode())
            a = np.ascontiguousarray(p)
            # sample large arrays: stable, fast, sensitive to real changes
            flat = a.reshape(-1)
            if flat.size > 4_000_000:
                idx = np.linspace(0, flat.size - 1, 1_000_000).astype(np.int64)
                h.update(flat[idx].tobytes())
            else:
                h.update(a.tobytes())
        elif isinstance(p, (str, Path)) and os.path.isfile(str(p)):
            st = os.stat(p)
            h.update(f"{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}".encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


class StageCache:
    def __init__(self, root: str | Path, project: str = "default", enabled: bool = True):
        self.root = Path(root) / project
        self.enabled = enabled
        if enabled:
            for s in STAGES:
                (self.root / s).mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"

    # -------------------------------------------------------------- state
    def state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"completed": {}, "chunks": {}}

    def _write_state(self, st: dict):
        if self.enabled:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=2))
            tmp.replace(self.state_path)

    def mark(self, stage: str, key: str, **info):
        st = self.state()
        st["completed"][stage] = {"key": key, "time": time.time(), **info}
        self._write_state(st)

    def mark_chunk(self, key: str, index: int, path: str):
        st = self.state()
        st["chunks"].setdefault(key, {})[str(index)] = path
        self._write_state(st)

    def done_chunks(self, key: str) -> dict:
        return {int(k): v for k, v in self.state().get("chunks", {}).get(key, {}).items() if Path(v).exists()}

    # -------------------------------------------------------------- arrays
    def path(self, stage: str, key: str, ext: str = "npz") -> Path:
        return self.root / stage / f"{key}.{ext}"

    def has(self, stage: str, key: str) -> bool:
        return self.enabled and (self.path(stage, key).exists() or self.path(stage, key, "json").exists())

    def save(self, stage: str, key: str, **arrays) -> Optional[Path]:
        if not self.enabled:
            return None
        p = self.path(stage, key)
        tmp = p.with_name(p.stem + ".tmp.npz")
        packed = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float32 and (k.endswith("mask") or k == "masks"):
                packed[k + "__u8"] = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)
            elif isinstance(v, np.ndarray) and v.dtype == np.float32 and k in ("frames", "generated", "composite"):
                packed[k + "__f16"] = v.astype(np.float16)
            elif isinstance(v, np.ndarray):
                packed[k] = v
            else:
                packed[k + "__json"] = np.frombuffer(json.dumps(v, default=str).encode(), np.uint8)
        np.savez_compressed(tmp, **packed)
        tmp.replace(p)
        self.mark(stage, key)
        return p

    def load(self, stage: str, key: str) -> dict:
        with np.load(self.path(stage, key), allow_pickle=False) as z:
            out = {}
            for k in z.files:
                if k.endswith("__u8"):
                    out[k[:-4]] = z[k].astype(np.float32) / 255.0
                elif k.endswith("__f16"):
                    out[k[:-5]] = z[k].astype(np.float32)
                elif k.endswith("__json"):
                    out[k[:-6]] = json.loads(z[k].tobytes().decode())
                else:
                    out[k] = z[k]
            return out

    def cached(self, stage: str, key: str, fn: Callable[[], dict]) -> dict:
        """Run fn() unless (stage,key) is cached; fn returns dict of arrays/json."""
        if self.has(stage, key):
            return self.load(stage, key)
        res = fn()
        self.save(stage, key, **res)
        return res
