"""Wan 2.x backends via diffusers (>= 0.40).

    wan22        T2V / I2V / TI2V, region edits and recasts (Wan VACE),
                 character ops delegated to Wan2.2-Animate
    wan_animate  Wan2.2-Animate only (motion transfer / character swap)

Model ids come from config.yaml `models:` so any compatible checkpoint
(e.g. a Wan2.2 VACE-Fun conversion, a distilled/LoRA-merged model) can be
dropped in without code changes.

Pipelines used (diffusers):
    WanPipeline, WanImageToVideoPipeline, WanVACEPipeline, WanAnimatePipeline
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import cv2
import numpy as np

from ..controls import ControlPackage, Operation, RefCategory
from ..references import reference_on_plain
from ..segmentation import dilate
from ..temporal import wan_frame_count
from ..video import resize_frames, resize_masks, to_uint8
from .base import CHARACTER_OPS, RECAST_OPS, REGION_OPS, VideoBackend, register_backend

log = logging.getLogger("genjutsu.wan")


def _pil_list(frames: np.ndarray):
    from PIL import Image

    if frames.ndim == 3:  # masks
        return [Image.fromarray(to_uint8(m)).convert("RGB") for m in frames]
    return [Image.fromarray(f) for f in to_uint8(frames)]


def _pil(img: np.ndarray):
    from PIL import Image

    return Image.fromarray(to_uint8(img))


def gen_size(width: int, height: int, max_side: int = 832, multiple: int = 16, max_area: Optional[int] = None) -> tuple[int, int]:
    """Aspect-preserving generation resolution, divisible by `multiple`."""
    s = max_side / max(width, height)
    if max_area:
        s = min(s, (max_area / (width * height)) ** 0.5)
    w = max(multiple, int(round(width * s / multiple)) * multiple)
    h = max(multiple, int(round(height * s / multiple)) * multiple)
    return w, h


def pad_4n1(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad along time (repeat last frame) up to the next 4k+1 length."""
    n = len(x)
    target = n if (n - 1) % 4 == 0 else wan_frame_count(n) + 4
    if target == n:
        return x, n
    pad = np.repeat(x[-1:], target - n, 0)
    return np.concatenate([x, pad], 0), n


MODEL_KEYS = {"t2v": "wan22_t2v", "i2v": "wan22_i2v", "ti2v": "wan22_ti2v", "vace": "wan_vace", "animate": "wan_animate"}


@register_backend("wan22")
class Wan22Backend(VideoBackend):
    supports = REGION_OPS | CHARACTER_OPS | RECAST_OPS

    def __init__(self, config):
        super().__init__(config)
        self._pipes: dict = {}

    # ---------------------------------------------------------------- loading
    @property
    def torch(self):
        import torch

        return torch

    def _dtype(self):
        return {"bfloat16": self.torch.bfloat16, "float16": self.torch.float16, "float32": self.torch.float32}[self.config.get("dtype", "bfloat16")]

    def _load(self, kind: str):
        if kind in self._pipes:
            return self._pipes[kind]
        if self.config.get("performance.low_vram", False) or len(self._pipes) >= 1:
            self.unload()  # 14B-class pipelines rarely fit together
        import diffusers
        from diffusers import AutoencoderKLWan

        mid = self.config.get("models." + MODEL_KEYS[kind])
        cls = {
            "t2v": diffusers.WanPipeline,
            "ti2v": diffusers.WanPipeline,
            "i2v": diffusers.WanImageToVideoPipeline,
            "vace": diffusers.WanVACEPipeline,
            "animate": diffusers.WanAnimatePipeline,
        }[kind]
        log.info("loading %s (%s)", mid, cls.__name__)
        kw = {"torch_dtype": self._dtype()}
        if kind != "animate":
            kw["vae"] = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=self.torch.float32)
        pipe = cls.from_pretrained(mid, **kw)
        if kind == "animate":
            pipe.vae.to(self.torch.float32)
        device = self.config.get("device", "cuda")
        if self.config.get("performance.low_vram", False):
            pipe.enable_sequential_cpu_offload()
        elif self.config.get("performance.cpu_offload", True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if self.config.get("performance.tile_size", 0):
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
        self._pipes[kind] = pipe
        return pipe

    def unload(self):
        self._pipes.clear()
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _gen(self, seed: int):
        return self.torch.Generator(device="cpu").manual_seed(int(seed))

    def _size(self, cp_or_wh) -> tuple[int, int]:
        W, H = cp_or_wh if isinstance(cp_or_wh, tuple) else cp_or_wh.size
        return gen_size(W, H, self.config.get("generation.max_side", 832), 16, self.config.get("generation.max_area", 832 * 480))

    @staticmethod
    def _back(out: np.ndarray, cp: ControlPackage, n: int) -> np.ndarray:
        out = np.asarray(out, np.float32)[:n]
        W, H = cp.size
        return resize_frames(out, W, H, cv2.INTER_LANCZOS4).clip(0, 1)

    # ---------------------------------------------------------------- basic
    def generate_t2v(self, prompt, negative_prompt, width, height, num_frames, seed=0, steps=40, guidance=4.0, model="t2v", **kw):
        pipe = self._load(model)
        w, h = self._size((width, height))
        out = pipe(prompt=prompt, negative_prompt=negative_prompt, width=w, height=h, num_frames=wan_frame_count(num_frames),
                   num_inference_steps=steps, guidance_scale=guidance, generator=self._gen(seed), output_type="np").frames[0]
        return resize_frames(np.asarray(out, np.float32), width, height, cv2.INTER_LANCZOS4)

    def generate_i2v(self, image, prompt, negative_prompt, num_frames, seed=0, steps=40, guidance=3.5, last_image=None, **kw):
        pipe = self._load("i2v")
        H, W = image.shape[:2]
        w, h = self._size((W, H))
        img = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        extra = {"last_image": _pil(cv2.resize(last_image, (w, h)))} if last_image is not None else {}
        out = pipe(image=_pil(img), prompt=prompt, negative_prompt=negative_prompt, width=w, height=h,
                   num_frames=wan_frame_count(num_frames), num_inference_steps=steps, guidance_scale=guidance,
                   generator=self._gen(seed), output_type="np", **extra).frames[0]
        return resize_frames(np.asarray(out, np.float32), W, H, cv2.INTER_LANCZOS4)

    # ---------------------------------------------------------------- VACE
    def _vace(self, cp: ControlPackage, video: np.ndarray, mask: np.ndarray, refs: list, scale: float) -> np.ndarray:
        pipe = self._load("vace")
        w, h = self._size(cp)
        video, n = pad_4n1(resize_frames(video, w, h))
        mask, _ = pad_4n1(resize_masks(mask, w, h))
        ref_imgs = [_pil(r) for r in refs[:3]] or None
        out = pipe(
            prompt=cp.prompt, negative_prompt=cp.negative_prompt, video=_pil_list(video), mask=_pil_list((mask > 0.5).astype(np.float32)),
            reference_images=ref_imgs, conditioning_scale=float(scale), height=h, width=w, num_frames=len(video),
            num_inference_steps=cp.steps, guidance_scale=cp.guidance, generator=self._gen(cp.seed), output_type="np",
        ).frames[0]
        return self._back(out, cp, n)

    def generate_region(self, cp: ControlPackage) -> np.ndarray:
        if cp.edit_mask is None:
            raise ValueError("Region edit needs an edit mask (run the Segmenter / Preservation nodes)")
        refs = cp.references.by_category(RefCategory.PRODUCT, RefCategory.OBJECT, RefCategory.PROP, RefCategory.OUTFIT, RefCategory.LOGO)
        refs = sorted(refs, key=lambda r: -r.weight)
        m = cp.edit_mask
        # VACE inpainting convention: region to generate is neutral gray in the source video
        video = cp.frames * (1 - m[..., None]) + 0.5 * m[..., None]
        return self._vace(cp, video, m, [reference_on_plain(r) for r in refs], cp.strengths.preservation)

    def _control_video(self, cp: ControlPackage) -> np.ndarray:
        kind = cp.extras.get("recast_control", "depth")
        if kind == "depth" and cp.depth is not None:
            return np.repeat(cp.depth[..., None], 3, -1)
        if kind in ("pose", "depth") and cp.pose_frames is not None:
            return cp.pose_frames
        # fallback: structure from edges (camera/layout survive, appearance doesn't)
        edges = np.stack([cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 60, 140) for f in to_uint8(cp.frames)])
        return np.repeat((edges.astype(np.float32) / 255.0)[..., None], 3, -1)

    def generate_recast(self, cp: ControlPackage) -> np.ndarray:
        op = Operation(cp.operation)
        control = self._control_video(cp)
        T, H, W = cp.frames.shape[:3]
        cats = {
            Operation.ENVIRONMENT_RECAST: (RefCategory.LOCATION, RefCategory.STYLE),
            Operation.STYLE_RECAST: (RefCategory.STYLE, RefCategory.TEXTURE),
            Operation.FULL_RECAST: (RefCategory.CHARACTER, RefCategory.LOCATION, RefCategory.STYLE, RefCategory.OBJECT, RefCategory.PRODUCT),
        }[op]
        refs = sorted(cp.references.by_category(*cats), key=lambda r: -r.weight)
        if op == Operation.ENVIRONMENT_RECAST and cp.subject_mask is not None:
            keep = dilate(cp.subject_mask, 2)
            video = cp.frames * keep[..., None] + control * (1 - keep[..., None])
            mask = 1.0 - keep
        else:
            video, mask = control, np.ones((T, H, W), np.float32)
        return self._vace(cp, video, mask, [reference_on_plain(r) if r.category != RefCategory.LOCATION else r.image for r in refs],
                          cp.strengths.motion)

    # ---------------------------------------------------------------- Animate
    def chunks_internally(self, op) -> bool:
        # VACE paths are chunked by the pipeline; Wan-Animate segments long clips itself
        return Operation(op) in CHARACTER_OPS

    def generate_character(self, cp: ControlPackage) -> np.ndarray:
        if cp.pose_frames is None or cp.face_frames is None:
            raise ValueError("Character operations need pose and face conditioning (run the Pose Extractor node)")
        ref = cp.references.primary(RefCategory.CHARACTER)
        if ref is None:
            raise ValueError("Character operations need a CHARACTER reference")
        pipe = self._load("animate")
        w, h = self._size(cp)
        T = cp.num_frames
        replace = Operation(cp.operation) == Operation.CHARACTER_SWAP and "background" in cp.invariants
        kw = {}
        if replace:
            if cp.subject_mask is None:
                raise ValueError("Character swap with background preservation needs a subject mask")
            m = dilate(cp.subject_mask, 8)
            bg = cp.frames * (1 - m[..., None])  # masked-out performer, as in Wan-Animate preprocessing
            kw = {"background_video": _pil_list(resize_frames(bg, w, h)), "mask_video": _pil_list(resize_masks(m, w, h))}
        seg = wan_frame_count(self.config.get("chunking.chunk_frames", 77))
        out = pipe(
            image=_pil(ref.image), pose_video=_pil_list(resize_frames(cp.pose_frames, w, h)), face_video=_pil_list(cp.face_frames),
            prompt=cp.prompt, negative_prompt=cp.negative_prompt, height=h, width=w, segment_frame_length=seg,
            prev_segment_conditioning_frames=int(self.config.get("generation.animate_prev_frames", 5)),
            num_inference_steps=int(self.config.get("generation.animate_steps", 20)), guidance_scale=1.0,
            mode="replace" if replace else "animate", generator=self._gen(cp.seed), output_type="np", **kw,
        ).frames[0]
        return self._back(out, cp, T)


@register_backend("wan_animate")
class WanAnimateBackend(Wan22Backend):
    supports = CHARACTER_OPS
    internal_chunking = True
