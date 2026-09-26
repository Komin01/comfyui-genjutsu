"""End-to-end orchestration:

SOURCE VIDEO -> UNDERSTAND -> EXTRACT CONTROLS -> CONTROL PACKAGE
-> GENERATE (chunked, cached, resumable) -> TEMPORAL REPAIR -> COMPOSITE
-> RESTORE / UPSCALE -> ENCODE (+audio)

Each stage is usable on its own (the ComfyUI nodes call them individually);
`GenjutsuPipeline.run` chains them for scripts and the one-node workflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .backends import get_backend
from .backends.base import CHARACTER_OPS, RECAST_OPS, REGION_OPS
from .cache import StageCache, fingerprint
from .config import GenjutsuConfig, load_config
from .controls import ControlPackage, Operation, RefCategory, ReferencePackage, Strengths, TemporalSettings
from .prompt import PromptSpec, compile_prompt
from .temporal import merge_chunks, plan_chunks

log = logging.getLogger("genjutsu.pipeline")

Progress = Optional[Callable[[str, float], None]]


@dataclass
class RunResult:
    frames: np.ndarray
    control: ControlPackage
    generated: np.ndarray
    output_path: Optional[str] = None
    project_path: Optional[str] = None
    report: dict = field(default_factory=dict)


class GenjutsuPipeline:
    def __init__(self, config: Optional[GenjutsuConfig] = None, project: str = "default", progress: Progress = None):
        self.config = config or load_config()
        self.cache = StageCache(self.config.get("cache.root"), project, self.config.get("cache.enabled", True))
        self.progress = progress or (lambda stage, frac: None)
        self.device = self.config.get("device", "cuda")

    # ------------------------------------------------------------ load
    def load(self, source: str, max_frames: int = 0, start_frame: int = 0, stride: int = 1, max_side: int = 0) -> ControlPackage:
        key = fingerprint(source, max_frames, start_frame, stride, max_side)
        self.progress("load", 0.0)
        res = self.cache.cached("frames", key, lambda: self._load(source, max_frames, start_frame, stride, max_side))
        self.progress("load", 1.0)
        return ControlPackage(frames=res["frames"], fps=float(res["meta"]["fps"]), source_path=str(source))

    @staticmethod
    def _load(source, max_frames, start_frame, stride, max_side):
        from .video import load_frames

        frames, info = load_frames(source, start_frame, max_frames, stride, max_side)
        return {"frames": frames, "meta": {"fps": info.fps, "has_audio": info.has_audio, "width": info.width, "height": info.height}}

    # ------------------------------------------------------------ understanding
    def analyze(self, cp: ControlPackage, flow: bool = True, camera: bool = True, cuts: bool = True,
                depth: bool = False, pose: bool = False) -> ControlPackage:
        base = fingerprint(cp.frames)
        if cuts:
            from .video import detect_scene_cuts

            cp.scene_cuts = detect_scene_cuts(cp.frames)
        if flow:
            self.progress("flow", 0.0)
            res = self.cache.cached("flow", base + "_" + self.config.get("optical_flow.model"), lambda: self._flow(cp.frames))
            cp.flow_fw, cp.flow_bw = res["fw"], res["bw"]
        if camera:
            from .camera import estimate_camera

            cp.camera = self.cache.cached("camera", base + fingerprint(cp.subject_mask),
                                          lambda: {"camera": estimate_camera(cp.frames, cp.subject_mask)})["camera"]
        if depth:
            self.progress("depth", 0.0)
            from .depth import estimate_depth

            cp.depth = self.cache.cached(
                "depth", base + "_" + self.config.get("depth.model"),
                lambda: {"depth": estimate_depth(cp.frames, self.config.get("depth.model"), self.config.get("depth.checkpoint"),
                                                 cp.fps, self.device, cp.flow_bw)},
            )["depth"]
        if pose:
            self.progress("pose", 0.0)
            from .pose import extract_pose

            def _pose():
                pr = extract_pose(cp.frames, self.device)
                return {"rendered": pr.rendered.astype(np.float16), "faces": pr.face_crops.astype(np.float16)}

            res = self.cache.cached("pose", base, _pose)
            cp.pose_frames, cp.face_frames = res["rendered"].astype(np.float32), res["faces"].astype(np.float32)
        return cp

    def _flow(self, frames):
        from .motion import compute_flow, make_flow_estimator

        est = make_flow_estimator(self.config.get("optical_flow.model"), self.config.get("optical_flow.fallback"), self.device)
        fw, bw = compute_flow(frames, est, max_side=self.config.get("optical_flow.max_side", 960))
        return {"fw": fw, "bw": bw}

    def select(self, cp: ControlPackage, text: str = "", box=None, points=None, prompt_frame: int = 0) -> np.ndarray:
        from .segmentation import segment_video

        key = fingerprint(cp.frames, text, box, points, prompt_frame, self.config.get("segmentation.model"))
        res = self.cache.cached(
            "masks", key,
            lambda: {"mask": segment_video(cp.frames, text, box, points, None, prompt_frame, self.config.get("segmentation.model"),
                                           self.config.get("segmentation.checkpoint"), self.device, cp.flow_bw, cp.flow_fw)},
        )
        return res["mask"]

    def build_masks(self, cp: ControlPackage, edit: np.ndarray, protect: Optional[list] = None, depth_occlusion: bool = True) -> ControlPackage:
        from .compositing import depth_occlusion_from_scene, occlusion_protect
        from .segmentation import build_preservation_masks, temporal_smooth_masks

        edit = temporal_smooth_masks(edit, cp.flow_bw, 0.4)
        prot = None
        if protect:
            prot = np.max(np.stack(protect), 0)
            if cp.depth is not None:
                prot = occlusion_protect(cp.depth, edit, protect)
        elif depth_occlusion and cp.depth is not None:
            prot = depth_occlusion_from_scene(cp.depth, edit)
        cp.subject_mask = (edit > 0.5).astype(np.float32)
        cp.edit_mask, cp.preserve_mask, cp.soft_mask = build_preservation_masks(
            edit, prot, self.config.get("compositing.dilate_px", 6), self.config.get("compositing.feather_px", 12)
        )
        return cp

    # ------------------------------------------------------------ generation
    def generate(self, cp: ControlPackage, backend_name: Optional[str] = None) -> np.ndarray:
        name = backend_name or self.config.get("backend")
        backend = get_backend(name, self.config)
        refs_fp = [(r.category.value, r.weight, fingerprint(r.image)) for r in cp.references.references]
        key = fingerprint(name, Operation(cp.operation).value, cp.prompt, cp.negative_prompt, cp.seed, cp.steps, cp.guidance,
                          refs_fp, cp.frames, cp.edit_mask, cp.subject_mask, cp.extras.get("recast_control"))
        T = cp.num_frames
        if backend.chunks_internally(cp.operation):
            chunks = plan_chunks(T, T + 1, 0, align_4n1=False)
        else:
            chunks = plan_chunks(T, cp.temporal.chunk_frames, cp.temporal.overlap_frames, cuts=cp.scene_cuts)
        done = self.cache.done_chunks(key)
        outs = []
        for i, ch in enumerate(chunks):
            self.progress("generate", i / len(chunks))
            if i in done:
                log.info("resuming chunk %d/%d from cache", i + 1, len(chunks))
                outs.append(self.cache.load("generated", f"{key}_c{i}")["generated"])
                continue
            sub = cp.slice(ch.start, ch.end)
            out = backend.generate(sub)
            p = self.cache.save("generated", f"{key}_c{i}", generated=out)
            if p is not None:
                self.cache.mark_chunk(key, i, str(p))
            outs.append(out)
        self.progress("generate", 1.0)
        return merge_chunks(outs, chunks, T) if len(chunks) > 1 else outs[0][:T]

    # ------------------------------------------------------------ post
    def postprocess(self, cp: ControlPackage, generated: np.ndarray, report: Optional[dict] = None) -> np.ndarray:
        from .compositing import composite
        from .segmentation import dilate, feather
        from .temporal import deflicker_luminance, temporal_repair

        report = report if report is not None else {}
        op = Operation(cp.operation)
        out = generated
        ts = cp.temporal
        if ts.enabled and cp.flow_fw is not None:
            region = cp.soft_mask if op in REGION_OPS else None
            out = temporal_repair(out, cp.flow_fw, cp.flow_bw, ts.repair_strength,
                                  self.config.get("temporal.inconsistency_threshold", 0.08), region)
            if op in RECAST_OPS | CHARACTER_OPS:
                out = deflicker_luminance(out, strength=ts.smoothing_strength)
        self.progress("temporal", 1.0)

        ident = cp.references.primary(RefCategory.CHARACTER)
        if self.config.get("identity.enabled") and ident is not None and ident.embedding is not None and op in CHARACTER_OPS | {Operation.FULL_RECAST}:
            try:
                from .identity import identity_scores

                rep = identity_scores(out, ident.embedding, self.config.get("identity.threshold", 0.45), stride=2, device=self.device)
                report["identity"] = rep.as_dict()
            except Exception as e:
                report["identity"] = {"error": str(e)}

        if self.config.get("compositing.enabled", True):
            cm = 1.0 if self.config.get("compositing.color_match", True) else 0.0
            if op in REGION_OPS and cp.soft_mask is not None:
                out = composite(cp.frames, out, cp.soft_mask, cp.preserve_mask, cp.edit_mask, color_match_strength=cm)
            elif op == Operation.CHARACTER_SWAP and "background" in cp.invariants and cp.subject_mask is not None:
                soft = feather(dilate(cp.subject_mask, 24), 24)
                out = composite(cp.frames, out, soft, None, dilate(cp.subject_mask, 24), color_match_strength=0.0, local_luminance=0.0)
            elif op == Operation.ENVIRONMENT_RECAST and cp.subject_mask is not None and cp.extras.get("restore_subject", True):
                keep = feather(cp.subject_mask, 6)
                out = composite(out, cp.frames, keep, None, cp.subject_mask, color_match_strength=0.0, local_luminance=0.0)
        self.progress("composite", 1.0)

        if self.config.get("upscale.enabled", False):
            from .restoration import restore

            out = restore(out, self.config.get("upscale.scale", 2), self.config.get("upscale.model_path"),
                          self.config.get("upscale.sharpen", 0.15), self.device)
        return out.clip(0, 1).astype(np.float32)

    # ------------------------------------------------------------ export
    def export(self, frames: np.ndarray, cp: ControlPackage, out_path: str, codec: Optional[str] = None, keep_audio: bool = True, audio_path: Optional[str] = None) -> str:
        from .video import write_video

        return write_video(frames, out_path, cp.fps, codec or self.config.get("output.codec", "h264"), self.config.get("output.crf", 16),
                           audio_from=audio_path or (cp.source_path if keep_audio else None))

    # ------------------------------------------------------------ one call
    def run(
        self,
        source: str,
        operation: Operation | str,
        references: ReferencePackage,
        target: str = "",
        box=None,
        prompt: str = "",
        preserve: Optional[list] = None,
        protect_targets: Optional[list] = None,
        seed: int = 0,
        steps: int = 30,
        guidance: float = 5.0,
        backend: Optional[str] = None,
        out_path: Optional[str] = None,
        project_path: Optional[str] = None,
        max_frames: int = 0,
        max_side: int = 0,
        evaluate: bool = True,
        strength: float = 1.0,
    ) -> RunResult:
        op = Operation(operation)
        cp = self.load(source, max_frames=max_frames, max_side=max_side)
        cp.operation = op
        cp.invariants = []
        cp.__post_init__()
        if preserve:
            cp.invariants = sorted(set(cp.invariants) | {p.lower() for p in preserve})
        cp.references = references
        cp.seed, cp.steps, cp.guidance = seed, steps, guidance
        cp.strengths = Strengths(preservation=strength, motion=strength)
        cp.temporal = TemporalSettings(
            enabled=self.config.get("temporal.enabled", True), strength=self.config.get("temporal.strength", 0.75),
            smoothing_strength=self.config.get("temporal.smoothing_strength", 0.3), repair_strength=self.config.get("temporal.repair_strength", 0.6),
            chunk_frames=self.config.get("chunking.chunk_frames", 81), overlap_frames=self.config.get("chunking.overlap_frames", 9),
        )
        backend_name = backend or self.config.get("backend")
        uses_models = backend_name != "classical"
        needs_pose = op in CHARACTER_OPS
        needs_depth = uses_models and (op in RECAST_OPS or op in REGION_OPS)
        self.analyze(cp, flow=True, camera=True, cuts=True, depth=False, pose=needs_pose and uses_models)
        if needs_depth:
            try:
                self.analyze(cp, flow=False, camera=False, cuts=False, depth=True)
            except Exception as e:
                log.warning("depth unavailable: %s", e)

        if op in REGION_OPS or op == Operation.CHARACTER_SWAP or (op == Operation.ENVIRONMENT_RECAST):
            tgt = target or ("person" if op in (Operation.CHARACTER_SWAP, Operation.ENVIRONMENT_RECAST, Operation.OUTFIT_SWAP) else "")
            if tgt or box is not None:
                mask = self.select(cp, tgt, box)
                protect = [self.select(cp, p) for p in (protect_targets or [])]
                self.build_masks(cp, mask, protect or None)
                cp.camera = None
                self.analyze(cp, flow=False, camera=True, cuts=False)  # re-estimate camera without the subject

        spec = PromptSpec(op, target=target, preserve=cp.invariants, user_prompt=prompt)
        cp.prompt, cp.negative_prompt = compile_prompt(spec, references)

        generated = self.generate(cp, backend_name)
        report: dict = {"backend": backend_name, "operation": op.value, "prompt": cp.prompt, "chunks": None}
        final = self.postprocess(cp, generated, report)

        if evaluate:
            from .metrics import evaluate as _eval

            ref = references.primary(RefCategory.PRODUCT, RefCategory.OBJECT, RefCategory.PROP)
            ident = references.primary(RefCategory.CHARACTER)
            fr = final if final.shape == cp.frames.shape else None
            if fr is not None:
                score = _eval(cp.frames, fr, cp.preserve_mask, cp.edit_mask, ref.image if ref else None, ref.mask if ref else None,
                              ident.embedding if ident else None, object_mask=cp.subject_mask)
                report["score"] = score.as_dict()
                if cp.preserve_mask is not None:
                    from .compositing import mask_leakage

                    report["mask_leakage"] = mask_leakage(cp.frames, fr, cp.preserve_mask)
        res = RunResult(final, cp, generated, report=report)
        if out_path:
            res.output_path = self.export(final, cp, out_path)
        if project_path:
            from .project import save_project

            res.project_path = save_project(project_path, cp, settings=self.config.data, state=self.cache.state(), output_path=res.output_path)
        return res
