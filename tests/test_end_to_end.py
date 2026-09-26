"""End-to-end: library pipeline, ComfyUI node graphs, shipped workflows, and
the Wan backends' calls into diffusers (with diffusers/torch stubbed, so the
exact arguments sent to WanVACEPipeline / WanAnimatePipeline are verified
without a GPU)."""
import json
import sys
import types

import numpy as np
import pytest

from conftest import ROOT, iou
from synth import make_frames, reference_image, write_with_audio


# ------------------------------------------------------------------ library

def test_pipeline_product_swap_end_to_end(tmp_path, cfg, product_ref):
    from genjutsu.controls import Operation, ReferencePackage
    from genjutsu.pipeline import GenjutsuPipeline
    from genjutsu.video import probe

    frames, masks, boxes = make_frames(T=24)
    src = write_with_audio(frames, tmp_path / "src.mp4")
    x0, y0, x1, y1 = boxes[0]
    res = GenjutsuPipeline(cfg, "e2e").run(src, Operation.PRODUCT_SWAP, ReferencePackage([product_ref]), box=[x0 - 3, y0 - 3, x1 + 3, y1 + 3],
                                           out_path=str(tmp_path / "out.mp4"), project_path=str(tmp_path / "proj"))
    # 1. selected object replaced (center of object is the reference's blue)
    for t in (0, 12, 23):
        bx0, by0, bx1, by1 = boxes[t]
        c = res.frames[t, (by0 + by1) // 2, (bx0 + bx1) // 2]
        assert c[2] > 0.7 and c[0] < 0.3, (t, c)
    # 1b. no remnant of the old object anywhere (strongly red pixels)
    red = (res.frames[..., 0] > 0.7) & (res.frames[..., 1] < 0.25) & (res.frames[..., 2] < 0.25)
    assert red.sum() / len(res.frames) < 3, red.sum((1, 2))
    # 2. unedited regions untouched
    assert res.report["mask_leakage"] == 0.0
    assert res.report["score"]["background"] > 0.99
    # 3. motion / camera preserved; audio + fps preserved; project written
    assert res.report["score"]["motion"] > 0.9 and res.report["score"]["camera"] > 0.9
    p = probe(res.output_path)
    assert p.has_audio and abs(p.fps - 12) < 1e-3
    assert res.project_path.endswith(".genjutsu")


def test_pipeline_reproducible_from_project(tmp_path, cfg, product_ref):
    """Definition of done #15: reopen the project and reproduce the result."""
    from genjutsu.controls import Operation, ReferencePackage
    from genjutsu.pipeline import GenjutsuPipeline
    from genjutsu.project import load_project

    frames, masks, boxes = make_frames(T=16)
    src = write_with_audio(frames, tmp_path / "src.mp4")
    x0, y0, x1, y1 = boxes[0]
    pipe = GenjutsuPipeline(cfg, "repro")
    res = pipe.run(src, Operation.PRODUCT_SWAP, ReferencePackage([product_ref]), box=[x0 - 3, y0 - 3, x1 + 3, y1 + 3],
                   project_path=str(tmp_path / "p"), evaluate=False)
    cp, _ = load_project(res.project_path)
    cp.flow_fw, cp.flow_bw = res.control.flow_fw, res.control.flow_bw
    cp.subject_mask = res.control.subject_mask
    again = pipe.postprocess(cp, pipe.generate(cp, "classical"))
    assert np.abs(again - res.frames).max() < 0.02


def test_pipeline_environment_recast_keeps_subject(tmp_path, cfg):
    from genjutsu.controls import Operation, ReferencePackage
    from genjutsu.pipeline import GenjutsuPipeline
    from genjutsu.references import build_reference

    frames, masks, boxes = make_frames(T=12)
    src = write_with_audio(frames, tmp_path / "src.mp4")
    loc = np.zeros((96, 128, 3), np.float32)
    loc[..., 2] = np.linspace(0.3, 1, 128)  # blue-ish world
    refs = ReferencePackage([build_reference(loc, "location", 1.0, "world")])
    x0, y0, x1, y1 = boxes[0]
    res = GenjutsuPipeline(cfg, "recast").run(src, Operation.ENVIRONMENT_RECAST, refs, box=[x0 - 3, y0 - 3, x1 + 3, y1 + 3], evaluate=False)
    bg = masks < 0.5
    assert res.frames[..., 2][bg].mean() > frames[..., 2][bg].mean() + 0.05  # environment changed
    core = np.stack([np.pad(m[3:-3, 3:-3], 3) for m in masks]) > 0.5
    assert np.abs(res.frames[core] - frames[core]).mean() < 0.06  # subject preserved


# ------------------------------------------------------------------ ComfyUI

def _load_nodes():
    sys.path.insert(0, ROOT + "/tools")
    from build_workflows import load_nodes

    return load_nodes()


def run_api_graph(graph: dict, nodes: dict, images: dict):
    """Minimal executor for API-format workflows (topological, memoized)."""
    results: dict = {}

    def run(nid):
        if nid in results:
            return results[nid]
        n = graph[nid]
        kw = {}
        for k, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                kw[k] = run(v[0])[v[1]]
            else:
                kw[k] = v
        if n["class_type"] == "LoadImage":
            img = images[kw["image"]]
            out = (img[None], np.zeros(img.shape[:2], np.float32)[None])
        else:
            cls = nodes[n["class_type"]]
            out = getattr(cls(), cls.FUNCTION)(**kw)
            if isinstance(out, dict):
                out = out["result"]
        results[nid] = out
        return out

    for nid in graph:
        run(nid)
    return results


def test_shipped_workflows_validate():
    sys.path.insert(0, ROOT + "/tools")
    from build_workflows import build

    assert build(check_only=True) == {}


def test_workflow_validator_catches_bad_links():
    sys.path.insert(0, ROOT + "/tools")
    from build_workflows import Graph

    g = Graph(_load_nodes())
    v = g.add("GenjutsuVideoLoader", video="x.mp4", max_frames=0, start_frame=0, stride=1, max_side=0)
    g.add("GenjutsuSegmenter", frames=v[1], target="person", text="", box="", points="", prompt_frame=0, model="sam2",
          temporal_smoothing=0.4, fill_holes=True, bogus=1)
    errs = " ".join(g.validate())
    assert "expects IMAGE" in errs and "unknown input 'bogus'" in errs


def test_node_graph_object_swap_runs(tmp_path, monkeypatch):
    """Execute a full node graph (no models: DIS flow, GrabCut, classical backend)."""
    from genjutsu.config import GenjutsuConfig

    nodes = _load_nodes()
    pkg = "ComfyUI_Genjutsu.nodes"
    out_dir = lambda: str(tmp_path / "out")  # noqa: E731
    cfg = lambda o=None: GenjutsuConfig().with_overrides({"cache": {"root": str(tmp_path / "c")}, **(o or {})})  # noqa: E731
    for mod, attr, val in [("_util", "output_dir", out_dir), ("video_output", "output_dir", out_dir),
                           ("_util", "get_config", cfg), ("wan_generator", "get_config", cfg)]:
        monkeypatch.setattr(sys.modules[f"{pkg}.{mod}"], attr, val)

    frames, masks, boxes = make_frames(T=16)
    src = write_with_audio(frames, tmp_path / "src.mp4")
    x0, y0, x1, y1 = boxes[0]
    sys.path.insert(0, ROOT + "/tools")
    from build_workflows import Graph

    g = Graph(nodes)
    v = g.add("GenjutsuVideoLoader", video="", video_path=src, max_frames=0, start_frame=0, stride=1, max_side=0)
    fl = g.add("GenjutsuOpticalFlow", frames=v[0], model="dis", max_side=512)
    seg = g.add("GenjutsuSegmenter", frames=v[0], target="custom", text="", box=f"{x0-3},{y0-3},{x1+3},{y1+3}", points="",
                prompt_frame=0, model="grabcut", temporal_smoothing=0.3, fill_holes=True, flow=fl[0])
    trk = g.add("GenjutsuTracker", frames=v[0], initial_mask=seg[0], start_frame=0, appearance_weight=0.5, reid_score=0.55, flow=fl[0])
    pm = g.add("GenjutsuPreservationMasks", edit_mask=trk[0], dilate_px=4, feather_px=8, depth_occlusion=True)
    img = g.add("LoadImage", image="product.png")
    ref = g.add("GenjutsuReference", image=img[0], category="product", weight=1.0, name="disc", description="a blue disc",
                compute_embeddings=False)
    cam = g.add("GenjutsuCameraAnalyzer", frames=v[0], exclude_mask=trk[0])
    cp = g.add("GenjutsuControlPackage", frames=v[0], operation="product_swap", video_info=v[1], references=ref[0], edit_mask=pm[0],
               preserve_mask=pm[1], soft_mask=pm[2], subject_mask=trk[0], flow=fl[0], camera=cam[0], seed=1, steps=4, guidance=5.0,
               identity_strength=0.8, motion_strength=1.0, reference_strength=0.8, preservation_strength=1.0, style_strength=0.5,
               chunk_frames=9, overlap_frames=2, recast_control="depth")
    pc = g.add("GenjutsuPromptCompiler", controls=cp[0], target="red cube", motion="original", preserve_person=True, preserve_hands=True,
               preserve_background=True, preserve_camera=True, preserve_lighting=True, preserve_expression=False, user_prompt="",
               environment="", style="", extra_negative="")
    gen = g.add("GenjutsuGenerate", controls=pc[0], backend="classical", max_side=256, max_area=65536, cpu_offload=True, low_vram=False, vae_tiling=False)
    tr = g.add("GenjutsuTemporalRepair", generated=gen[0], controls=pc[0], repair_strength=0.5, smoothing_strength=0.2,
               inconsistency_threshold=0.08, limit_to_edit_region=True)
    comp = g.add("GenjutsuCompositor", original=v[0], generated=tr[0], soft_mask=pm[2], preserve_mask=pm[1], edit_mask=pm[0],
                 mode="multiband", color_match=1.0, local_lighting=0.5)
    out = g.add("GenjutsuVideoOutput", frames=comp[0], filename_prefix="t/out", codec="h264", crf=18, fps=0.0, keep_audio=True, video_info=v[1])
    ev = g.add("GenjutsuEvaluate", source=v[0], output=comp[0], max_side=256, controls=pc[0])
    sp = g.add("GenjutsuSaveProject", controls=pc[0], name="t/proj", embed_source=False, output_path=out[0])
    lp = g.add("GenjutsuLoadProject", project_path=sp[0])
    assert g.validate() == []

    res = run_api_graph(g.g, nodes, {"product.png": reference_image()})
    final = np.asarray(res[comp.nid][0])
    assert res[comp.nid][1] == 0.0  # no leakage
    for t in (0, 8, 15):
        bx0, by0, bx1, by1 = boxes[t]
        c = final[t, (by0 + by1) // 2, (bx0 + bx1) // 2]
        assert c[2] > 0.7 and c[0] < 0.3, (t, c)
    assert "a blue disc" in res[pc.nid][1]
    from genjutsu.video import probe

    assert probe(res[out.nid][0]).has_audio
    assert res[ev.nid][1] > 0.7
    assert res[lp.nid][0].num_frames == 16


# ------------------------------------------------------------------ Wan backends (stubbed diffusers)

class _FakePipe:
    calls: list = []

    @classmethod
    def from_pretrained(cls, mid, **kw):
        p = cls()
        p.model_id, p.load_kw = mid, kw
        p.vae = types.SimpleNamespace(to=lambda *a, **k: None, enable_tiling=lambda: None)
        return p

    def enable_model_cpu_offload(self):
        self.offload = "model"

    def enable_sequential_cpu_offload(self):
        self.offload = "sequential"

    def to(self, d):
        return self

    def __call__(self, **kw):
        _FakePipe.calls.append((type(self).__name__, kw))
        n = kw.get("num_frames") or len(kw["pose_video"])
        return types.SimpleNamespace(frames=[np.full((n, kw["height"], kw["width"], 3), 0.5, np.float32)])


@pytest.fixture
def fake_diffusers(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"

    class Gen:
        def __init__(self, device="cpu"):
            pass

        def manual_seed(self, s):
            self.seed = s
            return self

    torch.Generator = Gen
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
    diff = types.ModuleType("diffusers")
    for name in ("WanPipeline", "WanImageToVideoPipeline", "WanVACEPipeline", "WanAnimatePipeline"):
        setattr(diff, name, type(name, (_FakePipe,), {}))
    diff.AutoencoderKLWan = types.SimpleNamespace(from_pretrained=lambda *a, **k: "vae")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diff)
    _FakePipe.calls.clear()
    from genjutsu.backends import base

    base._INSTANCES.clear()
    yield _FakePipe.calls
    base._INSTANCES.clear()


def _cp(op, T=30, H=90, W=160, **kw):
    from genjutsu.controls import ControlPackage, Operation, Reference, ReferencePackage, RefCategory

    frames = np.random.default_rng(0).random((T, H, W, 3)).astype(np.float32)
    m = np.zeros((T, H, W), np.float32)
    m[:, 30:60, 60:100] = 1
    refs = ReferencePackage([
        Reference(reference_image(), RefCategory.PRODUCT, 1.0, "p", mask=np.ones((64, 64), np.float32)),
        Reference(reference_image(), RefCategory.CHARACTER, 1.0, "c"),
        Reference(reference_image(), RefCategory.LOCATION, 1.0, "l"),
    ])
    return ControlPackage(frames=frames, fps=24, operation=Operation(op), edit_mask=m, subject_mask=m, soft_mask=m,
                          pose_frames=frames.copy(), face_frames=np.zeros((T, 512, 512, 3), np.float32), depth=m.copy(),
                          references=refs, prompt="p", negative_prompt="n", **kw)


def test_wan_vace_region_call(fake_diffusers):
    from genjutsu.backends import get_backend
    from genjutsu.config import GenjutsuConfig

    be = get_backend("wan22", GenjutsuConfig())
    out = be.generate(_cp("product_swap", T=30))
    name, kw = fake_diffusers[-1]
    assert name == "WanVACEPipeline"
    assert (kw["num_frames"] - 1) % 4 == 0 and kw["num_frames"] >= 30
    assert kw["height"] % 16 == 0 and kw["width"] % 16 == 0
    assert len(kw["video"]) == len(kw["mask"]) == kw["num_frames"]
    assert kw["reference_images"] and kw["conditioning_scale"] == 1.0
    assert out.shape == (30, 90, 160, 3)
    # the region to regenerate is grayed out in the conditioning video
    v = np.asarray(kw["video"][0]).astype(np.float32) / 255
    mk = np.asarray(kw["mask"][0])[..., 0] > 127
    assert abs(v[mk].mean() - 0.5) < 0.02


def test_wan_animate_replace_and_animate_modes(fake_diffusers):
    from genjutsu.backends import get_backend
    from genjutsu.config import GenjutsuConfig

    be = get_backend("wan22", GenjutsuConfig())
    out = be.generate(_cp("character_swap"))
    name, kw = fake_diffusers[-1]
    assert name == "WanAnimatePipeline" and kw["mode"] == "replace"
    assert len(kw["background_video"]) == len(kw["mask_video"]) == len(kw["pose_video"]) == 30
    assert kw["guidance_scale"] == 1.0 and (kw["segment_frame_length"] - 1) % 4 == 0
    assert out.shape == (30, 90, 160, 3)
    be.generate(_cp("motion_transfer"))
    name, kw = fake_diffusers[-1]
    assert kw["mode"] == "animate" and "background_video" not in kw
    assert be.chunks_internally("motion_transfer") and not be.chunks_internally("object_swap")


def test_wan_environment_recast_protects_subject(fake_diffusers):
    from genjutsu.backends import get_backend
    from genjutsu.config import GenjutsuConfig

    get_backend("wan22", GenjutsuConfig()).generate(_cp("environment_recast"))
    name, kw = fake_diffusers[-1]
    assert name == "WanVACEPipeline"
    mk = np.asarray(kw["mask"][0])[..., 0] > 127
    h, w = mk.shape
    assert not mk[int(h * 45 / 90), int(w * 80 / 160)]  # subject: not regenerated
    assert mk[2, 2]  # background: regenerated


def test_wan_low_vram_uses_sequential_offload(fake_diffusers):
    from genjutsu.backends import get_backend
    from genjutsu.config import GenjutsuConfig

    be = get_backend("wan22", GenjutsuConfig().with_overrides({"performance": {"low_vram": True}}), reuse=False)
    be.generate(_cp("object_swap"))
    assert be._pipes["vace"].offload == "sequential"
