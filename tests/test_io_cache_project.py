"""Video I/O, cache/resume, project files, prompts, identity helpers, metrics."""
import numpy as np
import pytest

from synth import make_frames, write_with_audio


def test_video_roundtrip_preserves_fps_size_audio(tmp_path, clip):
    from genjutsu.video import load_frames, probe, write_video

    frames, _, _ = clip
    src = write_with_audio(frames, tmp_path / "src.mp4", fps=12)
    got, info = load_frames(src)
    assert got.shape == frames.shape and abs(info.fps - 12) < 1e-3 and info.has_audio
    for codec, ext in (("h264", "mp4"), ("h265", "mp4"), ("prores", "mov"), ("vp9", "webm")):
        out = write_video(got, str(tmp_path / f"o_{codec}.{ext}"), info.fps, codec, audio_from=src)
        p = probe(out)
        assert (p.width, p.height) == (frames.shape[2], frames.shape[1])
        assert abs(p.fps - 12) < 1e-3 and p.has_audio, codec
    d = write_video(got, str(tmp_path / "seq"), 12, "png")
    assert len(list((tmp_path / "seq").glob("*.png"))) == len(frames)


def test_load_frames_stride_and_limits(tmp_path, clip):
    from genjutsu.video import load_frames

    frames, _, _ = clip
    src = write_with_audio(frames, tmp_path / "src.mp4", fps=12)
    f, info = load_frames(src, start_frame=4, max_frames=5, stride=2)
    assert len(f) == 5 and abs(info.fps - 6) < 1e-3


def test_scene_cut_detection():
    from genjutsu.video import detect_scene_cuts

    a, _, _ = make_frames(T=10, seed=0)
    b, _, _ = make_frames(T=10, seed=7)
    b = (1 - b).astype(np.float32)  # very different shot
    assert detect_scene_cuts(np.concatenate([a, b])) == [10]


def test_cache_resumes_after_failed_chunk(tmp_path, clip, cfg):
    from genjutsu.backends import base
    from genjutsu.backends.base import VideoBackend, register_backend
    from genjutsu.controls import ControlPackage, Operation
    from genjutsu.pipeline import GenjutsuPipeline

    calls = []

    @register_backend("flaky_test")
    class Flaky(VideoBackend):
        supports = {Operation.STYLE_RECAST}
        fail_on = 1

        def generate_recast(self, cp):
            calls.append(len(cp.frames))
            if len(calls) - 1 == Flaky.fail_on:
                raise RuntimeError("simulated OOM")
            return cp.frames * 0.5

    frames, _, _ = clip
    cp = ControlPackage(frames=frames, fps=12, operation=Operation.STYLE_RECAST)
    cp.temporal.chunk_frames, cp.temporal.overlap_frames = 9, 2
    pipe = GenjutsuPipeline(cfg, "resume")
    with pytest.raises(RuntimeError):
        pipe.generate(cp, "flaky_test")
    assert len(calls) == 2
    Flaky.fail_on = -1
    base._INSTANCES.pop("flaky_test", None)
    out = pipe.generate(cp, "flaky_test")
    # chunk 0 came from cache: only the remaining chunks were generated
    n_chunks = len(__import__("genjutsu.temporal", fromlist=["x"]).plan_chunks(len(frames), 9, 2))
    assert len(calls) == 2 + (n_chunks - 1)
    assert np.allclose(out, frames * 0.5, atol=2e-3)


def test_project_roundtrip(tmp_path, clip, product_ref):
    from genjutsu.controls import ControlPackage, Operation, ReferencePackage
    from genjutsu.project import load_project, save_project
    from genjutsu.segmentation import build_preservation_masks

    frames, masks, _ = clip
    src = write_with_audio(frames, tmp_path / "src.mp4")
    e, p, s = build_preservation_masks(masks, None)
    cp = ControlPackage(frames=frames, fps=12, operation=Operation.PRODUCT_SWAP, source_path=src, prompt="x", edit_mask=e,
                        preserve_mask=p, soft_mask=s, references=ReferencePackage([product_ref]), seed=123,
                        depth=np.random.default_rng(0).random(masks.shape).astype(np.float32))
    path = save_project(str(tmp_path / "proj"), cp, settings={"backend": "classical"}, state={"done": True})
    for embed in (False, True):
        path = save_project(str(tmp_path / f"proj{embed}"), cp, embed_source=embed)
        cp2, man = load_project(path, extract_dir=str(tmp_path / "x"))
        assert cp2.seed == 123 and cp2.operation == Operation.PRODUCT_SWAP and cp2.prompt == "x"
        assert np.abs(cp2.edit_mask - e).max() < 1 / 255 + 1e-6
        assert np.abs(cp2.depth - cp.depth).max() < 1e-4
        assert len(cp2.references) == 1 and cp2.references.references[0].mask is not None
        assert cp2.frames.shape == frames.shape
        assert man["source"]["sha256"]


def test_prompt_compiler():
    from genjutsu.controls import Operation, RefCategory, Reference, ReferencePackage
    from genjutsu.prompt import PromptSpec, compile_prompt

    refs = ReferencePackage([Reference(np.zeros((4, 4, 3), np.float32), RefCategory.PRODUCT, name="sodaX")])
    pos, neg = compile_prompt(PromptSpec(Operation.PRODUCT_SWAP, "can", ["person", "hand", "background", "camera"]), refs)
    assert "Replace only the can" in pos and "sodaX" in pos and "camera trajectory" in pos and "movement" in pos
    assert "original can still visible" in neg
    pos2, _ = compile_prompt(PromptSpec(Operation.ENVIRONMENT_RECAST, environment="a neon city"))
    assert "a neon city" in pos2


def test_identity_segments():
    from genjutsu.identity import segments_to_regenerate

    assert segments_to_regenerate([], 100) == []
    assert segments_to_regenerate([10, 12, 14, 60], 100, pad=2, min_gap=5) == [(8, 17), (58, 63)]


def test_reference_classification(product_ref):
    from genjutsu.controls import RefCategory
    from genjutsu.references import classify_reference

    assert product_ref.category == RefCategory.PRODUCT
    rng = np.random.default_rng(0)
    busy = rng.random((90, 160, 3)).astype(np.float32)
    assert classify_reference(busy)[0] == RefCategory.LOCATION


def test_metrics_rank_good_above_bad(clip):
    from genjutsu.metrics import evaluate
    from genjutsu.segmentation import build_preservation_masks

    frames, masks, _ = clip
    e, p, _ = build_preservation_masks(masks, None)
    good = evaluate(frames, frames, p, e)
    rng = np.random.default_rng(0)
    bad = evaluate(frames, (frames + rng.normal(0, 0.1, frames.shape)).clip(0, 1).astype(np.float32), p, e)
    assert good.background > 0.99 and good.overall() > bad.overall()
    assert "Motion" in good.table()
