"""Build the shipped ComfyUI workflows (API format) and validate every node,
input name and link type against the live node definitions.

    python tools/build_workflows.py            # writes workflows/*.json
    python tools/build_workflows.py --check    # validate only (used by tests)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = {  # the few ComfyUI core nodes the workflows use
    "LoadImage": {"inputs": {"image": "STRING"}, "outputs": ["IMAGE", "MASK"]},
}


def load_nodes():
    spec = importlib.util.spec_from_file_location("ComfyUI_Genjutsu", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Genjutsu"] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.g: dict = {}

    def add(self, cls: str, title: str = "", **inputs):
        nid = str(len(self.g) + 1)
        self.g[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title or cls}}
        return _Ref(nid, self)

    def validate(self):
        errs = []
        for nid, n in self.g.items():
            cls = n["class_type"]
            if cls in CORE:
                spec_in, outs = CORE[cls]["inputs"], CORE[cls]["outputs"]
                req = set(spec_in)
                allowed = {k: v for k, v in spec_in.items()}
            else:
                if cls not in self.nodes:
                    errs.append(f"{nid}: unknown node {cls}")
                    continue
                it = self.nodes[cls].INPUT_TYPES()
                req = set(it.get("required", {}))
                allowed = {k: v[0] for sec in ("required", "optional") for k, v in it.get(sec, {}).items()}
            for k in req - set(n["inputs"]):
                errs.append(f"{nid} {cls}: missing required input '{k}'")
            for k, v in n["inputs"].items():
                if k not in allowed:
                    errs.append(f"{nid} {cls}: unknown input '{k}'")
                    continue
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    src = self.g.get(v[0])
                    if src is None:
                        errs.append(f"{nid}: link to missing node {v[0]}")
                        continue
                    sc = src["class_type"]
                    outs = CORE[sc]["outputs"] if sc in CORE else list(self.nodes[sc].RETURN_TYPES)
                    got = outs[v[1]]
                    want = allowed[k]
                    if isinstance(want, list):  # combo fed by a link must be STRING-like
                        want = "STRING"
                    if got != want:
                        errs.append(f"{nid} {cls}.{k}: expects {want}, linked {sc}[{v[1]}]={got}")
                elif isinstance(allowed[k], list) and v not in allowed[k] and not (cls.endswith("Loader") or cls == "GenjutsuOneClick"):
                    errs.append(f"{nid} {cls}.{k}: '{v}' not in {allowed[k]}")
        return errs


class _Ref:
    def __init__(self, nid, g):
        self.nid, self.g = nid, g

    def __getitem__(self, i):
        return [self.nid, i]


LOADER = dict(video="input.mp4", max_frames=0, start_frame=0, stride=1, max_side=1280)


def cp_defaults(**kw):
    d = dict(seed=42, steps=30, guidance=5.0, identity_strength=0.85, motion_strength=1.0, reference_strength=0.75,
             preservation_strength=1.0, style_strength=0.5, chunk_frames=81, overlap_frames=9, recast_control="depth")
    d.update(kw)
    return d


def prompt_defaults(**kw):
    d = dict(target="", motion="original", preserve_person=True, preserve_hands=True, preserve_background=True, preserve_camera=True,
             preserve_lighting=True, preserve_expression=False, user_prompt="", environment="", style="", extra_negative="")
    d.update(kw)
    return d


GEN = dict(backend="wan22", max_side=832, max_area=832 * 480, cpu_offload=True, low_vram=False, vae_tiling=True)
TR = dict(repair_strength=0.6, smoothing_strength=0.3, inconsistency_threshold=0.08)
OUT = dict(codec="h264", crf=16, fps=0.0, keep_audio=True)


def ref(g, file, category, name, desc=""):
    img = g.add("LoadImage", f"{category.title()} reference", image=file)
    return img, dict(image=img[0], category=category, weight=1.0, name=name, description=desc, compute_embeddings=True)


def motion_transfer(g):
    v = g.add("GenjutsuVideoLoader", "Motion source video", **LOADER)
    pose = g.add("GenjutsuPoseExtractor", frames=v[0], face_size=512, smoothing=0.5)
    _, r = ref(g, "character.png", "character", "new_character")
    refs = g.add("GenjutsuReference", **r)
    cp = g.add("GenjutsuControlPackage", frames=v[0], operation="motion_transfer", video_info=v[1], references=refs[0],
               pose_frames=pose[0], face_frames=pose[1], **cp_defaults(recast_control="pose"))
    pc = g.add("GenjutsuPromptCompiler", controls=cp[0], **prompt_defaults(preserve_person=False, preserve_background=False, preserve_hands=False))
    gen = g.add("GenjutsuGenerate", controls=pc[0], **GEN)
    tr = g.add("GenjutsuTemporalRepair", generated=gen[0], controls=pc[0], limit_to_edit_region=False, **TR)
    idl = g.add("GenjutsuIdentityLock", generated=tr[0], references=refs[0], threshold=0.45, stride=2)
    g.add("GenjutsuVideoOutput", frames=tr[0], filename_prefix="genjutsu/motion_transfer", video_info=v[1], **OUT)


def _region_swap(g, operation, target_text, ref_file, ref_cat, prefix):
    v = g.add("GenjutsuVideoLoader", "Source video", **LOADER)
    flow = g.add("GenjutsuOpticalFlow", frames=v[0], model="raft", max_side=960)
    obj = g.add("GenjutsuSegmenter", f"Select {target_text}", frames=v[0], target="custom", text=target_text, box="", points="",
                prompt_frame=0, model="sam2", temporal_smoothing=0.4, fill_holes=True, flow=flow[0])
    hands = g.add("GenjutsuSegmenter", "Occluders (hands)", frames=v[0], target="hands", text="", box="", points="", prompt_frame=0,
                  model="sam2", temporal_smoothing=0.4, fill_holes=True, flow=flow[0])
    depth = g.add("GenjutsuDepthExtractor", frames=v[0], model="video_depth_anything", encoder="vits", fps=v[2], flow=flow[0])
    pm = g.add("GenjutsuPreservationMasks", edit_mask=obj[0], dilate_px=6, feather_px=12, depth_occlusion=True, protect_mask=hands[0], depth=depth[1])
    _, r = ref(g, ref_file, ref_cat, "replacement")
    refs = g.add("GenjutsuReference", **r)
    cp = g.add("GenjutsuControlPackage", frames=v[0], operation=operation, video_info=v[1], references=refs[0], edit_mask=pm[0],
               preserve_mask=pm[1], soft_mask=pm[2], subject_mask=obj[0], depth=depth[1], flow=flow[0], **cp_defaults())
    pc = g.add("GenjutsuPromptCompiler", controls=cp[0], **prompt_defaults(target=target_text))
    gen = g.add("GenjutsuGenerate", controls=pc[0], **GEN)
    tr = g.add("GenjutsuTemporalRepair", generated=gen[0], controls=pc[0], limit_to_edit_region=True, **TR)
    comp = g.add("GenjutsuCompositor", original=v[0], generated=tr[0], soft_mask=pm[2], preserve_mask=pm[1], edit_mask=pm[0],
                 mode="multiband", color_match=1.0, local_lighting=0.5)
    out = g.add("GenjutsuVideoOutput", frames=comp[0], filename_prefix=f"genjutsu/{prefix}", video_info=v[1], **OUT)
    g.add("GenjutsuEvaluate", source=v[0], output=comp[0], max_side=480, controls=pc[0])
    g.add("GenjutsuSaveProject", controls=pc[0], name=f"genjutsu/{prefix}", embed_source=False, output_path=out[0])


def object_swap(g):
    _region_swap(g, "object_swap", "wrist watch", "object.png", "object", "object_swap")


def product_swap(g):
    _region_swap(g, "product_swap", "bottle", "product.png", "product", "product_swap")


def character_swap(g):
    v = g.add("GenjutsuVideoLoader", "Source performance", **LOADER)
    flow = g.add("GenjutsuOpticalFlow", frames=v[0], model="raft", max_side=960)
    person = g.add("GenjutsuSegmenter", "Performer", frames=v[0], target="person", text="", box="", points="", prompt_frame=0,
                   model="sam2", temporal_smoothing=0.4, fill_holes=True, flow=flow[0])
    pose = g.add("GenjutsuPoseExtractor", frames=v[0], face_size=512, smoothing=0.5)
    _, r = ref(g, "character.png", "character", "new_character")
    refs = g.add("GenjutsuReference", **r)
    cp = g.add("GenjutsuControlPackage", frames=v[0], operation="character_swap", video_info=v[1], references=refs[0],
               subject_mask=person[0], pose_frames=pose[0], face_frames=pose[1], flow=flow[0], **cp_defaults())
    pc = g.add("GenjutsuPromptCompiler", controls=cp[0], **prompt_defaults(preserve_person=False, preserve_expression=True))
    gen = g.add("GenjutsuGenerate", controls=pc[0], **GEN)
    tr = g.add("GenjutsuTemporalRepair", generated=gen[0], controls=pc[0], limit_to_edit_region=False, **TR)
    g.add("GenjutsuIdentityLock", generated=tr[0], references=refs[0], threshold=0.45, stride=2)
    big = g.add("GenjutsuMaskOps", "Generous performer area", a=person[0], op="dilate_a", amount_px=24)
    soft = g.add("GenjutsuMaskOps", "Feather", a=big[0], op="feather_a", amount_px=24)
    comp = g.add("GenjutsuCompositor", "Keep original background", original=v[0], generated=tr[0], soft_mask=soft[0], edit_mask=big[0],
                 mode="alpha", color_match=0.0, local_lighting=0.0)
    g.add("GenjutsuVideoOutput", frames=comp[0], filename_prefix="genjutsu/character_swap", video_info=v[1], **OUT)


def environment_recast(g):
    v = g.add("GenjutsuVideoLoader", "Source video", **LOADER)
    flow = g.add("GenjutsuOpticalFlow", frames=v[0], model="raft", max_side=960)
    person = g.add("GenjutsuSegmenter", "Subject to keep", frames=v[0], target="person", text="", box="", points="", prompt_frame=0,
                   model="sam2", temporal_smoothing=0.4, fill_holes=True, flow=flow[0])
    depth = g.add("GenjutsuDepthExtractor", frames=v[0], model="video_depth_anything", encoder="vits", fps=v[2], flow=flow[0])
    cam = g.add("GenjutsuCameraAnalyzer", frames=v[0], exclude_mask=person[0])
    _, r = ref(g, "location.png", "location", "new_world", "a rain-soaked neon futuristic city street at night")
    refs = g.add("GenjutsuReference", **r)
    cp = g.add("GenjutsuControlPackage", frames=v[0], operation="environment_recast", video_info=v[1], references=refs[0],
               subject_mask=person[0], depth=depth[1], flow=flow[0], camera=cam[0], **cp_defaults(recast_control="depth"))
    pc = g.add("GenjutsuPromptCompiler", controls=cp[0], **prompt_defaults(preserve_background=False, environment="a futuristic city"))
    gen = g.add("GenjutsuGenerate", controls=pc[0], **GEN)
    tr = g.add("GenjutsuTemporalRepair", generated=gen[0], controls=pc[0], limit_to_edit_region=False, **TR)
    keep = g.add("GenjutsuMaskOps", "Feathered subject", a=person[0], op="feather_a", amount_px=6)
    comp = g.add("GenjutsuCompositor", "Restore original subject", original=tr[0], generated=v[0], soft_mask=keep[0], edit_mask=person[0],
                 mode="alpha", color_match=0.0, local_lighting=0.0)
    g.add("GenjutsuVideoOutput", frames=comp[0], filename_prefix="genjutsu/environment_recast", video_info=v[1], **OUT)


def full_genjutsu(g):
    _, r1 = ref(g, "character.png", "character", "hero")
    a = g.add("GenjutsuReference", **r1)
    _, r2 = ref(g, "location.png", "location", "world")
    b = g.add("GenjutsuReference", references=a[0], **r2)
    _, r3 = ref(g, "style.png", "style", "look")
    c = g.add("GenjutsuReference", references=b[0], **{**r3, "weight": 0.6})
    g.add("GenjutsuOneClick", "Genjutsu", video="input.mp4", operation="full_recast", references=c[0], target="person",
          prompt="", preserve_motion=True, preserve_camera=True, preserve_background=False, preserve_identity=False,
          preserve_lighting=False, quality="standard", strength=1.0, seed=42, backend="wan22", codec="h264", upscale=1.0,
          max_frames=0, box="", protect="", video_path="", save_project=True)


WORKFLOWS = {
    "MotionTransfer": motion_transfer,
    "ObjectSwap": object_swap,
    "CharacterSwap": character_swap,
    "ProductSwap": product_swap,
    "EnvironmentRecast": environment_recast,
    "FullGenjutsu": full_genjutsu,
}


def build(check_only: bool = False) -> dict:
    nodes = load_nodes()
    problems = {}
    for name, fn in WORKFLOWS.items():
        g = Graph(nodes)
        fn(g)
        errs = g.validate()
        if errs:
            problems[name] = errs
        if not check_only:
            (ROOT / "workflows").mkdir(exist_ok=True)
            (ROOT / "workflows" / f"{name}.json").write_text(json.dumps(g.g, indent=2))
    return problems


if __name__ == "__main__":
    probs = build("--check" in sys.argv)
    if probs:
        print(json.dumps(probs, indent=2))
        sys.exit(1)
    print(f"{len(WORKFLOWS)} workflows OK")
