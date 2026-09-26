# ComfyUI-Genjutsu (Genjutsu-OSS)

**A Genjutsu-style open-source video transformation pipeline for ComfyUI.**

Load a video, pick what should change and what must stay the same, give it
reference images, and get back the same performance, camera and timing with
the requested changes: a swapped product, a new character, a different world.

> This is an independent open-source project. It is **not** affiliated with
> Higgsfield, and it contains no Higgsfield code or models. It reproduces the
> *observable capabilities* of Genjutsu-style editors using publicly available
> open-source models. See [Observed vs. inferred vs. implemented](#observed-vs-inferred-vs-implemented).

---

## The idea: a transformation engine, not a generator

A video generator asks *"what video should I create?"*. Genjutsu-OSS asks:

> What parts of this existing video must stay **invariant**, what may
> **change**, and what information must be **transferred** from the original
> into the new generation?

```
SOURCE VIDEO + REFERENCES + TRANSFORMATION + INVARIANTS  →  NEW VIDEO
```

Invariants (motion, timing, camera, composition, identity, lighting,
background, objects) depend on the operation, and every stage enforces them:
masks decide what is *allowed* to change, the source's optical flow is the
yardstick for temporal repair, and the compositor guarantees preserved pixels
are bit-for-bit the original.

```
SOURCE ─► UNDERSTAND ─► EXTRACT CONTROLS ─► CONTROL PACKAGE ─► GENERATE ─► TEMPORAL REPAIR ─► COMPOSITE ─► RESTORE ─► ENCODE (+audio)
           cuts, flow,    masks, pose,        model-independent   Wan 2.2     flow-guided,       color/light    optional    FFmpeg
           camera         depth, faces                            (swappable) source-referenced  matched        upscale
```

## Operations

| Operation | Keeps | Changes | Backend path (default `wan22`) |
|---|---|---|---|
| `object_swap` / `product_swap` | person, hands, background, camera, lighting | the selected object | Wan VACE masked video inpainting + reference image, then composite |
| `outfit_swap` | identity, motion, background | clothing | Wan VACE masked inpainting |
| `character_swap` | motion, expression, camera, background | the performer | Wan2.2-Animate **replace** mode (pose + face + masked background) |
| `motion_transfer` | the source motion | everything else | Wan2.2-Animate **animate** mode |
| `environment_recast` | subject, motion, camera | environment, architecture, atmosphere | Wan VACE: subject pixels kept, background regenerated from depth control |
| `style_recast` / `full_recast` | motion, camera, composition | look / everything | Wan VACE with depth (or pose / edges) control video + references |

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-Genjutsu
cd ComfyUI-Genjutsu
python install.py              # core deps + a report of what is/isn't available
python install.py --all        # + SAM 2, DWPose, Video Depth Anything, InsightFace, rembg
```

Use the Python that runs ComfyUI (for the portable build:
`..\..\python_embeded\python.exe install.py`). FFmpeg must be on `PATH`.
Model weights download from Hugging Face on first use (set `HF_HOME` to
choose where).

Everything degrades gracefully: without SAM 2 the Segmenter uses GrabCut +
flow tracking from a box; without RAFT, OpenCV DIS flow; without Video Depth
Anything, per-frame Depth Anything V2 stabilized with flow. Missing optional
packages disable only the nodes that need them.

### VRAM

Figures from the Wan 2.2 README for its reference scripts at 720p. These are
not measured by this project, and diffusers offloading changes them.

| Model | Official guidance |
|---|---|
| Wan2.2 A14B (T2V/I2V), Animate-14B | ≥ 80 GB single-GPU with `--offload_model --convert_model_dtype` |
| Wan2.2 TI2V-5B | ≥ 24 GB (e.g. RTX 4090) |

On smaller GPUs: set `performance.cpu_offload: true` (default) or
`low_vram: true` (sequential offload, much slower), keep VAE tiling on,
generate at `max_side: 640–832`, use shorter `chunk_frames`, or point
`models.wan_vace` at the 1.3B VACE checkpoint.

## Workflows

In `workflows/` (ComfyUI API format; drag onto the canvas or queue through
the API). All six are generated and type-checked by `tools/build_workflows.py`.

| File | Inputs | Output |
|---|---|---|
| `MotionTransfer.json` | source video, character reference | character performing the original motion |
| `ObjectSwap.json` | source video, object reference | only the selected object replaced; hands stay in front via depth occlusion |
| `CharacterSwap.json` | source video, character reference | same performance, new performer, original background |
| `ProductSwap.json` | ad footage, product reference | same commercial, new product; chroma-safe color matching keeps brand colors |
| `EnvironmentRecast.json` | source video, location reference/prompt | same subject and camera, new world |
| `FullGenjutsu.json` | source video, character + location + style refs | complete transformation via the One-Click node |

Put `input.mp4` and the reference PNGs in `ComfyUI/input/`, or change the
file names in the loader nodes.

## Nodes

| Node | What it does |
|---|---|
| **Genjutsu ✦ One-Click Transform** | The simple front end: video, operation, references, target, prompt, preservation toggles, quality, strength, seed, codec. Runs the whole pipeline. |
| Video Loader | FFmpeg decode (handles rotation metadata), keeps fps + audio info for the output node. |
| Scene Analyzer | Hard cuts (color + structure cues), shot list, detected subjects (Grounding DINO). |
| Optical Flow | RAFT (torchvision) or DIS; forward + backward flow and a preview. |
| Camera Analyzer | Background feature tracks → per-frame translation/rotation/zoom, trajectory, shake, labels (pan, tilt, zoom, handheld, static). |
| Segmenter | SAM 2 video masks from text (Grounding DINO), box or points; person/face/hands/clothing/object/product/foreground/background. Hole filling + flow-guided temporal smoothing. |
| Tracker | Flow propagation + color-appearance refinement, disappearance detection, distance-gated re-identification. |
| Pose Extractor | DWPose whole-body skeleton frames + stabilized 512px face crops (for Wan-Animate). |
| Depth Extractor | Video Depth Anything (temporally consistent) or Depth Anything V2 + flow stabilization. |
| Reference | Chainable reference package. Auto-classifies (character/product/location/style), cut-out mask, optional identity/CLIP embedding, per-reference weight and description. |
| Preservation Masks | EDIT → dilate → feather → SOFT, with protected occluders and depth-based occlusion. |
| Mask Ops | union / subtract / intersect / invert / dilate / erode / feather. |
| Control Package | Bundles frames, masks, pose, faces, depth, flow, camera, cuts, references, strengths and temporal settings into one model-independent object. |
| Prompt Compiler | High-level intent → positive/negative prompts ("preserve hands, camera… replace only the watch…"). |
| Generate | Runs any registered backend with chunking, per-chunk cache and resume. |
| Temporal Repair | Checks generated frames against the **source** motion; fixes shimmer and exposure flicker, reports flicker before/after and outlier frames. |
| Identity Lock | Per-frame identity similarity to the character reference; outputs segments to regenerate. |
| Frame Interpolate | Flow-based in-betweening. |
| Compositor | Multiband / alpha / Poisson blending, color + local-lighting match, mask-leakage measurement. |
| Color / Lighting Match | Temporally smoothed LAB transfer; `offset` mode fixes white balance without recoloring the object. |
| Upscale / Restore | Real-ESRGAN or any spandrel model from `models/upscale_models`, capped sharpening. |
| Video Output | H.264 / H.265 / ProRes / VP9 / PNG sequence, original fps and audio (or replacement audio). |
| Save / Load Project | `.genjutsu` files. |
| Genjutsu Score | Identity, motion, camera, background, object, temporal metrics. |

The spec's suggested one-file-per-node layout is consolidated into fewer
modules: pose + depth live in `nodes/pose_depth.py`; flow, camera and scene
analysis in `nodes/video_analysis.py`; segmenter, tracker and mask nodes in
`nodes/segmentation.py`.

## How the hard parts work

**Preservation.** `build_preservation_masks` makes the edit mask (dilated),
its complement (preserve), and a soft mask that is exactly 1 inside the
object, feathers outward, and is forced to 0 on protected occluders. The
compositor then zeroes any blend weight outside the dilated edit area, so
preserved pixels are exactly the original. Tests assert zero leakage.

**Occlusion.** Occluders (e.g. hands) are segmented separately; where their
depth is nearer than the object's median depth they are protected, so the new
object passes behind them. Without explicit occluders, anything clearly nearer
inside the edit band is treated as one.

**Temporal consistency.** Generation runs in overlapping 4n+1-frame chunks
that never straddle a scene cut, cross-faded in the overlaps. Afterwards each
generated frame is compared with its neighbours warped by the *source's*
optical flow. Where they nearly agree (texture boiling) it is pulled toward
the warped neighbour; genuine changes and occlusions (forward-backward check)
are left alone. A second pass removes global exposure pumping.

**Color and lighting.** Statistics are sampled in a ring around the edit on
both the original and generated frames, the transfer parameters are smoothed
over time, and chroma is only offset (never rescaled) so product colors
survive. Low-frequency shading from the plate is re-applied inside the edit.

**Tracking.** Masks propagate by backward-warping with optical flow, then
snap to a foreground/background color model learned on the prompt frame
(essential for flat-colored objects where flow is weak). A track that loses
its area is reported as not visible; re-acquisition looks for object-colored
blobs of plausible size within a radius of the last position that widens the
longer the object is gone.

## Configuration

`config.yaml` covers every component (backend, model ids, segmentation,
depth, pose, flow, identity, temporal, compositing, upscale, chunking,
performance, cache, output). Set `GENJUTSU_CONFIG=/path/to/other.yaml` to
switch profiles.

## Projects, caching, resume

Every expensive stage (frames, flow, masks, pose, depth, camera, each
generated chunk) is keyed by a hash of its inputs and parameters and cached
under the cache root (ComfyUI's temp dir by default). If generation dies on
chunk 7, the next run reuses chunks 1–6 (tested).

A `.genjutsu` file is a zip holding the manifest (operation, prompts,
invariants, strengths, seed/steps/guidance, settings, processing state,
source path + SHA-256), the references (+cut-outs, embeddings), the masks,
depth and camera data, and optionally the source video itself. Loading it
rebuilds the ControlPackage, so a result can be reproduced (tested).

## Using it from Python

```python
from genjutsu.pipeline import GenjutsuPipeline
from genjutsu.references import load_references
from genjutsu.config import load_config

refs = load_references([("sneaker.png", "product", 1.0)])
pipe = GenjutsuPipeline(load_config(), project="ad01")
res = pipe.run("ad.mp4", "product_swap", refs, target="shoe", protect_targets=["hand"],
               out_path="ad_swapped.mp4", project_path="ad01")
print(res.report["score"], res.report["mask_leakage"])
```

### Adding a backend

The rest of the system never imports a model. A backend receives a
`ControlPackage` and returns frames:

```python
from genjutsu.backends.base import VideoBackend, register_backend, REGION_OPS

@register_backend("my_model")
class MyBackend(VideoBackend):
    supports = REGION_OPS

    def generate_region(self, cp):
        # cp.frames, cp.edit_mask, cp.soft_mask, cp.depth, cp.flow_fw,
        # cp.references, cp.prompt, cp.negative_prompt, cp.seed, cp.steps ...
        return frames  # (T,H,W,3) float32 in [0,1]
```

It appears in the Generate node's dropdown automatically. `backends/classical.py`
is a complete, model-free reference implementation (used by the tests).

## Evaluation

`genjutsu.metrics.evaluate` / the **Genjutsu Score** node report, in [0,1]:

| Metric | How |
|---|---|
| identity | ArcFace cosine similarity to the character reference (if InsightFace is installed) |
| motion | output flow vs. source flow (endpoint error relative to motion magnitude) |
| camera | per-frame camera transform agreement |
| background | SSIM inside the preserve mask |
| object | hue/saturation histogram intersection between reference and edited object |
| temporal | flicker score of the output relative to the source |

Metrics whose model is missing are reported as `n/a`, never guessed.

## Tests

```bash
pip install pytest
python -m pytest            # 47 tests, CPU only, ~15 s
```

Covered: mask accuracy and temporal stability, tracking persistence through
occlusion and re-identification, flow accuracy, depth stabilization and layer
order, camera estimation, flicker detection and repair, chunk planning and
merging, compositing leakage and boundaries, chroma-safe color matching,
occlusion protection, video I/O for every codec with audio, cache resume,
project round-trips, prompt compilation, metrics, a full node graph executed
end-to-end, validation of all shipped workflows, and the exact arguments the
Wan backends pass to `WanVACEPipeline` / `WanAnimatePipeline`.

## Status: what is verified and what is not

- **Verified here:** everything model-free (I/O, masks, tracking, flow (DIS),
  camera, temporal engine, compositing, caching, projects, metrics, node
  graph execution, workflow validity), on CPU with synthetic video.
- **Verified against the diffusers 0.40 source, not run on real weights:**
  the Wan 2.2 / VACE / Animate backends. Tests stub diffusers and assert the
  call contract (4n+1 frame counts, /16 resolutions, mask polarity, gray-out
  of regions to regenerate, replace vs. animate mode, offload settings).
  The first real GPU run is the next milestone.
- **Written against upstream APIs, not exercised:** SAM 2, Grounding DINO,
  RAFT, DWPose (rtmlib), Video Depth Anything, InsightFace, spandrel adapters.
  Each falls back cleanly if unavailable.
- **Design choices worth testing on real footage:** VACE with mixed
  subject-pixels + depth control for environment recasts; `recast_control`
  alternatives (depth / pose / edges).

### CPU benchmarks (model-free stages)

81 frames at 832×480 on a 2-vCPU cloud container:

| Stage | Time | per frame |
|---|---|---|
| Optical flow, DIS, forward + backward | 8.1 s | 100 ms |
| Tracking (flow given) | 5.6 s | 69 ms |
| Camera analysis | 1.5 s | 18 ms |
| Composite (multiband + color match) | 12.9 s | 160 ms |
| Temporal repair (bidirectional) | 8.4 s | 103 ms |

GPU generation benchmarks: to be measured.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found` | Install FFmpeg and make sure it is on the `PATH` ComfyUI sees. |
| `diffusers has no WanAnimatePipeline` | `pip install -U "diffusers>=0.40"` in ComfyUI's Python. |
| CUDA OOM during Generate | Lower `max_side`/`max_area`, enable `low_vram`, reduce `chunk_frames` (keep 4n+1), use the 1.3B VACE model. The run resumes from finished chunks. |
| Mask drifts onto the background | Give a tighter box or points; raise the Tracker's `appearance_weight`; install SAM 2. |
| New object appears in front of the hand | Add a `hands` Segmenter into `protect_mask` and connect depth to Preservation Masks. |
| Product colors look wrong | Color Match `chroma: offset` (the compositor already uses it); lower `local_lighting`. |
| Flicker in recasts | Raise Temporal Repair `repair_strength` and `smoothing_strength`; check `outlier_frames` in its report. |
| Seams at chunk boundaries | Increase `overlap_frames`; for character ops Wan-Animate segments internally (`generation.animate_prev_frames`). |
| Character references are classified as product/style | OpenCV 5 removed the built-in face detector used for auto-classification. Set the Reference node's `category` to `character` explicitly. |
| A node is missing | See ComfyUI's console for `[Genjutsu] failed to load nodes.<module>` and run `python install.py --check`. |

## Observed vs. inferred vs. implemented

| | |
|---|---|
| **Observed behavior** (of commercial Genjutsu-style tools) | Swap a character, object or product in real footage while keeping motion, camera and background; recast environments; reference-driven. |
| **Implementation inference** (ours, not known facts about any product) | Such systems likely combine segmentation/tracking, pose and depth controls, a reference-conditioned video diffusion model, and compositing with temporal post-processing. |
| **Open-source implementation** (this repo) | SAM 2, DWPose, Video Depth Anything, RAFT, Wan 2.2 / VACE / Animate via diffusers, and the invariant-driven control, temporal and compositing code here. |

## Roadmap

| Phase | Goal | State |
|---|---|---|
| 1 Foundation | ComfyUI nodes, video I/O, FFmpeg, Wan 2.2, output | Done (GPU run pending) |
| 2 Video understanding | SAM 2, DWPose, depth, RAFT, tracking → ControlPackage | Done (adapters + fallbacks) |
| 3 Motion transfer | Wan2.2-Animate animate mode | Wired, GPU validation pending |
| 4 Object swap | segmentation, tracking, VACE region generation, preservation, compositing | Done (classical path verified) |
| 5 Recast | environment / style / full recast | Wired, GPU validation pending |
| 6 Temporal engine | flow stabilization, flicker detection/repair, identity lock, occlusion | Done; automatic identity-driven regeneration next |
| 7 Optimization | caching, chunking, offload, resume | Done; latent caching and batch queueing next |

## License

GPL-3.0 (see `LICENSE`). Third-party code and model weights keep their own
licenses, which differ and in some cases prohibit commercial use; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
