# Third-party licenses

Genjutsu-OSS itself is licensed under **GPL-3.0** (see `LICENSE`), matching
ComfyUI, whose modules (`folder_paths`, `comfy.utils`) the nodes import.

**Code and model weights are licensed separately.** A permissive code license
says nothing about the weights it downloads. "Open source" does not mean
"free for commercial redistribution". Verify each entry yourself before any
commercial use or redistribution; licenses change, and this table is not
legal advice.

Status of each entry: checked against the upstream repository's LICENSE /
README on 2026-09-25 where marked ✔, otherwise taken from the package
metadata.

## Generation models

| Component | Repository | Code license | Weights license | Commercial use | Attribution / notes |
|---|---|---|---|---|---|
| Wan 2.2 (T2V-A14B, I2V-A14B, TI2V-5B, Animate-14B) ✔ | github.com/Wan-Video/Wan2.2 | Apache-2.0 | Apache-2.0 | Allowed | README adds use restrictions (no unlawful / harmful content, no targeting vulnerable groups). Keep NOTICE + license. |
| Wan 2.1 VACE (1.3B / 14B) | github.com/ali-vilab/VACE, Wan-AI on Hugging Face | Apache-2.0 | Apache-2.0 | Allowed | Used for region edits and recasts. |
| diffusers | github.com/huggingface/diffusers | Apache-2.0 | — | Allowed | |
| transformers / accelerate / huggingface_hub | github.com/huggingface | Apache-2.0 | — | Allowed | |

## Video understanding

| Component | Repository | Code license | Weights license | Commercial use | Notes |
|---|---|---|---|---|---|
| SAM 2 / 2.1 ✔ | github.com/facebookresearch/sam2 | Apache-2.0 | Apache-2.0 | Allowed | Some bundled components are BSD-3-Clause. |
| Grounding DINO ✔ | github.com/IDEA-Research/GroundingDINO | Apache-2.0 | Apache-2.0 | Allowed | Used via transformers for text prompts. |
| Video Depth Anything ✔ | github.com/DepthAnything/Video-Depth-Anything | Apache-2.0 | **Small: Apache-2.0. Base/Large: CC-BY-NC-4.0** | **Small only** | Genjutsu defaults to `vits` (Small) for this reason. |
| Depth Anything V2 (fallback) | github.com/DepthAnything/Depth-Anything-V2 | Apache-2.0 | Small: Apache-2.0; Base/Large/Giant: CC-BY-NC-4.0 | Small only | Fallback uses the Small model. |
| DWPose via rtmlib ✔ | github.com/Tau-J/rtmlib, github.com/IDEA-Research/DWPose | Apache-2.0 | Apache-2.0 (RTMPose/DWPose ONNX) | Allowed | |
| RAFT (torchvision) | github.com/pytorch/vision | BSD-3-Clause | BSD-3-Clause (torchvision weights) | Allowed | Weights trained on academic flow datasets; check your policy. |
| InsightFace ✔ | github.com/deepinsight/insightface | MIT | **Non-commercial research only** (buffalo_l etc.) | **No** | Identity Lock only. Disable (`identity.enabled: false`) or substitute a commercially licensed face model. |
| CLIP (optional embeddings) | github.com/openai/CLIP | MIT | MIT | Allowed | |
| rembg | github.com/danielgatis/rembg | MIT | u2net: Apache-2.0 | Allowed | Other rembg models have their own licenses. |

## Restoration and I/O

| Component | Repository | License | Commercial use | Notes |
|---|---|---|---|---|
| Real-ESRGAN ✔ | github.com/xinntao/Real-ESRGAN | BSD-3-Clause (code and official weights) | Allowed | Third-party ESRGAN fine-tunes each have their own license. |
| spandrel ✔ | github.com/chaiNNer-org/spandrel | MIT | Allowed | Model loader. |
| FFmpeg | ffmpeg.org | LGPL-2.1+ (GPL-2.0+ if built with libx264 / libx265) | Allowed with conditions | H.264/H.265 via x264/x265 makes the FFmpeg build GPL; codec patents are a separate question. |
| OpenCV | github.com/opencv/opencv | Apache-2.0 | Allowed | |
| NumPy / SciPy | numpy.org | BSD-3-Clause | Allowed | |
| PyTorch / torchvision | pytorch.org | BSD-3-Clause | Allowed | |
| Pillow | python-pillow.org | MIT-CMU (HPND) | Allowed | |
| PyYAML | pyyaml.org | MIT | Allowed | |
| ComfyUI ✔ | github.com/comfyanonymous/ComfyUI | GPL-3.0 | Allowed (copyleft) | Host application. |

## Commercial-use checklist

For a fully commercial-safe local setup:

1. `depth.checkpoint: vits` (the default).
2. `identity.enabled: false`, or replace InsightFace with a commercially licensed face-embedding model.
3. Use official Real-ESRGAN weights, or check the license of any community upscaler.
4. Keep Apache-2.0 NOTICE files and license texts when redistributing Wan, SAM 2, VACE or Grounding DINO weights.
5. Check that your FFmpeg build's license fits your distribution model.
