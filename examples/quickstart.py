"""Model-free smoke test of the whole pipeline (no GPU, no downloads).

    python examples/quickstart.py

Builds a synthetic clip (panning camera, moving red cube, audio tone),
swaps the cube for a blue-disc "product" with the classical backend, and
writes the result, a .genjutsu project and a GenjutsuScore to examples/out/.
Swap `backend="classical"` for "wan22" on a GPU machine to run the real model.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))

from synth import make_frames, reference_image, write_with_audio  # noqa: E402

from genjutsu.config import load_config  # noqa: E402
from genjutsu.controls import ReferencePackage  # noqa: E402
from genjutsu.pipeline import GenjutsuPipeline  # noqa: E402
from genjutsu.references import build_reference  # noqa: E402

out = os.path.join(HERE, "out")
os.makedirs(out, exist_ok=True)
frames, masks, boxes = make_frames(T=48, H=192, W=256, obj=32, obj_speed=2.0)
src = write_with_audio(frames, os.path.join(out, "source.mp4"), fps=24)

cfg = load_config().with_overrides({"backend": "classical", "segmentation": {"model": "grabcut"},
                                    "optical_flow": {"model": "dis"}, "cache": {"root": os.path.join(out, "cache")}})
product = build_reference(reference_image(128), "product", 1.0, "blue_disc")
x0, y0, x1, y1 = boxes[0]
res = GenjutsuPipeline(cfg, "quickstart").run(
    src, "product_swap", ReferencePackage([product]), box=[x0 - 4, y0 - 4, x1 + 4, y1 + 4],
    out_path=os.path.join(out, "product_swap.mp4"), project_path=os.path.join(out, "product_swap"),
)
print("video:  ", res.output_path)
print("project:", res.project_path)
print("leakage:", res.report["mask_leakage"])
from genjutsu.metrics import GenjutsuScore  # noqa: E402

print(GenjutsuScore(**{k: v for k, v in res.report["score"].items() if k != "overall"}).table())
