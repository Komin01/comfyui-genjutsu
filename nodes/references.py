from __future__ import annotations

import json

import numpy as np

from ._util import CATEGORY, REF_CATEGORIES, REFS, to_mask, to_np


class GenjutsuReference:
    """Add one reference image to a (chainable) Reference Package. Chain as
    many as needed: character + outfit + location + style + product..."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "category": (REF_CATEGORIES, {"default": "auto"}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "name": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": "", "multiline": True, "tooltip": "optional words the prompt compiler uses for this reference"}),
                "compute_embeddings": ("BOOLEAN", {"default": True, "tooltip": "identity (InsightFace) / appearance (CLIP) features when installed"}),
            },
            "optional": {"references": (REFS,), "mask": ("MASK", {"tooltip": "subject cut-out; auto-computed if omitted"})},
        }

    RETURN_TYPES = (REFS, "MASK", "STRING")
    RETURN_NAMES = ("references", "cutout_mask", "info")
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, image, category, weight, name, description, compute_embeddings, references=None, mask=None):
        from genjutsu.controls import ReferencePackage
        from genjutsu.references import build_reference

        from ._util import get_config

        img = to_np(image)[0]
        ref = build_reference(img, category, weight, name, compute_embeddings, get_config().get("device"))
        if mask is not None:
            m = to_np(mask)
            ref.mask = (m[0] if m.ndim == 3 else m).astype(np.float32)
        if description.strip():
            ref.metadata["description"] = description.strip()
        pkg = (references or ReferencePackage()).add(ref)
        info = {"category": ref.category.value, "auto_category": ref.metadata["auto_category"], "weight": ref.weight,
                "has_embedding": ref.embedding is not None, "analysis": ref.metadata["analysis"], "total_references": len(pkg)}
        cut = ref.mask if ref.mask is not None else np.ones(img.shape[:2], np.float32)
        return (pkg, to_mask(cut[None]), json.dumps(info, indent=2))


NODE_CLASS_MAPPINGS = {"GenjutsuReference": GenjutsuReference}
NODE_DISPLAY_NAME_MAPPINGS = {"GenjutsuReference": "Genjutsu Reference"}
