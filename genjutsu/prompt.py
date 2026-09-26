"""Prompt compiler: turns high-level intent (operation, target, preserve
list, references) into positive/negative prompts suited to video diffusion
models (section 16). Users never have to hand-write invariance prose."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .controls import Operation, RefCategory, ReferencePackage

PRESERVE_PHRASES = {
    "person": "the person's identity, face and body proportions",
    "identity": "the subject's identity and facial features",
    "hand": "the hands and their exact finger positions",
    "hands": "the hands and their exact finger positions",
    "background": "the original background and environment",
    "camera": "the original camera trajectory, framing and lens",
    "lighting": "the original lighting direction, color temperature and shadows",
    "motion": "the original body movement and timing",
    "expression": "the facial expressions",
    "composition": "the original composition",
    "clothing": "the original clothing",
}

OPERATION_TEMPLATES = {
    Operation.OBJECT_SWAP: "Replace only the {target} with {ref}. Keep correct geometry, scale, perspective, reflections, contact shadows and occlusion.",
    Operation.PRODUCT_SWAP: "Replace only the {target} with {ref}, reproducing its exact shape, materials, colors, label and branding. Keep correct perspective, reflections and contact shadows.",
    Operation.OUTFIT_SWAP: "Change only the clothing to {ref}, with natural fabric folds and physically plausible motion following the body.",
    Operation.CHARACTER_SWAP: "Replace the performer with {ref}, performing the exact same movement, pose and facial expressions.",
    Operation.MOTION_TRANSFER: "{ref} performing the movement, with natural body dynamics and consistent anatomy.",
    Operation.ENVIRONMENT_RECAST: "The same subject performing the same action in {env}. Rebuild the architecture, set dressing, lighting and atmosphere of the new environment.",
    Operation.STYLE_RECAST: "The same scene re-rendered in the style of {style}.",
    Operation.FULL_RECAST: "{ref} performing the same action in {env}, rendered in the style of {style}.",
}

BASE_NEGATIVE = (
    "flicker, temporal inconsistency, morphing, warping, jitter, identity drift, extra fingers, deformed hands, "
    "duplicated objects, floating objects, wrong scale, blurry, low quality, oversaturated, watermark, text artifacts, "
    "static frame, frozen motion"
)

QUALITY_SUFFIX = "Photorealistic, temporally stable, sharp detail, natural motion blur."


@dataclass
class PromptSpec:
    operation: Operation
    target: str = ""
    preserve: list[str] = field(default_factory=list)
    motion: str = "original"  # original | free
    user_prompt: str = ""
    reference_descriptions: dict = field(default_factory=dict)  # category -> text
    style: str = ""
    environment: str = ""
    extra_negative: str = ""


def _describe(refs: Optional[ReferencePackage], cats: tuple, fallback: str, descriptions: dict) -> str:
    for c in cats:
        if descriptions.get(c.value):
            return descriptions[c.value]
    if refs is not None:
        r = refs.primary(*cats)
        if r is not None:
            return f"the {r.category.value} shown in the reference image" + (f" ({r.name})" if r.name else "")
    return fallback


def compile_prompt(spec: PromptSpec, refs: Optional[ReferencePackage] = None) -> tuple[str, str]:
    op = Operation(spec.operation)
    d = spec.reference_descriptions
    target = spec.target.strip().lower() or {"object_swap": "selected object", "product_swap": "product"}.get(op.value, "subject")
    ref = _describe(
        refs,
        {
            Operation.PRODUCT_SWAP: (RefCategory.PRODUCT, RefCategory.OBJECT),
            Operation.OBJECT_SWAP: (RefCategory.OBJECT, RefCategory.PRODUCT, RefCategory.PROP),
            Operation.OUTFIT_SWAP: (RefCategory.OUTFIT,),
        }.get(op, (RefCategory.CHARACTER,)),
        "the reference",
        d,
    )
    env = spec.environment or _describe(refs, (RefCategory.LOCATION,), "the new environment", d)
    style = spec.style or _describe(refs, (RefCategory.STYLE, RefCategory.TEXTURE), "the reference style", d)

    parts = []
    preserve = list(dict.fromkeys(p.lower() for p in spec.preserve))
    if spec.motion == "original" and "motion" not in preserve and op not in (Operation.STYLE_RECAST,):
        preserve.insert(0, "motion")
    if preserve:
        phrases = [PRESERVE_PHRASES.get(p, f"the original {p}") for p in preserve]
        parts.append("Preserve " + ", ".join(phrases[:-1]) + (" and " if len(phrases) > 1 else "") + phrases[-1] + ".")
    parts.append(OPERATION_TEMPLATES[op].format(target=target, ref=ref, env=env, style=style))
    if spec.user_prompt.strip():
        parts.append(spec.user_prompt.strip().rstrip(".") + ".")
    if op not in (Operation.STYLE_RECAST,):
        parts.append(QUALITY_SUFFIX)
    neg = BASE_NEGATIVE
    if op in (Operation.OBJECT_SWAP, Operation.PRODUCT_SWAP):
        neg += f", original {target} still visible, mixed old and new object, distorted logo"
    if op in (Operation.CHARACTER_SWAP, Operation.MOTION_TRANSFER, Operation.FULL_RECAST):
        neg += ", face swap artifacts, inconsistent face, changing hairstyle, changing outfit between frames"
    if op in (Operation.ENVIRONMENT_RECAST, Operation.FULL_RECAST):
        neg += ", original background visible, camera moving differently"
    if spec.extra_negative:
        neg += ", " + spec.extra_negative
    return " ".join(parts), neg
