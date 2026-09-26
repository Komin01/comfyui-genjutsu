"""Reference understanding: load reference images, infer what each one
represents (identity / product / environment / style ...), cut out the
subject, and compute reusable features."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .controls import RefCategory, Reference, ReferencePackage
from .video import to_uint8

log = logging.getLogger("genjutsu.references")

_HAAR = None


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb * alpha + (1 - alpha)  # composite on white, product-shot style
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _faces(img: np.ndarray) -> list:
    global _HAAR
    if _HAAR is None:
        _HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    m = max(24, min(g.shape) // 12)
    return list(_HAAR.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(m, m)))


def border_uniformity(img: np.ndarray, frac: float = 0.06) -> float:
    """1 = plain studio background (typical product / cut-out reference)."""
    H, W = img.shape[:2]
    b = max(2, int(min(H, W) * frac))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3), img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    return float(np.exp(-border.std(0).mean() / 0.05))


def classify_reference(img: np.ndarray) -> tuple[RefCategory, dict]:
    """Heuristic categorization; the user can always override in the node.
    - a clear face covering a meaningful area -> CHARACTER
    - plain uniform border (studio shot)     -> PRODUCT
    - wide, busy, no dominant subject         -> LOCATION
    - otherwise                               -> STYLE"""
    H, W = img.shape[:2]
    faces = _faces(img)
    face_frac = max((w * h for (_, _, w, h) in faces), default=0) / float(H * W)
    uni = border_uniformity(img)
    edges = cv2.Canny(to_uint8(img), 80, 160).mean() / 255.0
    info = {"faces": len(faces), "face_area_frac": face_frac, "border_uniformity": uni, "edge_density": float(edges)}
    if faces and face_frac > 0.01:
        return RefCategory.CHARACTER, info
    if uni > 0.6:
        return RefCategory.PRODUCT, info
    if W >= H and edges > 0.06:
        return RefCategory.LOCATION, info
    return RefCategory.STYLE, info


def cutout(img: np.ndarray) -> np.ndarray:
    """Subject mask: rembg if installed, else GrabCut seeded from an inset
    rectangle (works well for product shots on plain backgrounds)."""
    try:
        from rembg import remove

        rgba = remove(to_uint8(img))
        return (np.asarray(rgba)[..., 3].astype(np.float32) / 255.0)
    except Exception:
        pass
    from .segmentation import grabcut

    H, W = img.shape[:2]
    return grabcut(img, [W * 0.04, H * 0.04, W * 0.96, H * 0.96])


def color_signature(img: np.ndarray, mask: Optional[np.ndarray] = None, hue_bins: int = 18, sat_bins: int = 4) -> np.ndarray:
    """Cheap appearance feature: smoothed hue/saturation histogram (robust to
    exposure changes and compression, sensitive to actual object color)."""
    hsv = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2HSV)
    m = None if mask is None else ((mask > 0.5) * 255).astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1], m, [hue_bins, sat_bins], [0, 180, 0, 256])
    # hue is circular: pad rows cyclically before smoothing
    hp = np.concatenate([h[-1:], h, h[:1]], 0)
    h = cv2.GaussianBlur(hp, (3, 3), 0.8, borderType=cv2.BORDER_REFLECT)[1:-1].ravel()
    return (h / (h.sum() + 1e-8)).astype(np.float32)


_CLIP = None


def clip_embedding(img: np.ndarray, model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda") -> Optional[np.ndarray]:
    global _CLIP
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if _CLIP is None:
            dev = device if torch.cuda.is_available() else "cpu"
            _CLIP = (CLIPModel.from_pretrained(model_id).to(dev).eval(), CLIPProcessor.from_pretrained(model_id), dev)
        model, proc, dev = _CLIP
        inputs = proc(images=Image.fromarray(to_uint8(img)), return_tensors="pt").to(dev)
        with torch.inference_mode():
            e = model.get_image_features(**inputs)[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)
    except Exception as e:
        log.info("CLIP embedding unavailable: %s", e)
        return None


def build_reference(
    image: np.ndarray,
    category: Optional[RefCategory | str] = None,
    weight: float = 1.0,
    name: str = "",
    compute_embeddings: bool = False,
    device: str = "cuda",
) -> Reference:
    auto_cat, info = classify_reference(image)
    cat = RefCategory(category) if category not in (None, "", "auto") else auto_cat
    mask = None
    if cat in (RefCategory.PRODUCT, RefCategory.OBJECT, RefCategory.PROP, RefCategory.CHARACTER, RefCategory.OUTFIT, RefCategory.LOGO):
        try:
            mask = cutout(image)
        except Exception as e:
            log.warning("cutout failed: %s", e)
    emb = None
    if compute_embeddings:
        if cat == RefCategory.CHARACTER:
            try:
                from .identity import face_embedding

                emb = face_embedding(image, device)
            except Exception as e:
                log.info("identity embedding unavailable: %s", e)
        if emb is None:
            emb = clip_embedding(image, device=device)
    meta = {"auto_category": auto_cat.value, "analysis": info, "color_signature": color_signature(image, mask).tolist()}
    return Reference(image=image.astype(np.float32), category=cat, weight=float(weight), name=name, mask=mask, embedding=emb, metadata=meta)


def load_references(paths_and_categories: list[tuple[str, Optional[str], float]], **kw) -> ReferencePackage:
    pkg = ReferencePackage()
    for path, cat, w in paths_and_categories:
        pkg = pkg.add(build_reference(load_image(path), cat, w, name=Path(path).stem, **kw))
    return pkg


def reference_on_plain(ref: Reference, bg: float = 1.0) -> np.ndarray:
    """Reference with its background replaced by a flat color. Video models
    copy reference backgrounds into the scene; this prevents that."""
    if ref.mask is None:
        return ref.image
    a = ref.mask[..., None]
    return (ref.image * a + bg * (1 - a)).astype(np.float32)
