"""ComfyUI node registrations. Each module is imported independently so a
missing optional dependency disables only the affected nodes."""
import importlib
import logging

log = logging.getLogger("genjutsu.nodes")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
IMPORT_ERRORS = {}

_MODULES = [
    "video_loader", "video_analysis", "segmentation", "pose_depth", "references", "control_package",
    "wan_generator", "temporal", "compositor", "video_output", "pipeline_node",
]

for _m in _MODULES:
    try:
        mod = importlib.import_module(f"{__name__}.{_m}")
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:  # pragma: no cover
        IMPORT_ERRORS[_m] = repr(e)
        log.warning("[Genjutsu] failed to load nodes.%s: %s", _m, e)
