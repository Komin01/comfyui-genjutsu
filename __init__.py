"""ComfyUI-Genjutsu: open-source, Genjutsu-style video transformation nodes.

This is an independent open-source project. It is not affiliated with, and
does not contain code or models from, Higgsfield or its Genjutsu product."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

if __package__:  # loaded by ComfyUI as a custom-node package
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402
else:  # imported as a plain module (e.g. by test collectors)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
