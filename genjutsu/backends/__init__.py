"""Video generation backends behind a single abstraction (section 31).

Nothing else in Genjutsu imports a model directly: the pipeline builds a
ControlPackage and hands it to whichever backend is configured.
"""
from .base import VideoBackend, available_backends, get_backend, register_backend  # noqa: F401
from . import classical  # noqa: F401  (registers "classical")
from . import wan  # noqa: F401  (registers "wan22", "wan_animate")
