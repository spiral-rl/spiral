"""Renderer for SPIRAL game observations.

Re-exports the renderer components from ``spiral.tinker.renderer``.
"""

from spiral.tinker.renderer import (
    INVALID_ACTION,
    RENDERER_NAME_MAP,
    SpiralRenderer,
    get_spiral_renderer,
)

__all__ = [
    "INVALID_ACTION",
    "RENDERER_NAME_MAP",
    "SpiralRenderer",
    "get_spiral_renderer",
]
