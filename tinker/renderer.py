"""Renderer for SPIRAL game observations.

Re-exports the renderer components from ``spiral.tinker.renderer``
and extends ``RENDERER_NAME_MAP`` with additional model aliases.
"""

from spiral.tinker.renderer import (
    INVALID_ACTION,
    RENDERER_NAME_MAP,
    SpiralRenderer,
    get_spiral_renderer,
)

# Extend RENDERER_NAME_MAP with aliases for additional model families.
# GPT-OSS reasoning models use the Harmony chat format from tinker-cookbook.
# Qwen3-Base models share the same tokenizer/format as Qwen3-Instruct.
RENDERER_NAME_MAP.update({
    "gpt_oss": "gpt_oss_medium_reasoning",
    "gpt_oss_low": "gpt_oss_low_reasoning",
    "gpt_oss_high": "gpt_oss_high_reasoning",
    "gpt_oss_no_sysprompt": "gpt_oss_no_sysprompt",
    "qwen3_base": "qwen3",
})

__all__ = [
    "INVALID_ACTION",
    "RENDERER_NAME_MAP",
    "SpiralRenderer",
    "get_spiral_renderer",
]
