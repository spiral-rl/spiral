"""SPIRAL Tinker addon - simplified sync training for the Tinker backend.

This top-level package provides a streamlined interface for SPIRAL
self-play training using Tinker. It wraps ``spiral.core`` shared
components and offers sync-only training (no FSP, no async
actor-learner).

Because this directory shadows the pip-installed ``tinker`` package, the
proxy below loads the pip package's exports so that existing code like
``from tinker import ModelInput`` continues to work.
"""

import importlib
import importlib.util
import os
import sys

# ── Proxy: expose pip-installed tinker alongside this local package ──────────


def _load_pip_tinker():
    """Find and load the pip-installed ``tinker`` package from site-packages."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)

    # 1. Remove local paths from sys.path so importlib finds the pip package
    removed: list[tuple[int, str]] = []
    idx = 0
    while idx < len(sys.path):
        p_abs = os.path.abspath(sys.path[idx]) if sys.path[idx] else os.path.abspath(".")
        if p_abs == parent_dir:
            removed.append((idx, sys.path.pop(idx)))
        else:
            idx += 1

    # 2. Remove local tinker from sys.modules
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key == "tinker" or key.startswith("tinker."):
            saved[key] = sys.modules.pop(key)

    pip_mod = None
    try:
        pip_mod = importlib.import_module("tinker")
    except ImportError:
        pass
    finally:
        # Capture pip submodules before restoring local
        pip_submodules = {
            k: v for k, v in sys.modules.items() if k.startswith("tinker.")
        }

        # 3. Restore sys.path
        for i, p in sorted(removed, key=lambda x: x[0]):
            sys.path.insert(i, p)

        # 4. Restore local module in sys.modules (overwrites pip entry)
        for key, mod in saved.items():
            sys.modules[key] = mod

        # 5. Keep pip submodules accessible (don't overwrite local ones)
        for key, mod in pip_submodules.items():
            if key not in sys.modules:
                sys.modules[key] = mod

    return pip_mod


_pip_tinker = _load_pip_tinker()

if _pip_tinker is not None:
    # Copy public attributes (ModelInput, TrainingClient, etc.)
    for _attr in dir(_pip_tinker):
        if not _attr.startswith("_"):
            globals()[_attr] = getattr(_pip_tinker, _attr)

    # Extend __path__ so pip submodules (tinker.types, etc.) are reachable
    if hasattr(_pip_tinker, "__path__"):
        for _p in _pip_tinker.__path__:
            if _p not in __path__:
                __path__.append(_p)

# ── SPIRAL Tinker addon public API ───────────────────────────────────────────

__all__ = [
    "create_spiral_train_loop",
    "do_sync_training_spiral",
    "SpiralRLDataset",
    "SpiralRLDatasetBuilder",
    "SpiralRenderer",
    "get_spiral_renderer",
    "TwoPlayerCoordinator",
    "SpiralTwoPlayerEnv",
    "SpiralTwoPlayerEnvGroupBuilder",
    "do_single_rollout",
    "do_group_rollout",
    "do_group_rollout_with_draw_retry",
    "train_step",
    "setup_logging",
    "compute_trajectory_metrics",
]


def __getattr__(name: str):
    """Lazy-load addon symbols to avoid circular imports at startup."""
    _lazy_map = {
        "create_spiral_train_loop": "tinker.training",
        "do_sync_training_spiral": "tinker.training",
        "SpiralRLDataset": "tinker.dataset",
        "SpiralRLDatasetBuilder": "tinker.dataset",
        "SpiralRenderer": "tinker.renderer",
        "get_spiral_renderer": "tinker.renderer",
        "TwoPlayerCoordinator": "tinker.env",
        "SpiralTwoPlayerEnv": "tinker.env",
        "SpiralTwoPlayerEnvGroupBuilder": "tinker.env",
        "do_single_rollout": "tinker.rollouts",
        "do_group_rollout": "tinker.rollouts",
        "do_group_rollout_with_draw_retry": "tinker.rollouts",
        "train_step": "tinker.train_step",
        "setup_logging": "tinker.utils",
        "compute_trajectory_metrics": "tinker.utils",
    }
    if name in _lazy_map:
        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
