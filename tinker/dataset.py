"""Simplified dataset builder for SPIRAL multi-environment training.

Re-exports ``SpiralRLDataset`` and ``SpiralRLDatasetBuilder`` from
``spiral.tinker.dataset``.
"""

from spiral.tinker.dataset import SpiralRLDataset, SpiralRLDatasetBuilder

__all__ = [
    "SpiralRLDataset",
    "SpiralRLDatasetBuilder",
]
