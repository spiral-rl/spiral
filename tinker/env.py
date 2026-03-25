"""Two-player game environment for SPIRAL self-play training with Tinker.

Re-exports the core environment components from ``spiral.tinker.training.env``.
"""

from spiral.tinker.training.env import (
    ILLEGAL_MOVE_REWARD,
    SpiralTwoPlayerEnv,
    SpiralTwoPlayerEnvGroupBuilder,
    TwoPlayerCoordinator,
)

__all__ = [
    "ILLEGAL_MOVE_REWARD",
    "SpiralTwoPlayerEnv",
    "SpiralTwoPlayerEnvGroupBuilder",
    "TwoPlayerCoordinator",
]
