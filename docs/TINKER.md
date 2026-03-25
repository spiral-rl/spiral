# SPIRAL Tinker Addon

Simplified synchronous training for SPIRAL using the [Tinker](https://github.com/tinker-engine/tinker) distributed training framework.

## Overview

The `tinker/` addon provides a streamlined interface for running SPIRAL self-play RL training. It wraps the core `spiral.tinker` implementation with:

- **Sync-only training** — no population-based self-play (FSP), no async actor-learner
- **Flat module layout** — `tinker.training`, `tinker.env`, `tinker.rollouts`, etc.
- **Ready-to-use bash scripts** — in `cmd/tinker/`

## Quick Start

### Prerequisites

```bash
pip install -e ".[tinker]"
```

This installs the core SPIRAL package plus Tinker dependencies (`tinker`, `tinker-cookbook`, `chz`).

### Required Environment Variables

```bash
export TINKER_API_KEY=your_tinker_key
export WANDB_API_KEY=your_wandb_key
export OPENROUTER_API_KEY=your_openrouter_key   # for LLM eval opponents
```

### Run Training

**Qwen3-4B (recommended for quick experiments):**

```bash
bash cmd/tinker/run_tinker_qwen3_4b.sh
```

**Llama-3-8B:**

```bash
bash cmd/tinker/run_tinker_llama_8b.sh
```

Both scripts accept overrides via environment variables:

```bash
BATCH_SIZE=64 LR=1e-4 ENV_IDS=KuhnPoker-v1 bash cmd/tinker/run_tinker_qwen3_4b.sh
```

### Direct Python Usage

```bash
python train_spiral_tinker.py \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    renderer_name=qwen3 \
    lora_rank=32 \
    env_ids='KuhnPoker-v1' \
    use_llm_obs_wrappers='True' \
    batch_size=128 \
    num_train_datapoints=12800 \
    learning_rate=4e-5 \
    max_tokens=8192
```

## Architecture

```
tinker/                        Top-level addon (simplified interface)
├── __init__.py                Proxy + lazy exports
├── training.py                Sync training loop (do_sync_training_spiral)
├── dataset.py                 SpiralRLDataset / SpiralRLDatasetBuilder
├── renderer.py                SpiralRenderer (boxed-action parsing)
├── env.py                     TwoPlayerCoordinator + SpiralTwoPlayerEnv
├── rollouts.py                Trajectory collection with draw retry
├── train_step.py              Per-turn RAE advantage computation
└── utils.py                   Logging, metrics

spiral/                        Core package (unchanged)
├── core/                      Shared components (envs, agents, utils)
├── oat/                       OAT backend
└── tinker/                    Full Tinker backend (FSP, async, eval)
```

### Import Paths

```python
# Addon (simplified)
from tinker.training import create_spiral_train_loop

# Core shared components
from spiral.core import make_env, RandomAgent, EMA

# Full Tinker backend (with FSP, async, eval)
from spiral.tinker.training import PopulationManager
```

## Training Pipeline

1. **Config** — `SpiralConfig` dataclass parsed via `chz` from CLI args
2. **Dataset** — `SpiralRLDatasetBuilder` creates environment group builders for each game
3. **Rollout** — Self-play trajectories collected with optional draw retry
4. **Train step** — Per-turn advantages computed using Role-conditioned Advantage Estimation (RAE)
5. **Eval** — Periodic async evaluation against random / LLM opponents

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Must be even (2 players per game) |
| `learning_rate` | 4e-5 | LoRA learning rate |
| `lora_rank` | 32 | LoRA adapter rank |
| `max_tokens` | 8192 | Max generation length |
| `filter_draw` | False | Retry games that end in draws |
| `max_draw_retries` | 5 | Max retries before accepting draw |
| `use_role_baseline` | True | RAE: subtract per-role EMA baseline |
| `role_baseline_ema_gamma` | 0.95 | EMA decay for role baselines |
| `gamma` | 1.0 | Discount factor for earlier turns |
| `eval_every` | 16 | Evaluate every N batches |

### Supported Games

- `TicTacToe-v0` — Classic tic-tac-toe
- `KuhnPoker-v1` — Simplified poker
- `SimpleNegotiation-v1` / `v2` — Resource negotiation
- `LiarsDice-v1` — Bluffing dice
- `TruthAndDeception-v1` — Deception game
- `PigDice-v1` — Pig dice
- `WordChains-v1` — Word chain
- `SpellingBee-v1` — Spelling bee
- `SimpleBlindAuction-v1` — Blind auction

## What This Addon Does NOT Include

- **Population-based self-play (FSP)** — use `spiral.tinker.training.population` directly
- **Async actor-learner** — use `spiral.tinker.async_actor_learner`
- **Complex eval runner** — use `spiral.tinker.eval` directly
- **Math test evaluation** — available in `spiral.tinker.eval.math_test`
