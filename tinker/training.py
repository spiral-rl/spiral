"""Simplified synchronous training loop for SPIRAL with Tinker.

This module provides ``create_spiral_train_loop`` and
``do_sync_training_spiral`` — a streamlined sync-only training pipeline
with no population-based self-play (FSP) and no async actor-learner.
"""

import asyncio
import logging
import re
import time
from typing import Optional

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import scope
from tqdm.asyncio import tqdm

from spiral.core.checkpoint_state import (
    get_checkpoint_state,
    get_checkpoint_state_path,
    register_checkpoint_state,
    restore_checkpoint_state,
    save_checkpoint_state,
)
from spiral.tinker.eval.evaluator import AsyncEvalRunner
from spiral.tinker.training.env import SpiralTwoPlayerEnvGroupBuilder
from spiral.tinker.training.rollouts import (
    do_group_rollout_and_filter_constant_reward,
)
from spiral.tinker.training.train_step import train_step as spiral_train_step
from spiral.tinker.utils import convert_to_json_serializable, setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@scope
async def do_sync_training_spiral(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: train.Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list,
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    eval_runner: Optional[AsyncEvalRunner],
    tokenizer: Tokenizer,
):
    """Synchronous on-policy training with draw retry (no FSP, no async).

    Each iteration:
      1. Save checkpoint (if due).
      2. Collect self-play trajectories with optional draw retry.
      3. Run custom SPIRAL train step (RAE advantages).
      4. Log metrics.
    """
    for i_batch in range(start_batch, end_batch):
        sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
            training_client, i_batch + 1, cfg.log_path, cfg.save_every
        )
        if i_batch % cfg.save_every == 0:
            logger.info(f"Saved checkpoint at batch {i_batch}")
            state_path = get_checkpoint_state_path(
                cfg.log_path, i_batch, name="training_state"
            )
            save_checkpoint_state(state_path)

        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Schedule async evaluation
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0 and eval_runner:
            eval_runner.schedule_eval(evaluators, sampling_client, i_batch)

        # Collect self-play trajectories
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            rollout_tasks = [
                asyncio.create_task(
                    do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        max_tokens=cfg.max_tokens,
                        do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                    ),
                    name=f"sample_task_{i}",
                )
                for i, builder in enumerate(env_group_builders_P)
            ]
            trajectory_groups_P: list[TrajectoryGroup] = await tqdm.gather(
                *rollout_tasks,
                desc=f"Batch {i_batch} rollouts",
            )

        logger.info(
            f"Training step {i_batch} with {len(trajectory_groups_P)} trajectory groups"
        )

        # Train step with RAE advantages
        train_step_metrics = await spiral_train_step(
            cfg,
            i_batch,
            training_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            sampling_client,
        )

        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        metrics = convert_to_json_serializable(metrics)
        ml_logger.log_metrics(metrics, step=i_batch)


async def create_spiral_train_loop(cfg: train.Config):
    """Main training loop for SPIRAL using Tinker (sync-only, no FSP).

    1. Set up logging (training + optional separate eval logger).
    2. Resume from checkpoint if available.
    3. Create LoRA training client.
    4. Build dataset and evaluators.
    5. Run ``do_sync_training_spiral``.
    """
    # Set up loggers
    ml_logger = setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    eval_logger = None
    eval_runner = None
    if cfg.eval_every > 0 and cfg.wandb_project:
        eval_wandb_name = f"{cfg.wandb_name}_eval" if cfg.wandb_name else None
        eval_logger = setup_logging(
            log_dir=cfg.log_path,
            wandb_project=cfg.wandb_project,
            config=cfg,
            wandb_name=eval_wandb_name,
            reinit="create_new",
        )
        eval_runner = AsyncEvalRunner(eval_logger)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Determine start batch
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    elif hasattr(cfg, "load_checkpoint_path") and cfg.load_checkpoint_path:
        match = re.search(r"/weights/(\d{6})$", cfg.load_checkpoint_path)
        start_batch = int(match.group(1)) if match else 0
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    register_checkpoint_state(
        "role_baselines", SpiralTwoPlayerEnvGroupBuilder._role_baseline_emas
    )

    # Resume or create training client
    load_state_path = (
        resume_info["state_path"] if resume_info else getattr(cfg, "load_checkpoint_path", None)
    )
    if load_state_path:
        logger.info(f"Resuming from {load_state_path} at batch {start_batch}")
        training_client = await service_client.create_training_client_from_state_async(
            path=load_state_path, user_metadata=None
        )
        if start_batch > 0:
            ema_path = getattr(cfg, "ema_resume_path", None)
            state_path = ema_path or get_checkpoint_state_path(
                cfg.log_path, start_batch, name="training_state"
            )
            restore_checkpoint_state(state_path)
            restored = get_checkpoint_state("role_baselines")
            if restored is not None:
                SpiralTwoPlayerEnvGroupBuilder._role_baseline_emas = restored
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    tokenizer = training_client.get_tokenizer()
    dataset, _ = await cfg.dataset_builder()
    evaluators = [eb() for eb in cfg.evaluator_builders]

    num_batches = len(dataset)
    logger.info(f"Training for {num_batches} batches (sync mode, no FSP)")

    await do_sync_training_spiral(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        eval_runner=eval_runner,
        tokenizer=tokenizer,
    )

    ml_logger.close()
    if eval_logger:
        eval_logger.close()
    logger.info("Training completed successfully")
