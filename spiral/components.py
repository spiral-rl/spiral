# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implement components defined by Oat, but not critical for our self-play framework."""

import functools
import logging
import time
from itertools import accumulate
from multiprocessing import Pool
from typing import Any, List, Tuple

import numpy as np
import torch
import tree
from oat.actors.base import ActorBase
from oat.collectors import FeedbackCollector
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TransitionData
from oat.utils.ipc import PlasmaShmClient
from oat.utils.math_grader import boxed_reward_fn, answer_tag_reward_fn
from torch.utils.data import Dataset


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(
        self, template, verifier_version, correct_reward, incorrect_reward
    ) -> None:
        super().__init__()
        if template in ["qwen3_general", "r1_general", "llama_instruct_general"]:
            math_reward_fn = boxed_reward_fn
        else:
            raise ValueError(f"Unsupported template: {template}")

        self.math_reward_fn = functools.partial(
            math_reward_fn,
            fast=verifier_version == "fast",
            correct_reward=correct_reward,
            incorrect_reward=incorrect_reward,
        )
        self.incorrect_reward = incorrect_reward
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        # Parameters used by Oat when using model-based reward, here we don't need.
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, r = res.get(timeout=1)
                rewards.append(r)
                infos.append(info)
            except TimeoutError:
                rewards.append(self.incorrect_reward)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


class SelfPlayCollector(FeedbackCollector):
    """Custom collector for self-play that handles game-based data."""

    def __init__(self, args, actors: List[ActorBase], ipc_client: PlasmaShmClient):
        self.args = args
        self.actors = actors
        self.ipc_client = ipc_client

    def collect_feedback(self, prompts, formatted_prompts, refs, same_actor_group):
        """
        Collect game-based feedback from actors.

        This method ignores the provided prompts and formatted_prompts,
        as the environment generates these during gameplay.

        Returns:
            Tuple of (feedback_data, metrics)
        """
        del prompts, formatted_prompts, refs

        st_time = time.time()

        # Select actor based on rank (in distributed setting)
        rank = torch.distributed.get_rank()
        actor = self.actors[(rank // self.args.num_gpus_per_actor) % len(self.actors)]
        logging.info(
            f"Learner {rank} local actor rank: {(rank // self.args.num_gpus_per_actor) % len(self.actors)}"
        )

        if torch.distributed.get_rank(same_actor_group) == 0:
            # Get trajectories from the actor
            handle = actor.step(
                ["dummy"]
                * (
                    self.args.rollout_batch_size_per_device
                    * self.args.num_gpus_per_actor
                )
            )  # No arguments needed as environment provides prompts
            feedback_data = self.ipc_client.deserialize_ipc(handle)
            logging.info(
                f"Group Leader Learner {rank} feedback_data size: {len(feedback_data)}"
            )
            rank_lengths = [
                self.args.rollout_batch_size_per_device
            ] * self.args.num_gpus_per_actor
            assert len(feedback_data) == sum(rank_lengths)
            feedback_data = [
                feedback_data[i:j]
                for i, j in zip(
                    [0] + list(accumulate(rank_lengths)), accumulate(rank_lengths)
                )
            ]
        else:
            feedback_data = None

        scattered_feedback_data = [None]
        torch.distributed.scatter_object_list(
            scattered_feedback_data, feedback_data, group=same_actor_group, group_src=0
        )
        scattered_feedback_data = scattered_feedback_data[0]
        logging.info(
            f"Learner {rank} scattered_feedback_data size: {len(scattered_feedback_data)}"
        )

        actor_time = time.time() - st_time
        return scattered_feedback_data, self._get_metrics(
            actor_time, scattered_feedback_data
        )

    def _get_metrics(self, actor_time: float, feedback_data: List[TransitionData]):
        """Extract and calculate metrics from the collected data."""
        metrics = {
            "actor/total_time": actor_time,
            "actor/num_trajectories": len(feedback_data),
        }

        if feedback_data:
            # Calculate statistics about generated responses
            metrics.update(
                {
                    "actor/generate_avg_str_len": np.mean(
                        [len(t.response) for t in feedback_data]
                    ),
                    "actor/avg_reward": np.mean(
                        [max(t.rewards) for t in feedback_data]
                    ),
                }
            )

            mean_info = tree.map_structure(
                lambda *x: np.mean(x), *[p.info for p in feedback_data]
            )
            metrics.update(mean_info)

        return metrics


# Dummy dataset for OAT's infrastructure
class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy OAT's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return "", "", ""
