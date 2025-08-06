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

"""Self-Play Reinforcement Learning Pipeline using OAT and TextArena."""

import copy
import functools
import json
import logging
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import textarena as ta
import torch.distributed as dist
import vllm
from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TrajectoryData
from oat.utils.data import load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import DataLoader
from tqdm import tqdm

from spiral.agents.random import RandomAgent
from spiral.agents.utils import get_valid_action_parser
from spiral.components import DummyPromptDataset, MATHOracle, SelfPlayCollector
from spiral.envs import make_env, make_vec_env
from spiral.metrics import EvaluationMetrics
from spiral.template import TEMPLATE_FACTORY
from spiral.utils import EMA, GameState, extract_boxed_answer

logging.basicConfig(level=logging.DEBUG)


INVALID_ACTION = "[｜INVALID_ACTION｜]"


"""
1. Define extra arguments needed besides Oat's PPOArgs, mainly about self-play configurations.
"""


@dataclass
class SelfPlayArgs(PPOArgs):
    # Environment settings
    env_ids: List[str] = field(default_factory=lambda: ["KuhnPoker-v1"])
    use_llm_obs_wrappers: List[bool] = field(
        default_factory=lambda: [True]
    )  # Encode opponent history in the obs

    # Self-play specific settings
    num_envs: int = 1
    fixed_opponent: Literal[
        "", "random", "google/gemini-2.0-flash-lite-001", "google/gemini-2.0-flash-001"
    ] = ""
    filter_zero_adv: bool = (
        True  # Make gradient less noisy by filtering zero-gradient trajectories
    )
    use_role_baseline: bool = True  # Use role baseline for reward shaping
    role_baseline_ema_gamma: float = 0.95

    # Game evaluation
    eval_games: int = 16  # Number of games for evaluation
    eval_env_ids: List[str] = field(
        default_factory=lambda: ["TicTacToe-v0", "KuhnPoker-v1", "SimpleNegotiation-v1"]
    )
    eval_use_llm_obs_wrappers: List[bool] = field(default_factory=lambda: [False, True, True])
    eval_opponent_names: List[str] = field(
        default_factory=lambda: ["random", "google/gemini-2.0-flash-lite-001"]
    )
    eval_prompt_template: Literal["qwen3_general", "r1_general"] = "qwen3_general"

    # Dump all game data.
    dump_game_state_every: int = 1

    # Template settings
    prompt_template: Literal["qwen3", "r1"] = "qwen3"
    # Optional override for specific environments
    prompt_template_overrides: str = ""  # Format: "env1:template1,env2:template2"

    # Reward settings
    reward_scaling: float = 1.0  # Scale factor for rewards
    gamma: float = 1.0  # Discount factor for Monte Carlo returns

    # Game settings
    max_context_length: int = 32768  # Maximum context length for game history
    max_turns: int = 50  # Maximum turns before truncating a game
    use_intermediate_rewards: bool = True  # Whether to use intermediate rewards

    # Math reasoning evaluation
    eval_data: Optional[str] = "./data"
    eval_input_key: str = "input"
    eval_output_key: str = "answer"
    eval_split: str = "all"


"""
2. Instantiate the actor based on Oat's PPOActor, which generates the self-play experiences.
"""


class SelfPlayActor(PPOActor):
    """Actor class for self-play reinforcement learning."""

    def _parse_template_overrides(self, override_str: str) -> Dict[str, str]:
        """Parse template overrides from string format 'env1:template1,env2:template2'."""
        if not override_str:
            return {}

        overrides = {}
        for pair in override_str.split(","):
            if ":" in pair:
                env, template = pair.split(":")
                overrides[env.strip()] = template.strip()
        return overrides

    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        if actor_id == 0:
            os.makedirs(self.game_state_save_path, exist_ok=True)
        self.args: SelfPlayArgs = self.args
        args = self.args
        self.oracle = MATHOracle(
            args.eval_prompt_template, "fast", correct_reward=1, incorrect_reward=0
        )

        # Set up sampling parameters (copied from PPOActor)
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=1,  # Override to only generate 1 response per prompt for self-play
            logprobs=True,
        )

        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,  # Override to only generate 1 response per prompt for self-play
            logprobs=True,
        )

        self.step_count = 0
        self.online_model_player = actor_id % 2
        if self.args.fixed_opponent not in ["", "random"]:
            self.open_router_opponent = ta.agents.OpenRouterAgent(
                self.args.fixed_opponent
            )
        if self.args.use_role_baseline:
            self.role_baseline_ema = {}
            for env_id in self.args.env_ids:
                self.role_baseline_ema[env_id] = {
                    0: EMA(self.args.role_baseline_ema_gamma),
                    1: EMA(self.args.role_baseline_ema_gamma),
                }
            logging.info("Using role baseline for reward shaping")

        # Parse overrides once during initialization
        self._template_overrides = self._parse_template_overrides(
            self.args.prompt_template_overrides
        )

    def step(
        self, prompts=None, formatted_prompts=None, references=None
    ) -> List[TrajectoryData]:
        """
        Override step method to play full games rather than single-turn inference.

        Returns:
            serialized trajectories data
        """
        # The provided parameters are ignored since we generate prompts from the environment
        del prompts, formatted_prompts, references

        logging.info(
            f"Actor-{self.actor_id} starting to collect game trajectories at step {self.step_count}"
        )
        info = {}

        # Play multiple games to generate trajectory data
        all_trajectories = []
        st = time.time()

        for i in range(int(1e9)):
            # Shuffle environments to mitigate order bias
            env_ids = copy.deepcopy(self.args.env_ids)
            random.shuffle(env_ids)
            for env_id in env_ids:
                game_trajectories = self.play_game_vectorized(
                    env_id=env_id, seed=int(time.time_ns())
                )
                all_trajectories.extend(game_trajectories)

            if len(all_trajectories) >= self.args.rollout_batch_size_per_device:
                subsample_indices = np.random.choice(
                    len(all_trajectories),
                    self.args.rollout_batch_size_per_device,
                    replace=False,
                )
                all_trajectories = [all_trajectories[si] for si in subsample_indices]
                break

        info["actor/game_time"] = time.time() - st
        info["actor/num_games"] = i + 1
        info["actor/num_trajectories"] = len(all_trajectories)

        # Log rewards statistics
        rewards = [np.max(t.rewards) for t in all_trajectories]
        info["actor/mean_reward"] = np.mean(rewards)
        info["actor/max_reward"] = np.max(rewards)
        info["actor/min_reward"] = np.min(rewards)

        logging.info(f"Actor finished collecting {len(all_trajectories)} trajectories")

        self.step_count += 1
        # Serialize and return the trajectories
        handle = self.ipc_client.serialize_ipc(all_trajectories)
        return handle

    def play_game_vectorized(
        self,
        env_id: str,
        seed: Optional[int] = None,
    ) -> List[TrajectoryData]:
        # Create and initialize vectorized environments
        vec_envs = make_vec_env(
            env_id,
            self.args.num_envs,
            use_llm_obs_wrapper=self.args.env_to_llm_obs_wrapper[env_id],
        )

        for i, env in enumerate(vec_envs):
            env.reset(num_players=2, seed=seed + i)
            env.state.error_allowance = 0

        # Initialize game state
        vec_game_states = [
            GameState(
                max_context_length=self.args.max_context_length,
                max_turns=self.args.max_turns,
            )
            for _ in range(self.args.num_envs)
        ]
        vec_done = [False] * self.args.num_envs
        vec_rewards = [None] * self.args.num_envs

        # Main game loop
        while not all(vec_done):
            # Get current player and observation
            vec_player_id = []
            vec_observation = []
            for i in range(self.args.num_envs):
                if not vec_done[i]:
                    env = vec_envs[i]
                    player_id, observation = env.get_observation()
                    vec_player_id.append(player_id)
                    vec_observation.append(observation)
                else:
                    vec_player_id.append(None)
                    vec_observation.append(None)

            _mean_pid = np.mean([x for x in vec_player_id if x is not None])
            assert _mean_pid == 0 or _mean_pid == 1, "vec_env player_id not consistent"
            _curr_pid = vec_player_id[0]

            # --- [BEGIN] Fixed Opponent Logic Init ---
            agent_act = self.agent_act
            _fixed_opponent = ""
            if self.args.fixed_opponent and _curr_pid == 1 - self.online_model_player:
                logging.info(
                    f"player{_curr_pid} using fixed opponent={self.args.fixed_opponent}"
                )
                _fixed_opponent = self.args.fixed_opponent
                agent_act = partial(
                    self.fixed_opponent_act, opponent_type=_fixed_opponent
                )
            # --- [END] Fixed Opponent Logic Init ---

            vec_action, vec_extras = agent_act(vec_observation, env_id=env_id)

            for i in range(self.args.num_envs):
                if not vec_done[i]:
                    game_state = vec_game_states[i]
                    player_id = vec_player_id[i]
                    observation = vec_observation[i]
                    action = vec_action[i]
                    extras = vec_extras[i]

                    # Store trajectory data
                    game_state.add_trajectory_data(
                        player_id,
                        {
                            "prompt": observation,
                            "action": action,
                            "action_is_valid": action != INVALID_ACTION,
                            "player_id": (
                                player_id if not _fixed_opponent else _fixed_opponent
                            ),
                            "turn": game_state.turn_count,
                            **extras,
                        },
                    )

                    # Add to game history
                    _thinking = extras["response"]
                    _thinking += (
                        "...(truncated)" if extras["response_is_truncated"] else ""
                    )
                    game_state.add_interaction(
                        player_id, observation, action, _thinking
                    )

            # Take step in environment
            for i in range(self.args.num_envs):
                if not vec_done[i]:
                    env = vec_envs[i]
                    action = vec_action[i]
                    player_id = vec_player_id[i]
                    done, _ = env.step(action=action)
                    if action == INVALID_ACTION:
                        done = True
                    vec_done[i] = done
                    if done and action == INVALID_ACTION:
                        rewards = {0: 0.5, 1: 0.5}
                        rewards[player_id] = -1.5
                        vec_rewards[i] = rewards

            # Check if game should be truncated
            for i in range(self.args.num_envs):
                if not vec_done[i]:
                    game_state = vec_game_states[i]
                    if game_state.is_truncated():
                        logging.warning(
                            f"Game truncated after {game_state.turn_count} turns"
                        )
                        # Set draw rewards
                        rewards = {0: 0, 1: 0}
                        vec_done[i] = True
                        vec_rewards[i] = rewards

        for i in range(self.args.num_envs):
            if vec_rewards[i] is None:
                assert vec_done[i]
                vec_rewards[i] = vec_envs[i].close()
        # Dump the game state for debugging.
        if (
            self.args.dump_game_state_every > 0
            and self.step_count % self.args.dump_game_state_every == 0
        ):
            pickle.dump(
                {
                    "vec_game_states": vec_game_states,
                    "vec_rewards": vec_rewards,
                },
                open(
                    os.path.join(
                        self.game_state_save_path,
                        f"actor{self.actor_id}_step{self.step_count}.pkl",
                    ),
                    "wb",
                ),
            )
            vec_history = [gs.long_history for gs in vec_game_states]

            json.dump(
                [{"reward": r, "history": h} for r, h in zip(vec_rewards, vec_history)],
                open(
                    os.path.join(
                        self.game_state_save_path,
                        f"actor{self.actor_id}_step{self.step_count}.json",
                    ),
                    "w",
                ),
                indent=4,
            )

        trajectories = []
        for game_state, rewards in zip(vec_game_states, vec_rewards):
            trajectories.extend(self.prepare_trajectories(game_state, rewards, env_id))

        return trajectories

    def fixed_opponent_act(
        self, vec_observation: List[str], env_id: str, opponent_type: str = "random"
    ) -> Tuple[str, dict]:
        clean_actions = []
        extras = []
        for observation in vec_observation:
            if observation is None:
                clean_actions.append(None)
                extras.append(None)
                continue

            if opponent_type == "random":
                clean_action = RandomAgent(env_id)(observation)
            else:
                # Not clean_action, but env will parse the last [x].
                clean_action = self.open_router_opponent(observation)

            clean_actions.append(clean_action)
            extras.append(
                {
                    "formatted_observation": "",
                    "prompt_ids": [],
                    "response": f"This action is taken by a fixed agent: {opponent_type}",
                    "response_ids": [],
                    "response_is_truncated": True,
                }
            )
        return clean_actions, extras

    def agent_act(self, vec_observation: List[str], env_id: str) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.

        """
        clean_actions = []
        extras = []
        for observation in vec_observation:
            if observation is None:
                clean_actions.append(None)
                extras.append(None)
                continue

            # Get template for this specific environment
            template_name = self._template_overrides.get(
                env_id, self.args.prompt_template
            )

            formatted_observation = TEMPLATE_FACTORY[template_name](
                observation, system_prompt=None
            )
            sampling_params = (
                self.eval_sampling_params if self.eval_mode else self.sampling_params
            )
            outputs = self.generate([formatted_observation], sampling_params)
            raw_action = outputs[0].outputs[0].text
            prompt_token_ids = outputs[0].prompt_token_ids
            token_ids = outputs[0].outputs[0].token_ids

            if env_id in ["DontSayIt-v0", "SimpleNegotiation-v1"]:  # DontSayIt-v0 don't have fixed action space
                clean_action = self.extract_chat_action(raw_action)
            else:
                action_space = get_valid_action_parser(env_id)(observation)
                clean_action = self.extract_action(raw_action, action_space)
            response_is_truncated = outputs[0].outputs[0].finish_reason == "length"

            clean_actions.append(clean_action)
            extras.append(
                {
                    "formatted_observation": formatted_observation,
                    "prompt_ids": prompt_token_ids,
                    "response": raw_action,
                    "response_ids": token_ids,
                    "response_is_truncated": response_is_truncated,
                }
            )
        return clean_actions, extras

    def extract_chat_action(self, text: str) -> str:
        answer_match = extract_boxed_answer(text)

        if answer_match is not None:
            # Found boxed content
            raw_action = answer_match.strip()
            if raw_action.strip("\n ") == "":
                return INVALID_ACTION
            return raw_action
        # If no boxed content, try to find <answer> tags
        else:
            return INVALID_ACTION

    def prepare_trajectories(
        self, game_state: GameState, rewards: Dict[int, float], env_id: str
    ) -> List[TrajectoryData]:
        """
        Prepare language trajectories created in the game.

        Args:
            game_state: Game state with trajectory data
            rewards: Final rewards for each player

        Returns:
            List of trajectory data
        """
        trajectory_data = []

        player_ids_for_training = [0, 1]
        if self.args.fixed_opponent:
            player_ids_for_training = [self.online_model_player]
        logging.info(f"player_ids_for_training: {player_ids_for_training}")

        for player_id in player_ids_for_training:
            player_trajectories = game_state.get_player_trajectories(player_id)
            player_reward = rewards[player_id] * self.args.reward_scaling

            if self.args.use_role_baseline:
                # Get the baseline before updating to be unbiased
                baseline = self.role_baseline_ema[env_id][player_id].get()
                # Update role-baseline ema
                self.role_baseline_ema[env_id][player_id].update(player_reward)
                player_reward -= baseline

            # Compute returns for each action (turn) for this player
            for i, step_data in enumerate(player_trajectories):
                # For intermediate rewards, we can decay based on steps from end
                if self.args.use_intermediate_rewards:
                    # Earlier moves get more discounted rewards
                    steps_from_end = len(player_trajectories) - i - 1
                    discounted_reward = player_reward * (
                        self.args.gamma**steps_from_end
                    )
                else:
                    # Only final outcome matters
                    discounted_reward = player_reward

                # Distribute turn-based reward to token-level reward
                dense_rewards = self.compute_token_level_rewards(
                    step_data["response_ids"], discounted_reward
                )

                if self.args.filter_zero_adv and discounted_reward == 0:
                    continue

                # Add trajectory data
                trajectory_data.append(
                    TrajectoryData(
                        prompt=step_data["prompt"],
                        prompt_ids=step_data["prompt_ids"],
                        response=step_data["response"],
                        response_ids=step_data["response_ids"],
                        response_logprobs=None,  # Re-calculated on learner side.
                        # response_logprobs=step_data["response_logprobs"],
                        rewards=dense_rewards,
                        loss_mask=(
                            not step_data["response_is_truncated"]
                            if self.args.ignore_no_eos
                            else True
                        ),
                        info={
                            "actor/player_id": player_id,
                            "actor/current_turn": step_data["turn"],
                            "actor/game_length": game_state.turn_count,
                            "actor/action_is_valid": step_data["action_is_valid"],
                            "actor/final_reward": player_reward,
                            "actor/discount_factor": self.args.gamma,
                            "actor/discounted_turn_reward": discounted_reward,
                            "actor/response_is_truncated": step_data[
                                "response_is_truncated"
                            ],
                            "actor/draw": rewards[0] == rewards[1] == 0,
                        },
                    )
                )

        return trajectory_data

    def compute_token_level_rewards(
        self, token_ids: List[int], discounted_reward: float
    ) -> List[float]:
        # Initialize all tokens with zero reward
        dense_rewards = [0.0] * len(token_ids)
        # Last token gets full discounted reward
        dense_rewards[-1] = discounted_reward
        return dense_rewards

    def extract_action(self, text: str, action_space: list) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            # First extract the raw action based on template format
            raw_action = ""

            if self.args.prompt_template == "r1":
                # Extract content from <answer> tags
                answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

                if answer_match:
                    # Found answer tags
                    raw_action = answer_match.group(1).strip()
                else:
                    # Fallback: try to find content after </think> if no answer tags
                    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                    if think_match:
                        # Get position after </think> tag
                        think_end_pos = text.find("</think>") + len("</think>")
                        # Extract everything after the closing think tag
                        raw_action = text[think_end_pos:].strip()
                    else:
                        # No tags found, use the whole text
                        raw_action = text.strip()

            elif self.args.prompt_template == "deepseek_r1_distill_qwen":
                # Extract content from \boxed{} notation
                boxed_match = re.search(r"\\boxed\{([^}]*)\}", text, re.DOTALL)

                if boxed_match:
                    # Found boxed content
                    raw_action = boxed_match.group(1).strip()
                else:
                    # Fallback: try to find content after </think> tag
                    think_match = re.search(r"</think>(.*)", text, re.DOTALL)
                    if think_match:
                        raw_action = think_match.group(1).strip()
                    else:
                        # No tags found, use the whole text
                        raw_action = text.strip()

            elif self.args.prompt_template in ["qwen", "qwen3"]:
                raw_action = extract_boxed_answer(text)
                if raw_action is None:
                    raw_action = text.strip()
                    
            elif self.args.prompt_template in ["octothinker", "octothinker_enforce_thinking"]:
                # OctoThinker templates use \boxed{} format for actions
                raw_action = extract_boxed_answer(text)
                if raw_action is None:
                    # Fallback: if enforce_thinking, try to get content after </think>
                    if "octothinker_enforce_thinking" in self.args.prompt_template:
                        think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
                        if think_match:
                            raw_action = think_match.group(1).strip()
                        else:
                            raw_action = text.strip()
                    else:
                        raw_action = text.strip()

            else:
                raise NotImplementedError

            # Now apply any necessary formatting to make the action valid for the environment

            # 1. Convert \boxed{} format to [content] format if found in the action
            formatted_action = re.sub(
                r"\\boxed\{([^}]*)\}",  # Match \boxed{...} capturing everything up to the matching }
                r"[\1]",  # Replace with brackets around the captured content
                raw_action,
            )

            # 2. If there are no brackets but we should have them, add them
            if "[" not in formatted_action and "]" not in formatted_action:
                # Check if this is a short action that likely needs brackets
                words = formatted_action.split()
                if (
                    len(words) <= 5
                ):  # Heuristic for a short action that might need brackets
                    formatted_action = f"[{formatted_action}]"

            # 3. Additional cleaning to ensure valid formatting
            # Remove any extra newlines, tabs, or multiple spaces
            formatted_action = re.sub(r"\s+", " ", formatted_action).strip()

            # NOTE(zc): ad-hoc postprocessing, strictly enforcing action space.
            if formatted_action not in action_space:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION

    def run_eval_episode(self, env_id, opponent_name):
        player_id = self.online_model_player

        logging.info(
            f"Eval on {env_id} against {opponent_name} agent as player{player_id}"
        )

        assert self.eval_mode

        opponent_id = 1 - player_id
        agents = {
            player_id: lambda obs: self.agent_act([obs], env_id)[0][0],
            opponent_id: (
                RandomAgent(env_id)
                if opponent_name == "random"
                else ta.agents.OpenRouterAgent(opponent_name)
            ),
        }

        _use_llm_obs_wrapper = dict(
            zip(self.args.eval_env_ids, self.args.eval_use_llm_obs_wrappers)
        )[env_id]
        env = make_env(env_id, _use_llm_obs_wrapper)
        env.reset(num_players=2, seed=int(time.time_ns()))
        env.state.error_allowance = 0

        turn_counter = 0
        done = False
        invalid_rewards = None
        while not done:
            pid, observation = env.get_observation()
            action = agents[pid](observation)
            done, info = env.step(action)
            if action == INVALID_ACTION:
                done = True
            turn_counter += 1
            if done and action == INVALID_ACTION:
                invalid_rewards = {0: 1, 1: 1}
                invalid_rewards[pid] = -1
                rewards = {0: 1, 1: 1}
                rewards[pid] = -1
        if "rewards" not in locals():
            rewards = env.close()

        if invalid_rewards:
            invalid_move = (invalid_rewards[0] == 1 and invalid_rewards[1] == -1) or (
                invalid_rewards[0] == -1 and invalid_rewards[1] == -1
            )
        else:
            invalid_move = False

        if rewards[player_id] > rewards[opponent_id]:
            outcome = "win"
        elif rewards[player_id] < rewards[opponent_id]:
            outcome = "loss"
        else:
            outcome = "draw"

        metrics = {
            "outcome": outcome,
            "invalid_move": invalid_move,
            "reason": info.get("reason", ""),
            "num_turns": turn_counter,
            "opponent_reward": rewards[opponent_id],
            "model_reward": rewards[player_id],
            "env_id": env_id,
            "opponent_name": opponent_name,
            "model_pid": player_id,
        }

        return metrics


"""
3. Instantiate the learner based on PPOLearner. Here we adapt the `evaluate` logic to run online evaluation for both game and math.
"""


class SelfPlayLearner(PPOLearner):
    """Learner class for self-play reinforcement learning."""

    def _init(self, args: SelfPlayArgs, actors: List[ActorBase]) -> None:
        """
        Initialize the self-play learner.

        CRITICAL: We override this method to skip OAT's dataset loading mechanism.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Replace the standard collector with our self-play collector
        if actors:
            self.collector = SelfPlayCollector(args, actors, self.collector.ipc_client)

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

    def prepare_data(self, strategy, tokenizer):
        """
        Override the data preparation to avoid loading external datasets.
        Instead, create dummy datasets just to keep OAT's infrastructure happy.
        """
        # Create dummy dataset that satisfies OAT's requirements
        # but doesn't actually load any data
        # Used to control the training episode, set a large number.
        self.prompts_dataset = DummyPromptDataset(size=int(1e9))
        self.eval_prompts_dataset = DummyPromptDataset(size=self.args.eval_games)

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )

        self.eval_prompts_dataloader = DataLoader(
            self.eval_prompts_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # Load any other reasoning benchmark for online eval
        self.eval_dataset_dict = load_data_from_disk_or_hf(args.eval_data)
        if args.eval_split != "all":
            self.eval_dataset_dict = {
                k: v
                for k, v in self.eval_dataset_dict.items()
                if k in args.eval_split.split(",")
            }

        strategy.print("Using dummy dataset for self-play (no external data needed)")

    def eval_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.eval_prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def process_feedback_data(self, data_list: List[TrajectoryData]):
        """Process collected feedback data, adding it to buffer."""

        logging.info("adding data into buffer")

        # Add to buffer
        self.pi_buffer.extend(data_list)

        # Also add to all_buffer if we're tracking all data
        if self.args.dump_all_buffer:
            self.all_buffer.extend(data_list)

        # Update query step (for tracking progress)
        self.query_step += len(data_list)

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        # Return without baseline
        rewards = rewards.sum(-1)
        return rewards

    def evaluate(self, _unused_dataloader, steps):
        """
        Online evaluation with hierarchical metrics.

        We do three things here:
        1) Evaluation on games, either in-domain or out-domain, against various opponents (random, rule-based, LLMs);
        2) Evaluation on general reasoning tasks, including math, etc.
        """
        del _unused_dataloader
        assert not self.pi_beta_lags_behind, "pi beta lags behind for evaluation"
        self._pre_evaluate()

        self.strategy.print(f"Start evaluating on games at step {steps}")

        # 1) Game eval.
        t0 = time.time()
        # ------------------------------------------------------------------
        # Initialize metrics tracking
        # ------------------------------------------------------------------
        eval_env_ids = self.args.eval_env_ids
        eval_opponent_names = self.args.eval_opponent_names
        game_metrics = EvaluationMetrics(
            eval_env_ids, eval_opponent_names
        )  # Initialize metrics across all ranks

        # ------------------------------------------------------------------
        # Rank 0 distributes evaluation workloads to all ranks then collects and populates metrics
        # ------------------------------------------------------------------
        if self.strategy.is_rank_0():
            total_games = self.args.eval_games

            # Generate evaluation runs
            eval_runs_list = []
            for env_id in eval_env_ids:
                for opponent_name in eval_opponent_names:
                    if opponent_name == "random":
                        try:
                            RandomAgent(env_id)
                        except NotImplementedError:
                            logging.warning(
                                f"Random opponent is not supported for {env_id}, skipping"
                            )
                            continue

                    for game_nr in range(total_games):
                        eval_runs_list.append((env_id, opponent_name, game_nr))

            # Run evaluation
            futs = []
            progress_bar = tqdm(range(len(eval_runs_list)), desc="Evaluating")
            random.shuffle(eval_runs_list)

            for i, (env_id, opponent_name, game_nr) in enumerate(eval_runs_list):
                actor = self.actors[i % len(self.actors)]
                futs.append(actor.futures.run_eval_episode(env_id, opponent_name))

                # Process results in batches
                if len(futs) == len(self.actors) or i == len(eval_runs_list) - 1:
                    for fut in futs:
                        result = fut.result()
                        game_metrics.add_result(result)
                        progress_bar.update(1)

                    futs.clear()

            game_metrics.aggregate()

        dist.barrier()
        game_metrics_dict = game_metrics.to_dict()
        game_metrics_dict["eval/game_eval_time"] = time.time() - t0
        game_metrics_dict = self.strategy.broadcast(game_metrics_dict)

        self._post_evaluate()

        # 2) Single-turn verifiable reasoning eval.
        t0 = time.time()
        non_game_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            metrics = {
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "eval/accuracy",
                    "eval/score",
                    "eval/response_tok_len",
                    "eval/elapse",
                ]
            }
            non_game_metrics.update(
                {
                    k.replace("eval/", f"eval/general/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        non_game_metrics.update(
            {
                "eval/general/average/accuracy": np.mean(accuracies),
                "eval/general/average/score": np.mean(scores),
                "eval/general/average/response_tok_len": np.mean(lens),
            }
        )

        # ------------------------------------------------------------------
        # Synchronize metrics across all ranks
        # ------------------------------------------------------------------
        metrics_dict = {
            **game_metrics_dict,
            **non_game_metrics,
        }
        return metrics_dict


"""
4. Compose the distributed program.
"""


def run_self_play_rl(args: SelfPlayArgs):
    """
    Run the self-play reinforcement learning training pipeline.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(
        args, learner_cls=SelfPlayLearner, actor_cls=SelfPlayActor
    )

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


"""
5. Argument validation and entry point.
"""

if __name__ == "__main__":
    # Get default arguments and customize them
    args: SelfPlayArgs = get_default_args(SelfPlayArgs)

    # Customization
    args.algo = "PPO"
    args.eval_batch_size = 32

    # CRITICAL: Disable oracle and dataset loading
    args.oracle = ""  # Empty string for no external oracle
    args.prompt_data = None  # Don't load any dataset

    args = default_args_validation(args)

    # Validate that the number of environments matches the number of wrapper settings
    assert len(args.env_ids) == len(args.use_llm_obs_wrappers), \
        f"Number of env_ids ({len(args.env_ids)}) must match number of use_llm_obs_wrappers ({len(args.use_llm_obs_wrappers)})"
    
    # Create environment to wrapper mapping for quick access
    args.env_to_llm_obs_wrapper = dict(zip(args.env_ids, args.use_llm_obs_wrappers))
    
    # Validate environment-specific requirements
    for env_id in args.env_ids:
        if env_id == "KuhnPoker-v1":
            assert args.num_envs == 1, "Please set --num_envs 1 for KuhnPoker-v1"
            assert args.env_to_llm_obs_wrapper[env_id], \
                "Please set --use_llm_obs_wrappers True for KuhnPoker-v1"
        elif env_id == "TicTacToe-v0":
            assert not args.env_to_llm_obs_wrapper[env_id], \
                "Please set --use_llm_obs_wrappers False for TicTacToe-v0"
        elif env_id == "SimpleNegotiation-v1":
            assert args.env_to_llm_obs_wrapper[env_id], \
                "Please set --use_llm_obs_wrappers True for SimpleNegotiation-v1"
    
    assert len(args.eval_env_ids) == len(args.eval_use_llm_obs_wrappers)

    # Let's go
    run_self_play_rl(args)
