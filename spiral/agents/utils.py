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

import random
import re


def kuhn_poker_parse_available_actions(observation: str):
    last_line = observation.strip().split("\n")[-1]
    available_actions = re.findall(r"\[(.*?)\]", last_line)

    # Add brackets
    available_actions = [f"[{action}]" for action in available_actions]
    # Remove [GAME]
    available_actions = [action for action in available_actions if action != "[GAME]"]
    return available_actions


def tic_tac_toe_parse_available_moves(observation: str):
    # Find the section after "Available Moves:" and before "Next Action:"
    moves_section_pattern = r"Available Moves:(.*?)Next Action:"
    moves_section = (
        re.search(moves_section_pattern, observation, re.DOTALL).group(1).strip()
    )

    # Now extract the moves from this section
    pattern = r"'\[(\d+)\]'"
    available_moves = re.findall(pattern, moves_section)

    available_moves = [f"[{move}]" for move in available_moves]

    return available_moves


def simple_negotiation_parse_available_actions(observation: str):
    valid_actions = []

    our_player_pattern = r"You are Player (\d+)"
    our_player_match = re.search(our_player_pattern, observation)
    if not our_player_match:
        return ["I'll think about my strategy."]

    our_player_id = int(our_player_match.group(1))

    resources_pattern = r"\[(\w+)\]\s+Qty:\s+(\d+)"
    resources = {}

    for match in re.finditer(resources_pattern, observation):
        resource_name = match.group(1)
        quantity = int(match.group(2))
        resources[resource_name] = quantity

    offer_to_us_pattern = r"Player (\d+) made the following offer to Player (\d+):"
    offer_matches = list(re.finditer(offer_to_us_pattern, observation))

    if offer_matches:
        last_offer = offer_matches[-1]
        from_player = int(last_offer.group(1))
        to_player = int(last_offer.group(2))

        if to_player == our_player_id:
            offer_position = last_offer.end()
            remaining_text = observation[offer_position:]

            if not re.search(r"(accepted|denied|implicitly denied)", remaining_text):
                valid_actions.extend(
                    [
                        "[Accept]",
                        "[Deny]",
                        "That sounds good to me. [Accept]",
                        "I'll pass on this offer. [Deny]",
                    ]
                )

    if resources and len(resources) >= 2:
        resource_names = list(resources.keys())

        for _ in range(3):
            offer_resource = random.choice(resource_names)
            request_resource = random.choice(
                [r for r in resource_names if r != offer_resource]
            )

            max_offer = min(3, resources[offer_resource])
            if max_offer > 0:
                offer_qty = random.randint(1, max_offer)
                request_qty = random.randint(1, 3)

                offer_str = f"[Offer: {offer_qty} {offer_resource} -> {request_qty} {request_resource}]"
                valid_actions.append(offer_str)

    chat_actions = [
        "Let me think about what would be a fair trade.",
        "What are you looking to trade?",
        "I'm open to negotiation.",
    ]
    valid_actions.extend(chat_actions)

    valid_actions = list(dict.fromkeys(valid_actions))

    return valid_actions if valid_actions else ["I'll think about my options."]


_VALID_ACTION_PARSER = {
    "TicTacToe-v0": tic_tac_toe_parse_available_moves,
    "KuhnPoker-v1": kuhn_poker_parse_available_actions,
    "SimpleNegotiation-v1": simple_negotiation_parse_available_actions,
}


def get_valid_action_parser(env_id: str):
    try:
        return _VALID_ACTION_PARSER[env_id]
    except KeyError:
        raise NotImplementedError(f"valid action parser not implemented for {env_id}")
