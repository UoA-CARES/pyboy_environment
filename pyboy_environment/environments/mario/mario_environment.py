"""
The link below has all the ROM memory data for Super Mario Land. 
It is used to extract the game state for the MarioEnvironment class.

https://datacrystal.tcrf.net/wiki/Super_Mario_Land/RAM_map
"""

import logging
from functools import cached_property
from typing import Dict, List

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.environment import PyboyEnvironment
from pyboy_environment.environments.mario import mario_constants as mc


class MarioEnvironment(PyboyEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: List[WindowEvent] = [
            # WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            # WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: List[WindowEvent] = [
            # WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            # WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        super().__init__(
            task="mario",
            rom_name="SuperMarioLand.gb",
            init_name="init.state",
            act_freq=act_freq,
            valid_actions=valid_actions,
            release_button=release_button,
            emulation_speed=emulation_speed,
            headless=headless,
        )

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def observation_space(self) -> int:
        return len(self._get_state())

    @cached_property
    def action_num(self) -> int:
        return len(self.valid_actions)

    def sample_action(self) -> np.ndarray:
        action = []
        for _ in range(self.action_num):
            action.append(np.random.rand())
        return action

    def _get_state(self) -> np.ndarray:
        return self.game_area().flatten().tolist()

    def _run_action_on_emulator(self, action: List[float]) -> None:
        # Toggles the buttons being on or off
        for i, toggle in enumerate(action):
            if toggle >= 0.5:
                self.pyboy.send_input(self.valid_actions[i])
            else:
                self.pyboy.send_input(self.release_button[i])

        for i in range(self.act_freq):
            self.pyboy.tick()

    def _generate_game_stats(self) -> Dict[str, int]:
        return {
            "lives": self._get_lives(),
            "score": self._get_score(),
            "coins": self._get_coins(),
            "stage": self._get_stage(),
            "world": self._get_world(),
            "x_position": self._get_x_position(),
            "time": self._get_time(),
            "dead_timer": self._get_dead_jump_timer(),
            "game_over": self._get_game_over(),
        }

    def _reward_stats_to_reward(self, reward_stats: Dict[str, int]) -> int:
        reward_total: int = 0
        for name, reward in reward_stats.items():
            logging.debug(f"{name} reward: {reward}")
            reward_total += reward
        return reward_total

    def _calculate_reward_stats(self, new_state: Dict[str, int]) -> Dict[str, int]:
        return {
            "position_reward": self._position_reward(new_state),
            "dead_reward": self._dead_reward(new_state),
            "score_reward": self._score_reward(new_state),
        }

    def _position_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["x_position"] - self.prior_game_stats["x_position"]

    def _dead_reward(self, new_state: Dict[str, int]) -> int:
        return -10 if new_state["dead_timer"] > 0 else 0

    def _score_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["score"] - self.prior_game_stats["score"]

    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["lives"] - self.prior_game_stats["lives"]

    def _time_reward(self, new_state: Dict[str, int]) -> int:
        time_reward = min(0, (new_state["time"] - self.prior_game_stats["time"]) * 10)
        return max(time_reward, -10)

    def _get_x_position(self):
        # Copied from: https://github.com/lixado/PyBoy-RL/blob/main/AISettings/MarioAISettings.py
        # Do not understand how this works...
        level_block = self._read_m(0xC0AB)
        mario_x = self._read_m(0xC202)
        scx = self.pyboy.screen.tilemap_position_list[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
        real_x_position = level_block * 16 + real + mario_x
        return real_x_position

    def _get_time(self):
        hundreds = self._read_m(0x9831)
        tens = self._read_m(0x9832)
        ones = self._read_m(0x9833)
        return int(str(hundreds) + str(tens) + str(ones))

    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]

    def _check_if_truncated(self, game_stats):
        # Setting truncated to true if mario loses a life or N steps have been exceded
        return self._get_dead_jump_timer() > 0 or self.steps > 1000

    def _get_lives(self):
        return self._read_m(0xDA15)

    def _get_score(self):
        return self._bit_count(self._read_m(0xC0A0))

    def _get_coins(self):
        return self._read_m(0xFFFA)

    def _get_stage(self):
        return self._read_m(0x982E)

    def _get_world(self):
        return self._read_m(0x982C)

    def _get_game_over(self):
        return self._read_m(0xFFB3) == 0x39

    def _get_mario_pose(self):
        return self._read_m(0xC203)

    def _get_dead_jump_timer(self):
        return self._read_m(0xC0AC)

    def game_area(self) -> np.ndarray:
        mario = self.pyboy.game_wrapper
        mario.game_area_mapping(mario.mapping_compressed, 0)
        return mario.game_area()
