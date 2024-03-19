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
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: List[WindowEvent] = [
            # WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.button_states = [0 for _ in range(len(valid_actions))]

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
        return self.game_area().flatten().tolist() + self.button_states

    def _run_action_on_emulator(self, action: List[float]) -> None:
        # Toggles the buttons being on or off
        for i, toggle in enumerate(action):
            if toggle >= 0.5:
                self.button_states[i] = 1
                self.pyboy.send_input(self.valid_actions[i])
            else:
                self.button_states[i] = 0
                self.pyboy.send_input(self.release_button[i])

        for i in range(self.act_freq):
            self.pyboy.tick()

    def _generate_game_stats(self) -> Dict[str, int]:
        return {
            "lives": self._get_lives(),
            "score": self._get_score(),
            "powerup": self._get_powerup(),
            "coins": self._get_coins(),
            "stage": self._get_stage(),
            "world": self._get_world(),
            "game_over": self._get_game_over(),
            "x_position": self._get_x_position(),
            "time": self._get_time(),
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
            "lives_reward": self._lives_reward(new_state),
            "time_reward": self._time_reward(new_state),
        }

    def _position_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["x_position"] - self.prior_game_stats["x_position"]

    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["lives"] - self.prior_game_stats["lives"]

    def _time_reward(self, new_state: Dict[str, int]) -> int:
        return min(0, (new_state["time"] - self.prior_game_stats["time"]) * 10)

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
        # DA00       3    Timer (frames, seconds (Binary-coded decimal),
        # hundreds of seconds (Binary-coded decimal)) (frames count down from 0x28 to 0x01 in a loop)
        # return self._read_m(0xDA00)
        hundreds = self._read_m(0x9831)
        tens = self._read_m(0x9832)
        ones = self._read_m(0x9833)
        return int(str(hundreds) + str(tens) + str(ones))

    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]

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
        # Resetting game so that the agent doesn't need to use start button to start game
        if self._read_m(0xFFB3) == 0x3A:
            self.reset()
            return 1
        return 0

    def _get_powerup(self):
        # 0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 0x03 = shrinking, 0x04 = invincibility blinking
        # FFB5 (Does Mario have the Superball (0x00 = no, 0x02 = yes)
        # 3 = invincible (starman?), 2 = superball, 1 = big, 0 = small
        if self._read_m(0xFF99) != 0x04:
            if self._read_m(0xFFB5) != 0x02:
                if self._read_m(0xFF99) != 0x02:
                    return 0
                return 1
            return 2
        return 3

    def _get_airbourne(self):
        return self._read_m(0xC20A)

    def game_area(self) -> np.ndarray:
        mario = self.pyboy.game_wrapper
        mario.game_area_mapping(mario.mapping_compressed, 0)
        return mario.game_area()
