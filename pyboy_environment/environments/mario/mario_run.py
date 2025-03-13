import logging
from functools import cached_property
from typing import Dict, List

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.mario.mario_environment import MarioEnvironment


class MarioRun(MarioEnvironment):
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
            act_freq=act_freq,
            valid_actions=valid_actions,
            release_button=release_button,
            emulation_speed=emulation_speed,
            headless=headless,
        )

        self.max_level_progress = 0

    def reset(self) -> np.ndarray:
        self.max_level_progress = 0
        return super().reset()

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

    def _run_action_on_emulator(self, action: List[float]) -> None:
        # Toggles the buttons being on or off
        for i, toggle in enumerate(action):
            if toggle >= 0.5:
                self.pyboy.send_input(self.valid_actions[i])
            else:
                self.pyboy.send_input(self.release_button[i])

        for i in range(self.act_freq):
            self.pyboy.tick()

    def _calculate_reward(self, new_state: Dict[str, int]) -> float:
        reward_stats = {
            "position_reward": self._position_reward(new_state),
            "lives_reward": self._lives_reward(new_state),
            "score_reward": self._score_reward(new_state),
        }

        reward_total: int = 0
        for name, reward in reward_stats.items():
            logging.debug(f"{name} reward: {reward}")
            reward_total += reward
        return reward_total

    def _position_reward(self, new_state: Dict[str, int]) -> int:
        delta_distance = new_state["x_position"] - self.max_level_progress

        if new_state["x_position"] > self.max_level_progress:
            self.max_level_progress = new_state["x_position"]

        return max(0, delta_distance)

    def _score_reward(self, new_state: Dict[str, int]) -> int:
        return new_state["score"] - self.prior_game_stats["score"]

    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        return 5 * (new_state["lives"] - self.prior_game_stats["lives"])

    def _time_reward(self, new_state: Dict[str, int]) -> int:
        time_reward = min(0, (new_state["time"] - self.prior_game_stats["time"]) * 10)
        return max(time_reward, -10)

    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]

    def _check_if_truncated(self, game_stats):
        # Truncated if mario dies or if done more than a 1000 steps/actions
        return self.steps >= 1000 or game_stats["game_over"]
