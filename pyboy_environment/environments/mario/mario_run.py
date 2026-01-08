import logging
from typing import Dict

import numpy as np

from pyboy_environment.environments.mario.mario_environment import MarioEnvironment


class MarioRun(MarioEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        image_observation: bool = False,
    ) -> None:
        self.max_level_progress = 0     

        custom_actions = [
            ["down"],
            ["left"],
            ["right"],
            ["a"],
            ["b"],
            ["right", "a"],
            ["left", "a"],
            ["right", "b"],
            ["left", "b"],
        ]

        super().__init__(
            act_freq=act_freq,
            action_space=custom_actions,
            image_observation=image_observation,
            emulation_speed=emulation_speed,
            headless=headless,
        )


    def reset(self, training: bool = False) -> np.ndarray:
        state = super().reset()
        self.max_level_progress = self.prior_game_stats["x_position"]
        return state
    

    def _calculate_reward(self, new_state: Dict[str, int]) -> float:
        in_menu = new_state["world"] == 44
        if in_menu:
            return -1.0

        reward_stats = {
            "position_reward": self._position_reward(new_state),
            "lives_reward": self._lives_reward(new_state),
            "score_reward": self._score_reward(new_state),
        }

        reward_total: int = -0.01
        for name, reward in reward_stats.items():
            logging.debug(f"{name} reward: {reward}")
            reward_total += reward

        tanh_reward = np.tanh(reward_total)

        return tanh_reward


    def _position_reward(self, new_state: Dict[str, int]) -> int:
        delta_distance = new_state["x_position"] - self.max_level_progress

        if delta_distance > 0:
            self.max_level_progress = new_state["x_position"]
            return delta_distance * 0.02

        return 0


    def _score_reward(self, new_state: Dict[str, int]) -> int:
        delta_score = new_state["score"] - self.prior_game_stats["score"]
        if not delta_score:
            return 0
        # Typical score reward is 100 e.g. jumping on enemies or collecting coins
        return delta_score * 0.1


    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        delta_lives = new_state["lives"] - self.prior_game_stats["lives"]
        if not delta_lives:
            return 0
        else:
            return max(0, delta_lives) * 10


    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]


    def _check_if_truncated(self, game_stats):
        # Truncated if mario dies or if done more than a 4000 steps/actions
        return self.steps >= 4000 or game_stats["game_over"]
