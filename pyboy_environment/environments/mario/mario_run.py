import logging
from functools import cached_property
from typing import Dict, List

import numpy as np
from collections import deque
from pyboy.utils import WindowEvent

from pyboy_environment.environments.mario.mario_environment import MarioEnvironment


class MarioRun(MarioEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        image_observation: bool = False,
    ) -> None:

        valid_actions: List[List[WindowEvent]] = [
            [WindowEvent.PRESS_ARROW_DOWN],
            [WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_ARROW_RIGHT],
            # [WindowEvent.PRESS_ARROW_UP],
            [WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B]
        ]

        release_button: List[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            # WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.actions: List[List[str]] = [
            ["down"],
            ["left"],
            ["right"],
            # ["up"],
            ["a"],
            ["b"],
            ["right", "a"],
            ["left", "a"],
            ["right", "b"],
            ["left", "b"],
        ]

        self.release_button_offset = 8
        self.stack_frames = 3
        self.prev_frames = deque(maxlen=self.stack_frames)
        self.max_level_progress = 0
        self.prev_actions = deque(maxlen=3)
        self.count = 0

        if image_observation:
            self._get_state = self._get_state_image
        else:
            self._get_state = self._get_state_vector        

        super().__init__(
            act_freq=act_freq,
            valid_actions=valid_actions,
            release_button=release_button,
            emulation_speed=emulation_speed,
            headless=headless,
        )


    def _get_state_image(self) -> np.ndarray:
        frame = self.game_area()[np.newaxis, ...]
        self.prev_frames.append(frame)
        return np.concatenate(self.prev_frames)
    

    def _get_state_vector(self) -> Dict[str, int]:
        frame = self.game_area().flatten()
        self.prev_frames.append(frame)
        return np.concatenate(self.prev_frames)


    def reset(self, training: bool = False) -> np.ndarray:
        super().reset()
        self.prev_actions.clear()
        self.max_level_progress = self.prior_game_stats["x_position"]

        state = self.prev_frames[-1]
        while len(self.prev_frames) < self.stack_frames:
            self.prev_frames.append(state)
        state = np.concatenate(self.prev_frames)
        
        return state


    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def max_action_value(self) -> float:
        return len(self.valid_actions)

    @cached_property
    def observation_space(self) -> int | tuple[int]:
        shape = self._get_state().shape
        if len(shape) > 1:
            return shape
        else:
            return shape[0]

    @cached_property
    def action_num(self) -> int:
        return len(self.valid_actions)
    
    def sample_action(self) -> list[int]:
        length = len(self.valid_actions)
        random_index = np.random.randint(0, length)
        return np.array([random_index])
    
    def get_overlay_info(self) -> dict:
        return {
            "Action": self.prev_actions[-1] if self.prev_actions else "NULL"
        }

    def _run_action_on_emulator(self, action, actionable_ticks=4) -> None:
        # Configure action
        pyboy_action_idx = int(action)
        if pyboy_action_idx == len(self.valid_actions) + 1:
            pyboy_action_idx = len(self.valid_actions) - 1
        
        curr_action = self.actions[pyboy_action_idx]

        # button() automatically releases after act_freq ticks unless re-pressed
        for button in curr_action:
            self.pyboy.button(button, self.act_freq)

        self.pyboy.tick(self.act_freq - 1, sound=False)

        if self._get_mario_on_ground():
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            
        self.pyboy.tick(1, sound=False)

        self.prev_actions.append(curr_action)


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
            return delta_distance * 0.1

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
