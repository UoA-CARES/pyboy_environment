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

        self.release_button_offset = 8
        self.stack_frames = 3
        self.prev_frames = deque(maxlen=self.stack_frames)

        if image_observation:
            self._get_state = self._get_state_image
        else:
            self._get_state = self._get_state_vector
            
        self.max_level_progress = 0
        self.prev_action = []   

        super().__init__(
            act_freq=act_freq,
            valid_actions=valid_actions,
            release_button=release_button,
            emulation_speed=emulation_speed,
            headless=headless,
        )


    def _get_state_image(self) -> np.ndarray:
        return np.array(self.game_area())[np.newaxis, ...]


    def _get_state_vector(self) -> Dict[str, int]:
        frame = self.game_area().flatten()
        self.prev_frames.append(frame)
        state = np.concatenate(self.prev_frames)
        return state


    def reset(self, training: bool = False) -> np.ndarray:
        self.prev_action = []
        self.prev_frames.clear()
        self.max_level_progress = self.prior_game_stats["x_position"]

        state = super().reset()
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
        return {}

    def _run_action_on_emulator(self, action, actionable_ticks=4) -> None:
        pyboy_action_idx = int(action)

        if pyboy_action_idx >= len(self.valid_actions):
            pyboy_action_idx = len(self.valid_actions) - 1
        
        curr_action = self.valid_actions[pyboy_action_idx]

        for action_event in curr_action:
            self.pyboy.send_input(action_event)

        for action_event in self.prev_action:
            if action_event not in curr_action:
                self.pyboy.send_input(action_event + self.release_button_offset)
        
        self.pyboy.tick(self.act_freq, sound=False)

        self.prev_action = curr_action


    def _calculate_reward(self, new_state: Dict[str, int]) -> float:
        in_menu = new_state["world"] == 44
        if in_menu:
            return -1.0

        reward_stats = {
            "position_reward": self._position_reward(new_state),
            "lives_reward": self._lives_reward(new_state),
            "score_reward": self._score_reward(new_state),
        }

        reward_total: int = -1
        for name, reward in reward_stats.items():
            logging.debug(f"{name} reward: {reward}")
            reward_total += reward

        sigmoid_reward = 1/(1+np.exp(-reward_total))

        return sigmoid_reward

    def _position_reward(self, new_state: Dict[str, int]) -> int:
        delta_distance = new_state["x_position"] - self.max_level_progress

        if new_state["x_position"] > self.max_level_progress:
            self.max_level_progress = new_state["x_position"]

        return 0.1 * max(0, delta_distance)

    def _score_reward(self, new_state: Dict[str, int]) -> int:
        delta_score = new_state["score"] - self.prior_game_stats["score"]
        if not delta_score:
            return 0
        # Typical score reward is 100 e.g. jumping on enemies or collecting coins
        return 1/(1+np.exp(-delta_score/100)) 

    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        delta_lives = new_state["lives"] - self.prior_game_stats["lives"]
        if not delta_lives:
            return 0
        if abs(delta_lives) > 0:
            return delta_lives * 0.5
        else:
            return -1

    def _time_reward(self, new_state: Dict[str, int]) -> int:
        time_reward = min(0, (new_state["time"] - self.prior_game_stats["time"]) * 10)
        return max(time_reward, -10)

    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]

    def _check_if_truncated(self, game_stats):
        # Truncated if mario dies or if done more than a 4000 steps/actions
        return self.steps >= 4000 or game_stats["game_over"]
