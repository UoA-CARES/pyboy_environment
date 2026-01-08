"""
The link below has all the ROM memory data for Super Mario Land.
It is used to extract the game state for the MarioEnvironment class.

https://datacrystal.tcrf.net/wiki/Super_Mario_Land/RAM_map

https://github.com/Baekalfen/PyBoy/blob/master/pyboy/plugins/game_wrapper_super_mario_land.py
https://github.com/lixado/PyBoy-RL/blob/main/AISettings/MarioAISettings.py
"""

from abc import ABCMeta

import numpy as np
from pyboy.utils import WindowEvent
from functools import cached_property
from collections import deque

from pyboy_environment.environments.pyboy_environment import PyboyEnvironment

DEFAULT_ACTIONS = [
    ["up"],
    ["down"],
    ["left"],
    ["right"],
    ["a"],
    ["b"],
    ["left", "a"],
    ["left", "b"],
    ["right", "a"],
    ["right", "b"],
]

class MarioEnvironment(PyboyEnvironment, metaclass=ABCMeta):
    def __init__(
        self,
        act_freq: int,
        action_space: list[list[str]] = DEFAULT_ACTIONS,
        init_state: str = "init.state",
        image_observation: bool = False,
        stack_states: int = 3,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        if image_observation:
            self._get_state = self._get_image_state

        self.prev_states = deque(maxlen=stack_states)
        self.prev_action = "None"

        super().__init__(
            rom_name="SuperMarioLand.gb",
            domain="mario",
            actions=action_space,
            init_state_file_name=init_state,
            act_freq=act_freq,
            emulation_speed=emulation_speed,
            headless=headless,
        )


    def _get_image_state(self) -> np.ndarray:
        frame = self.game_area()[np.newaxis, ...]
        self.prev_states.append(frame)
        return np.concatenate(self.prev_states)
    

    def _get_state(self) -> np.ndarray:
        vector = self.game_area().flatten()
        self.prev_states.append(vector)
        return np.concatenate(self.prev_states)
    

    def reset(self) -> np.ndarray:
        self.prev_states.clear()
        state = super().reset()
        self.prev_action = "None"

        while len(self.prev_states) < self.prev_states.maxlen:
            self.prev_states.append(state)
        state = np.concatenate(self.prev_states)
        
        return state


    def _run_action_on_emulator(self, action: int | float) -> None:
        """
        Mario-specific parsing of action inputs to emulator using `pyboy.button()`
        
        :param self
        :param action: Index of action to perform
        """
        pyboy_action_idx = int(action) # Cast outputs of continuous algs to int
        curr_action = self.actions[pyboy_action_idx]

        # Queue each action button press and press duration in emulator
        for button in curr_action:
            self.pyboy.button(button, self.act_freq)

        self.pyboy.tick(self.act_freq - 1, sound=False)

        # Release 'A' one tick early if Mario is on the ground so that next jump can be performed
        if self._get_mario_on_ground():
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

        self.pyboy.tick(1, sound=False)
        
        self.prev_action = curr_action


    def _generate_game_stats(self) -> dict[str, int]:
        return {
            "lives": self._get_lives(),
            "score": self._get_score(),
            "coins": self._get_coins(),
            "stage": self._get_stage(),
            "world": self._get_world(),
            "x_position": self._get_x_position(),
            "time": self._get_time(),
            "dead_timer": self._get_dead_timer(),
            "dead_jump_timer": self._get_dead_jump_timer(),
            "game_over": self._get_game_over(),
        }

    def _get_x_position(self):
        # Copied from: https://github.com/lixado/PyBoy-RL/blob/main/AISettings/MarioAISettings.py
        # Do not understand how this works...
        level_block = self._read_m(0xC0AB)
        mario_x = self._read_m(0xC202)
        scx = self.pyboy.screen.tilemap_position_list[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
        real_x_position = level_block * 16 + real + mario_x
        return real_x_position
    
    def _get_mario_on_ground(self):
        return self._read_m(0xC20A)

    def _get_time(self):
        hundreds = self._read_m(0x9831)
        tens = self._read_m(0x9832)
        ones = self._read_m(0x9833)
        return int(str(hundreds) + str(tens) + str(ones))

    def _get_lives(self):
        return self._read_m(0xDA15)

    def _get_score(self):
        mario = self.pyboy.game_wrapper
        return mario.score

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

    def _get_dead_timer(self):
        return self._read_m(0xFFA6)

    def _get_dead_jump_timer(self):
        return self._read_m(0xC0AC)

    def game_area(self) -> np.ndarray:
        mario = self.pyboy.game_wrapper
        mario.game_area_mapping(mario.mapping_compressed, 0)
        return mario.game_area()

    def get_overlay_info(self) -> dict:
        return {
            "Action": self.prev_action
        }

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def max_action_value(self) -> float:
        return len(self.actions)

    @cached_property
    def observation_space(self) -> int | tuple[int]:
        shape = self._get_state().shape
        if len(shape) > 1:
            return shape
        else:
            return shape[0]

    @cached_property
    def action_num(self) -> int:
        return len(self.actions)
    
    def sample_action(self) -> list[int]:
        length = self.action_num
        random_index = np.random.randint(0, length)
        return np.array([random_index])