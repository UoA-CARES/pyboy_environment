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

from pyboy_environment.environments.pyboy_environment import PyboyEnvironment


class MarioEnvironment(PyboyEnvironment, metaclass=ABCMeta):
    def __init__(
        self,
        act_freq: int,
        valid_actions: list[WindowEvent],
        release_button: list[WindowEvent],
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        super().__init__(
            task="mario",
            rom_name="SuperMarioLand.gb",
            init_state_file_name="init.state",
            domain="mario",
            act_freq=act_freq,
            valid_actions=valid_actions,
            release_button=release_button,
            emulation_speed=emulation_speed,
            headless=headless,
        )

    def _get_state(self) -> np.ndarray:
        # TODO parameter as to whether to flatten this view or not
        # TODO image based being frame or game area frame...
        return self.game_area().flatten().tolist()

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
