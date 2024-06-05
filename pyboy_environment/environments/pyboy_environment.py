from abc import ABCMeta, abstractmethod
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from pyboy import PyBoy


class PyboyEnvironment(metaclass=ABCMeta):

    def __init__(
        self,
        task: str,
        domain: str,
        rom_name: str,
        init_state_file_name: str,
        act_freq: int,
        valid_actions: list,
        release_button: list,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        self.task = task
        self.domain = domain

        path = f"{Path.home()}/cares_rl_configs/{self.domain}"
        self.rom_path = f"{path}/{rom_name}"
        self.init_path = f"{path}/task_init_states/{init_state_file_name}"

        self.combo_actions = 0

        self.valid_actions = valid_actions

        self.release_button = release_button

        self.act_freq = act_freq

        head = "null" if headless else "SDL2"
        self.pyboy = PyBoy(
            self.rom_path,
            window=head,
        )

        self.prior_game_stats = self._generate_game_stats()
        self.screen = self.pyboy.screen

        self.steps = 0

        self.seed = 0

        self.pyboy.set_emulation_speed(emulation_speed)

        self.reset()

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        # There isn't a random element to set that I am aware of...

    def reset(self) -> np.ndarray:
        self.steps = 0

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.prior_game_stats = self._generate_game_stats()

        return self._get_state()

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = np.array(self.screen.image)
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def game_area(self) -> np.ndarray:
        return self.pyboy.game_area()

    def step(self, action) -> tuple:
        self.steps += 1

        self._run_action_on_emulator(action)

        state = self._get_state()

        current_game_stats = self._generate_game_stats()
        reward = self._calculate_reward(current_game_stats)

        done = self._check_if_done(current_game_stats)
        truncated = self._check_if_truncated(current_game_stats)

        self.prior_game_stats = current_game_stats

        return state, reward, done, truncated

    def _read_m(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def _read_bit(self, addr: int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self._read_m(addr))[-bit - 1] == "1"

    # built-in since python 3.10
    def _bit_count(self, bits: int) -> int:
        return bin(bits).count("1")

    def _read_triple(self, start_add: int) -> int:
        return (
            256 * 256 * self._read_m(start_add)
            + 256 * self._read_m(start_add + 1)
            + self._read_m(start_add + 2)
        )

    def _read_bcd(self, num: int) -> int:
        return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

    @abstractmethod
    @cached_property
    def min_action_value(self) -> float:
        pass

    @abstractmethod
    @cached_property
    def max_action_value(self) -> float:
        pass

    @abstractmethod
    @cached_property
    def observation_space(self) -> int:
        pass

    @abstractmethod
    @cached_property
    def action_num(self) -> int:
        pass

    @abstractmethod
    def sample_action(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def _run_action_on_emulator(self, action) -> None:
        pass

    @abstractmethod
    def _generate_game_stats(self) -> dict:
        pass

    @abstractmethod
    def _calculate_reward(self, new_state: dict) -> float:
        pass

    @abstractmethod
    def _check_if_done(self, game_stats: dict) -> bool:
        pass

    @abstractmethod
    def _check_if_truncated(self, game_stats: dict) -> bool:
        pass
