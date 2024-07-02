from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

# Reward
# - Catch Unique Pokemon
# - Level Up Pokemon
# - Gain XP
#
# Input State
# - Current frame
# - Previous frames

class PokemonTryPlay(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.num_times_received_reward = 0

        super().__init__(
            act_freq=act_freq,
            task="try_play",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        pass

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = -1 # Default reward of -1
        reward += self._caught_reward(new_state) * 1000
        reward += self._levels_reward(new_state) * 100
        reward += self._xp_reward(new_state)
        if reward > 0:
            self.num_times_received_reward += 1
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["party_size"] > self.prior_game_stats["party_size"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        return self.num_times_received_reward > 20
    
    def reset(self):
        PokemonEnvironment.reset(self)
        self.num_times_received_reward = 0
