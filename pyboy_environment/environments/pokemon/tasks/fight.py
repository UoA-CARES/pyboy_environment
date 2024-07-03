from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonFight(PokemonEnvironment):
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

        super().__init__(
            act_freq=act_freq,
            task="fight",
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
        reward = -10
        reward += self._xp_reward(new_state) * 10
        reward += self._enemy_health_reward(new_state) * 50
        # reward += self._player_defeated_reward(new_state)
        reward += self._levels_reward(new_state)
        reward += self._start_battle_reward(new_state)
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["party_size"] > self.prior_game_stats["party_size"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 500

    def _start_battle_reward(self, new_state) -> int:
        if (new_state["battle_type"] != 0):
            return 20
        return 0

    def _levels_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if (old_levels[i] != 0):
                reward += 10000 * (new_levels[i] / old_levels[i] - 1)
        return reward
