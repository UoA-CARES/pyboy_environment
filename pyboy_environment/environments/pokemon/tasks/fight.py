from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

# rewards

do_nothing_base = -1
start_battle_reward = 100
enemy_health_loss_multiplier = 10
xp_multiplier = 10
level_up_multiplier = 1000

# other params
num_steps_truncate = 500


class PokemonFight(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        discrete: bool = False,
    ) -> None:

        super().__init__(
            act_freq=act_freq,
            task="fight",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            headless=headless,
            discrete=discrete,
        )

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        return np.array([])

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self._xp_increase_reward(new_state) * xp_multiplier
        reward += (
            self._enemy_health_decrease_reward(new_state) * enemy_health_loss_multiplier
        )
        # reward += self._player_defeated_reward(new_state)
        reward += self._levels_increase_reward(new_state) * level_up_multiplier
        reward += self._start_battle_reward(new_state)
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["party_size"] > self.prior_game_stats["party_size"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= num_steps_truncate

    def _start_battle_reward(self, new_state) -> int:
        if new_state["battle_type"] != 0 and self.prior_game_stats["battle_type"] == 0:
            return start_battle_reward
        return 0

    def _levels_increase_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if old_levels[i] != 0:
                reward += new_levels[i] / old_levels[i] - 1
        return reward
