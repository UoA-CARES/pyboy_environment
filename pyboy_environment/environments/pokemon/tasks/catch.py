from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

# rewards
do_nothing_base = -1
start_battle_reward = 10
pokeball_thrown_multiplier = 100
caught_multiplier = 500
bought_pokeball_multiplier = 100

num_steps_truncate = 1000

class PokemonCatch(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        discrete: bool = False,
    ) -> None:

        super().__init__(
            act_freq=act_freq,
            task="catch",
            init_name="outside_pokemart.state",
            emulation_speed=emulation_speed,
            headless=headless,
            discrete=discrete,
        )

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        pass

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self._start_battle_reward(new_state)
        reward += pokeball_thrown_multiplier * self._pokeball_thrown_reward(new_state)
        reward += caught_multiplier * self._caught_reward(new_state)
        reward += bought_pokeball_multiplier * self._bought_pokeball_reward(new_state)
        return reward


    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return False


    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        return self.steps >= num_steps_truncate
