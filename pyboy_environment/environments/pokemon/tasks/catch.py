import numpy as np

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)

# rewards
DO_NOTHING_BASE = -1
START_BATTLE_REWARD = 10
THROW_POKEBALL_REWARD = 100
CATCH_POKEMON_REWARD = 500
BUY_POKEBALL_REWARD = 100

# other params
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
        return np.array([])

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = DO_NOTHING_BASE
        reward += self._start_battle_reward(new_state)
        reward += self._throw_pokeball_reward(new_state, reward=THROW_POKEBALL_REWARD)
        reward += self._catch_pokemon_reward(new_state, reward=CATCH_POKEMON_REWARD)
        reward += self._buy_pokeball_reward(new_state, reward=BUY_POKEBALL_REWARD)
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return False

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        return self.steps >= num_steps_truncate
