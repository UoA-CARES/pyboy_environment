import numpy as np

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)

# rewards
DO_NOTHING_BASE = -1
START_BATTLE_REWARD = 100
DEAL_DAMAGE_MULTIPLIER = 10
XP_MULTIPLIER = 10
LEVEL_UP_MULTIPLIER = 1000

# other params
NUM_STEPS_TRUNCATE = 500


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

    def _calculate_reward(self, new_state: dict[str, any]) -> float:
        # Implement your reward calculation logic here
        reward = DO_NOTHING_BASE
        reward += self._xp_increase_reward(new_state, multiplier=XP_MULTIPLIER)
        reward += self._deal_damage_reward(new_state, multiplier=DEAL_DAMAGE_MULTIPLIER)
        reward += self._levels_increase_reward(
            new_state, multiplier=LEVEL_UP_MULTIPLIER
        )
        reward += self._start_battle_reward(new_state, reward=START_BATTLE_REWARD)
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["party_size"] > self.prior_game_stats["party_size"]

    def _check_if_truncated(self, game_stats: dict[str, any]) -> bool:
        # Implement your truncation check logic here
        return self.steps >= NUM_STEPS_TRUNCATE
