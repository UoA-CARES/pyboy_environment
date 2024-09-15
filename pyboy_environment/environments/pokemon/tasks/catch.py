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
pokeball_thrown_multiplier = 100
caught_multiplier = 500

num_steps_truncate = 500

class PokemonCatch(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        # We don't include start button here because we don't need it for this task
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
            task="catch",
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
        reward = do_nothing_base
        reward += self._start_battle_reward(new_state)
        reward += pokeball_thrown_multiplier * self._pokeball_thrown_reward(new_state)
        reward += caught_multiplier * self._caught_reward(new_state)
        return reward

    def _pokeball_thrown_reward(self, new_state) -> int:
        new_items = new_state["items"]
        old_items = self.prior_game_stats["items"]

        for item, count in old_items:
            if item > 0x4:
                pass

            if item not in new_items:
                return 1
            elif (count - new_items[item] != 0):
                return 1

        return 0

    def _start_battle_reward(self, new_state) -> int:
        if (new_state["battle_type"] != 0 and self.prior_game_stats["battle_type"] == 0):
            return start_battle_reward
        return 0

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["party_size"] > self.prior_game_stats["party_size"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        items = game_stats["items"]
        no_balls = True

        for i in range(0x0, 0x5):
            if i in items:
                no_balls = False

            # Maybe if we run out of pokeballs...? or a max step count
        return no_balls or self.steps >= num_steps_truncate
