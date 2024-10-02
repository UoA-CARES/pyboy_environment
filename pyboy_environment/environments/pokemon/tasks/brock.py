


from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
import math
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

from pyboy_environment.environments import PyboyEnvironment

from pyboy_environment.environments.pokemon.helpers.unique_im import ImageStorage

# rewards

do_nothing_base = -1
start_battle_reward = 100
enemy_health_loss_multiplier = 10
xp_multiplier = 10
level_up_multiplier = 1000

pokeball_thrown_multiplier = 100
caught_multiplier = 500
bought_pokeball_multiplier = 100

uniqueness_multiplier = 0.2 

# Outside Pokemart: x:29, y:23
class GoToLocation():
    def __init__(self, pokemon_env: PokemonEnvironment, location):
        self.pokemon = pokemon_env
        self.target_location = location
        game_stats = self.pokemon._generate_game_stats()
        self.previous_distance = self._get_distance(game_stats)
        self.name = f"Go to {location['x']}, {location['y']}"

    def _get_distance(self, game_stats):

        current_location = game_stats["location"]

        if current_location["map_id"] != self.target_location["map_id"]:
            return float("inf")

        distance = math.sqrt((current_location['x'] - self.target_location['x'])**2 + (current_location['y'] - self.target_location['y'])**2)
        return distance

    def get_reward(self, game_stats):
        return -min(self._get_distance(game_stats), 100)

    def is_done(self, game_stats):
        dist = self._get_distance(game_stats)
        return dist < 5.0

class Explore():
    def __init__(self, pokemon_env: PokemonEnvironment):
        self.pokemon = pokemon_env
        self.target_location = location
        game_stats = self.pokemon._generate_game_stats()
        self.previous_distance = self._get_distance(game_stats)
        self.name = f"Go to {location['x']}, {location['y']} (id: {location['map_id']})"

        self.image_manager = ImageStorage()

    def get_reward(self, game_stats):

        frame = self.pokemon.grab_frame()

        uniqueness = (self.image_manager.add_image(np.mean(frame, axis=2))) * uniqueness_multiplier 

        if (uniqueness > 10.0):
            return uniqueness
        else:
            return 0

class PurchasePokeballs():
    def __init__(self, pokemon_env: PokemonEnvironment):
        self.pokemon = pokemon_env

        stats = self.pokemon._generate_game_stats()
        self.previous_pokeball_count = self.pokemon._get_pokeball_count(stats["items"])
        self.name = f"Purchase Pokeballs"

    def get_reward(self, game_stats):
        current_pokeball_count = self.pokemon._get_pokeball_count(game_stats["items"])
        diff = current_pokeball_count - self.previous_pokeball_count

        self.previous_pokeball_count = current_pokeball_count
        return diff

    def is_done(self, game_stats):
        self.pokemon._get_pokeball_count(game_stats["items"]) > 4

class CatchPokemon():
    def __init__(self, pokemon_env: PokemonEnvironment) -> None:
        self.name = "Catch Pokemon"
        self.pokemon = pokemon_env

    def get_reward(self, game_stats):

        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self._start_battle_reward(game_stats)
        reward += pokeball_thrown_multiplier * self._pokeball_thrown_reward(game_stats)
        reward += caught_multiplier * self._caught_reward(game_stats)
        reward += bought_pokeball_multiplier * self._bought_pokeball_reward(game_stats)
        return reward
    
    def is_done(self, game_stats):
        return False


class LevelUpPokemon():
    def __init__(self, pokemon_env: PokemonEnvironment, target_levels) -> None:
        self.name = "Fight pokemon"
        self.pokemon = pokemon_env
        self.target_levels = target_levels

    def get_reward(self, game_stats):
        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self.pokemon._xp_reward(game_stats) * xp_multiplier
        reward += self.pokemon._enemy_health_reward(game_stats) * enemy_health_loss_multiplier
        # reward += self._player_defeated_reward(new_state)
        reward += self.pokemon._levels_reward(game_stats) * level_up_multiplier
        reward += self.pokemon._start_battle_reward(game_stats)
        return reward
    
    def is_done(self, game_stats):
        levels = game_stats["levels"]
        for level in levels:
            if level < self.target_levels and level != 0:
                return False

        return True

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        discrete: bool = False,
    ) -> None:

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            headless=headless,
            discrete=discrete,
        )

        self.last_step_checkpoint = self.steps

    def get_current_task(self):
        return self.tasks[self.task_index]

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        pass

    def _calculate_reward(self, new_state: dict) -> float:
        
        current_task = self.tasks[self.task_index]
        reward = current_task.get_reward(new_state)

        # Need to incorporate task index in new_state somehow
        if (current_task.is_done(new_state)):
            reward += 1000
            self.task_index += 1
            self.last_step_checkpoint = self.steps

        return reward

    def reset(self):
        self.explore_task = Explore(self)
        self.tasks = [FightPokemon]

        self.task_index = 0
        return super().reset()

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return self.task_index >= len(self.tasks)

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        return self.steps - self.last_step_checkpoint >= 500
