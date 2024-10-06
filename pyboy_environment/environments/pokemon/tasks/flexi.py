

import random


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
start_battle_multiplier = 100
enemy_health_loss_multiplier = 20
own_health_loss_multiplier = 10
xp_multiplier = 1
level_up_multiplier = 100

pokeball_thrown_multiplier = 100
pokemon_caught_multiplier = 500
bought_pokeball_multiplier = 100
enter_gym_multiplier = 100

uniqueness_multiplier = 0.1

# Outside Pokemart: x:29, y:23

class GoToPokemart():
    def __init__(self, pokemon_env: PokemonEnvironment):
        self.pokemon = pokemon_env
        game_stats = self.pokemon._generate_game_stats()
        self.name = f"GoToPokemart"
        self.map_id_lowest_y_vals = {}
        self.prev_dist = 0

    def get_reward(self, game_stats):
        current_location = game_stats["location"]

        reward = 0

        current_y_val = current_location["y"]
        map_id = current_location["map_id"]
        dist_from_target = (abs(29 - current_location["x"]) + abs(23 - current_location["y"]))

        if map_id == 1:
            diff = self.prev_dist - dist_from_target
            reward += 30 * diff
        else:
            min_y_value_on_this_map = self.map_id_lowest_y_vals.get(map_id, 1000)

            if current_y_val < min_y_value_on_this_map:
                reward += 30
                self.map_id_lowest_y_vals[map_id] = current_y_val
            
        self.prev_dist = dist_from_target
        
        return reward
    
    def is_done(self, game_stats):
        current_location = game_stats["location"]
        distance = abs(29 - current_location["x"]) + abs(23 - current_location["y"])

        return distance < 2 and current_location["map_id"] == 1
        

class Explore():
    def __init__(self, pokemon_env: PokemonEnvironment):
        self.pokemon = pokemon_env
        game_stats = self.pokemon._generate_game_stats()
        self.name = f"Explore"

        self.image_manager = ImageStorage()

    def get_reward(self, game_stats):

        frame = self.pokemon.grab_frame()

        uniqueness = (self.image_manager.add_image(np.mean(frame, axis=2))) * uniqueness_multiplier 

        if (uniqueness > 1.0):
            return uniqueness
        else:
            return 0

class PurchasePokeballs():
    def __init__(self, pokemon_env: PokemonEnvironment):
        self.pokemon = pokemon_env

        stats = self.pokemon._generate_game_stats()
        self.previous_pokeball_count = self.pokemon._get_pokeball_count(stats["items"])
        self.name = f"Purchase Pokeballs"
        self.has_been_in_shop = False
        self.had_left_shop = False

    def get_reward(self, game_stats):
        reward = 0

        current_pokeball_count = self.pokemon._get_pokeball_count(game_stats["items"])
        pokeball_count_increase = current_pokeball_count - self.previous_pokeball_count
        self.previous_pokeball_count = current_pokeball_count

        reward += pokeball_count_increase

        is_in_shop = game_stats["location"]["map_id"] == 42
        if (not self.has_been_in_shop and is_in_shop):
            self.has_been_in_shop = True
            reward += 300

        if (self.has_been_in_shop and not is_in_shop and not self.has_left_shop):
            self.has_left_shop = True
            reward += 300

        return reward

    def is_done(self, game_stats):
        self.pokemon._get_pokeball_count(game_stats["items"]) > 4 and self.has_left_shop

class CatchPokemon():
    def __init__(self, pokemon_env: PokemonEnvironment, target_party_size) -> None:
        self.name = "Catch Pokemon"
        self.pokemon = pokemon_env
        self.target_party_size = target_party_size

    def get_reward(self, game_stats):

        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self.pokemon._start_battle_reward(game_stats) * start_battle_multiplier
        reward += self.pokemon._pokeball_thrown_reward(game_stats) * pokeball_thrown_multiplier
        reward += self.pokemon._caught_reward(game_stats) * pokemon_caught_multiplier
        reward += self.pokemon._bought_pokeball_reward(game_stats) * bought_pokeball_multiplier 
        return reward
    
    def is_done(self, game_stats):
        return game_stats["party_size"] > self.target_party_size


class LevelUpPokemon():
    def __init__(self, pokemon_env: PokemonEnvironment, target_levels) -> None:
        self.name = "Level Up Pokemon"
        self.pokemon = pokemon_env
        self.target_levels = target_levels

    def get_reward(self, game_stats):
        # Implement your reward calculation logic here
        reward = do_nothing_base
        reward += self.pokemon._xp_increase_reward(game_stats) * xp_multiplier
        reward += self.pokemon._enemy_health_decrease_reward(game_stats) * enemy_health_loss_multiplier
        # reward += self._player_defeated_reward(new_state)
        reward += self.pokemon._levels_increase_reward(game_stats) * level_up_multiplier
        reward += self.pokemon._start_battle_reward(game_stats) * start_battle_multiplier
        return reward
    
    def is_done(self, game_stats):
        levels = game_stats["levels"]
        for level in levels:
            if level < self.target_levels and level != 0:
                return False

        return True
    
class FightBrock():
    def __init__(self, pokemon_env: PokemonEnvironment) -> None:
        self.name = "Fight Brock"
        self.pokemon = pokemon_env
        self.has_been_in_gym = False

    def get_reward(self, game_stats):
        # Implement your reward calculation logic here
        reward = do_nothing_base

        map_id = game_stats["location"]["map_id"]

        if not self.has_been_in_gym and map_id == 54:
            self.has_been_in_gym = True
            reward += enter_gym_multiplier

        reward += self.pokemon._start_battle_reward(game_stats, battle_type=2) * start_battle_multiplier
        reward += self.pokemon._xp_increase_reward(game_stats) * xp_multiplier
        reward += self.pokemon._enemy_health_decrease_reward(game_stats) * enemy_health_loss_multiplier
        reward += self.pokemon._levels_increase_reward(game_stats) * level_up_multiplier

        reward += self.pokemon._own_pokemon_health_decrease_punishment(game_stats) * own_health_loss_multiplier
        reward += self.pokemon._player_defeated_punishment(game_stats)
        return reward
    
    def is_done(self, game_stats):
        return self.pokemon._get_badge_count() > 0

class PokemonFlexiEnv(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        discrete: bool = False,
    ) -> None:


        super().__init__(
            act_freq=act_freq,
            task="flexi",
            init_name="caught_three_pokemon_outside_gym.state",
            emulation_speed=emulation_speed,
            headless=headless,
            discrete=discrete,
        )

        self.starting_state_paths = [self.init_path]

        self.last_step_checkpoint = self.steps

    def get_current_task(self):
        return self.tasks[self.task_index]

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        pass

    def _calculate_reward(self, new_state: dict) -> float:
        
        current_task = self.tasks[self.task_index]

        reward = do_nothing_base
        reward += current_task.get_reward(new_state)

        if (current_task.is_done(new_state)):
            with open(f"task_index_{self.task_index}.state", "wb") as f:
                self.pyboy.save_state(f)
                file_name = f.name
                if (file_name not in self.starting_state_paths):
                    self.starting_state_paths.append(f.name)
            reward += 300
            self.task_index += 1
            self.last_step_checkpoint = self.steps

        return reward

    def reset(self):
        self.tasks = [
            FightBrock(self)
        ]
        
        self.task_index = 0

        if (not hasattr(self, "starting_state_paths")):
            self.starting_state_paths = [self.init_path]

        # Picks a random starting state from existing save points
        self.init_path = random.choice(self.starting_state_paths)
        return super().reset()

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return self.task_index >= len(self.tasks)

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        return self.steps - self.last_step_checkpoint >= 900
