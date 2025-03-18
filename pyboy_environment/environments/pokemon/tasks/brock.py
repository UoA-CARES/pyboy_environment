import numpy as np
from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)

# Reward Constants
# Larger value giving to sparser rewards
# Smaller value giving to more frequently experiened rewards
BASE_REWARD = -2
IN_GRASS_REWARD = 1
START_BATTLE_REWARD = 100
DEAL_DAMAGE_MULTIPLIER = 100
GAIN_XP_MULTIPLER = 10
LEVEL_UP_MULTIPLIER = 10000
MOVE_UP_REWARD = 1
ENTER_POKEMART_REWARD = 1000
PURCHASE_POKEBALL_MULTIPLIER = 500
THROW_POKEBALL_MULTIPLIER = 100
CATCH_POKEMON_REWARD = 1000
TASK_COMPLETION_MULTIPLIER = 10000
MOVE_CLOSER_TO_GYM_REWARD = 10000

STEPS_TRUNCATION = 500
TASK_COMPLETION_EXTRA_STEPS = 600
LEVEL_UP_EXTRA_STEPS_MULTIPLIER = 10
FIND_BROCK_EXTRA_STEPS = 200

NUM_TASKS = 8


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
        discrete: bool = False,
    ) -> None:
        self.tasks = [0] * NUM_TASKS
        self.tasks[0] = 1
        self.current_task = 0

        super().__init__(
            act_freq=act_freq,
            task="fight",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            headless=headless,
            discrete=discrete,
        )

    ################################################################
    ########################### Game Info ##########################
    ################################################################

    # Represent tasks as array of 0/1 values for each task
    def _set_tasks(self, game_stats: dict):
        active_task = self._select_task(game_stats)

        if active_task != self.current_task:
            self.tasks[self.current_task] = 0
            self.tasks[active_task] = 1
            self.current_task = active_task

        game_stats["tasks"] = self.tasks

    def _select_task(self, game_stats: dict) -> int:
        if game_stats["levels"][0] < 8:
            return 0  # fight
        elif game_stats["party_size"] < 3 and self._get_num_pokeballs() < 10:
            if game_stats["map_id"] != 1 and game_stats["map_id"] != 0x2A:
                return 1  # enter pokemart village
            elif game_stats["map_id"] == 1:
                return 2  # enter pokemart
            elif game_stats["map_id"] == 0x2A:
                return 3  # purchase pokeballs
        elif game_stats["party_size"] < 3:
            return 4  # catch pokemon
        elif self._assert_min_pokemon_level(game_stats, 4):
            return 5  # train party to specified level
        elif game_stats["map_id"] != 0x36:
            return 6  # find gym
        else:
            return 7  # defeat brock

    def _assert_min_pokemon_level(self, game_stats: dict, min: int) -> bool:
        num_pokemon = game_stats["party_size"]
        levels = game_stats["levels"]

        for i in range(num_pokemon):
            if levels[i] < min:
                return False

        return True

    def _get_location(self) -> dict[str, any]:
        # OVERRIDE to remove map name (string)
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {
            "x": x_pos,
            "y": y_pos,
            "map_id": map_n,
        }

    def _get_current_selected_menu_item(self) -> int:
        return self._read_m(0xCC26)

    def _get_index_current_pokemon(self) -> int:
        return self._read_m(0xCC2F)

    def _get_current_pokemon_id(self) -> int:
        return self._read_m(0xD014)

    def _get_num_pokeballs(self) -> int:
        items = self._read_items()
        keys = items.keys()
        num_pokeballs = 0

        for i in range(0x5):
            key = f"item_{i}"
            if key in keys:
                num_pokeballs += items[key]

        return num_pokeballs

    ################################################################
    ######################## Training Info #########################
    ################################################################

    # Defines which information should be displayed in training videos
    def get_overlay_info(self) -> dict:
        return {
            "task": self.tasks.index(1),
        }

    def _generate_game_stats(self) -> dict[str, any]:

        game_stats = {
            **self._get_location(),
            "in_grass": self._is_in_grass_tile(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            "levels": self._read_party_level(),
            **self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            "money": self._read_money(),
            "battle_type": self._read_battle_type(),
            "enemy_pokemon_health": self._get_enemy_pokemon_health(),
            "current_pokemon_id": self._get_current_pokemon_id(),
            "num_pokeballs": self._get_num_pokeballs(),
            "current_selected_menu_item": self._get_current_selected_menu_item(),
        }

        self._set_tasks(game_stats)

        return game_stats

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()
        state = self._get_state_from_stats(game_stats)
        return state

    def _get_state_from_stats(self, game_stats: dict) -> np.ndarray:
        state = []
        for value in game_stats.values():
            if isinstance(value, list):
                for i in value:
                    state.append(i)
            else:
                state.append(value)
        return np.array(np.array(state))

    ################################################################
    ####################### Reward Functions #######################
    ################################################################

    ### Override to include custom step logic
    def _levels_reward(self, new_state: dict[str, any]) -> float:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if new_levels[i] > old_levels[i]:
                reward += (new_levels[i] / old_levels[i] - 1) * LEVEL_UP_MULTIPLIER
                self.steps -= new_levels[i] ** 2 * LEVEL_UP_EXTRA_STEPS_MULTIPLIER
                # add extra hundred steps after a level up is performed
        return reward

    ################################################################
    ##################### Task Reward Functions ####################
    ################################################################

    def _reward_task_fight_pokemon(self, new_state: dict) -> float:
        reward = self._is_in_grass_reward(reward=IN_GRASS_REWARD)
        reward += self._start_battle_reward(new_state, reward=START_BATTLE_REWARD)
        reward += self._deal_damage_reward(new_state, multiplier=DEAL_DAMAGE_MULTIPLIER)
        reward += self._xp_increase_reward(new_state, multiplier=GAIN_XP_MULTIPLER)
        reward += self._levels_reward(new_state)
        return reward

    def _reward_task_enter_v_city(self, new_state: dict) -> float:
        if new_state["map_id"] != 0x0C or new_state["map_id"] != 0x00:
            return 0

        if new_state["y"] < self.prior_game_stats["x"] or (
            new_state["map_id"] == 0x0C and self.prior_game_stats["map_id"] == 00
        ):
            return MOVE_UP_REWARD

        return 0

    def _reward_task_enter_pokemart(self, new_state: dict) -> float:
        if self.prior_game_stats["map_id"] != 0x2A and new_state["map_id"] == 0x2A:
            return ENTER_POKEMART_REWARD
        else:
            return 0

    def _reward_task_buy_pokeball(self, new_state: dict) -> float:
        delta_pokeball = self._get_num_pokeballs(new_state) - self._get_num_pokeballs(
            self.prior_game_stats
        )
        if delta_pokeball > 0:
            return delta_pokeball * PURCHASE_POKEBALL_MULTIPLIER
        return 0

    def _reward_task_catch_pokemon(self, new_state: dict, reward) -> float:
        reward = self._throw_pokeball_reward(new_state, THROW_POKEBALL_MULTIPLIER)
        reward += self._catch_pokemon_reward(
            new_state, CATCH_POKEMON_REWARD, reward > 0
        )
        return reward

    def _reward_task_find_gym(self, new_state: dict) -> float:
        room_ids = [0x00, 0x0C, 0x01, 0x0D, 0x32, 0x33, 0x2F, 0x0D, 0x02, 0x36]
        old_index = room_ids.index(self.prior_game_stats["map_id"])
        new_index = room_ids.index(new_state["map_id"])

        if new_index > old_index:
            self.steps -= FIND_BROCK_EXTRA_STEPS
            return MOVE_CLOSER_TO_GYM_REWARD
        elif new_index < old_index:
            if self.prior_game_stats["map_id"] == 0x2F and new_state["map_id"] == 0x0D:
                self.steps -= FIND_BROCK_EXTRA_STEPS
                return MOVE_CLOSER_TO_GYM_REWARD  # Edge case as 0x0d is found twice
            else:
                return -MOVE_CLOSER_TO_GYM_REWARD
        else:
            return 0

    def _reward_task_fight_brock(self, new_state: dict) -> float:
        reward = 0
        if new_state["battle_type" == 2]:
            if self.prior_game_stats["battle_type"] == 2:
                reward += 5
            else:
                reward += 100

        reward += self._deal_damage_reward(new_state)

    def _get_task_index_diff(self, new_state: dict) -> float:
        old_task = self.prior_game_stats["tasks"].index(1)
        new_task = new_state["tasks"].index(1)

        if new_task > old_task:
            1

        return new_task - old_task

    def _calculate_reward(self, new_state: dict) -> float:
        reward = BASE_REWARD

        # compute the reward for the new state given the previous task and the corresponding action
        task = self.prior_game_stats["tasks"].index(1)
        if task == 0 or task == 5:
            reward += self._reward_task_fight_pokemon(new_state)
        elif task == 1:
            reward += self._reward_task_enter_v_city(new_state)
        elif task == 2:
            reward += self._reward_task_enter_pokemart(new_state)
        elif task == 3:
            reward += self._reward_task_buy_pokeball(new_state)
        elif task == 4:
            reward += self._reward_task_catch_pokemon(new_state)
        elif task == 5:
            reward += self._reward_task_find_gym(new_state)
        else:  # must be 7
            reward += self._reward_task_fight_brock(new_state)

        # reward for completing a task (is negative for reverting to previous task)
        task_diff = self._get_task_index_diff(new_state)

        if task_diff == 1:
            self.steps -= TASK_COMPLETION_EXTRA_STEPS
        else:
            task_diff * TASK_COMPLETION_MULTIPLIER

        return reward

    ################################################################
    ##################### Termination Functions ####################
    ################################################################

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        return self.steps >= STEPS_TRUNCATION
