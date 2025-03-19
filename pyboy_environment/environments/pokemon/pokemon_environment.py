from functools import cached_property
from abc import abstractmethod

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pyboy_environment import PyboyEnvironment
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonEnvironment(PyboyEnvironment):
    def __init__(
        self,
        act_freq: int,
        task: str,
        emulation_speed: int = 0,
        headless: bool = False,
        init_name: str = "has_pokedex.state",
        discrete: bool = False,
    ) -> None:

        self.discrete = discrete

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
            task=task,
            rom_name="PokemonRed.gb",
            domain="pokemon",
            init_state_file_name=init_name,
            act_freq=act_freq,
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    ##################################################################################
    ############################## ENVIRONMENT CONTRACT ##############################
    ##################################################################################

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def observation_space(self) -> int:
        return len(self._get_state())

    @cached_property
    def action_num(self) -> int:
        if self.discrete:
            return len(self.valid_actions)

        return 1

    def sample_action(self) -> list[int]:
        if self.discrete:
            length = len(self.valid_actions)
            random_index = np.random.randint(0, length)
            return random_index

        return np.array([np.random.random()])

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here - compact state based representation

        game_stats = self._generate_game_stats()
        (state,) = (
            [
                game_stats["location"]["x"],
                game_stats["location"]["y"],
                game_stats["location"]["map_id"],
                game_stats["battle_type"],
                game_stats["current_pokemon_health"],
                game_stats["enemy_pokemon_health"],
                game_stats["party_size"],
                game_stats["caught_pokemon"],
                game_stats["seen_pokemon"],
            ]
            + game_stats["hp"]["current"]
            + game_stats["hp"]["max"]
            + game_stats["xp"],
        )

        return state

    def _run_action_on_emulator(self, action, actionable_ticks=5) -> None:
        if self.discrete:
            pyboy_action_idx = action
        else:
            value = np.clip(action[0], 0.0, 0.9999999)

            bin_width = 1.0 / len(self.valid_actions)

            pyboy_action_idx = int(value // bin_width)

        if pyboy_action_idx >= len(self.valid_actions):
            pyboy_action_idx = len(self.valid_actions) - 1

        # At 2 ticks the agent can change direction it is looking on the spot
        # At 3 ticks the behaviour is not consistent
        # At 4 and more ticks the agent can change direction only by moving in that direction
        action_ticks = 4
        self.pyboy.send_input(self.valid_actions[pyboy_action_idx])
        self.pyboy.tick(action_ticks, render=False, sound=False)

        self.pyboy.send_input(self.release_button[pyboy_action_idx])
        self.pyboy.tick(self.act_freq - action_ticks, sound=False)

    @abstractmethod
    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        pass

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        pass

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        pass

    ##################################################################################
    ############################# MEMORY READING HELPERS #############################
    ##################################################################################

    def _generate_game_stats(self) -> dict[str, any]:
        stats = {
            "location": self._get_location(),
            "battle_type": self._read_battle_type(),
            "current_pokemon_id": self._get_active_pokemon_id(),
            "current_pokemon_health": self._get_current_pokemon_health(),
            "enemy_pokemon_health": self._get_enemy_pokemon_health(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            "type_id": self._read_party_type(),
            "type": [pkc.get_type(id) for id in self._read_party_type()],
            "hp": self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            "caught_pokemon": self._read_caught_pokemon_count(),
            "seen_pokemon": self._read_seen_pokemon_count(),
            "money": self._read_money(),
            "events": self._read_events(),
            "items": self._read_items(),
        }
        return stats

    def _get_location(self) -> dict[str, any]:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {
            "x": x_pos,
            "y": y_pos,
            "map_id": map_n,
            "map": pkc.get_map_location(map_n),
        }

    def _get_party_size(self) -> int:
        return self._read_m(0xD163)

    def _get_badge_count(self) -> int:
        return self._bit_count(self._read_m(0xD356))

    def _is_in_grass_tile(self) -> bool:
        player_sprite_status = self._read_m(0xC207)
        return player_sprite_status == 0x80

    def _get_pokeball_count(self, items) -> int:
        total_count = 0

        # Iterate through the dictionary of items the player (keys) has and their counts (values)
        for itemId, count in items.items():
            # Iterate through the types of Pokeballs. If the item (key) matches any of the Pokeball type ids, add the count to the total number of Pokeballs
            if itemId in range(0x0, 0x5):
                total_count += count

        return total_count

    def _read_party_id(self) -> list[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/pokemon_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def _read_party_type(self) -> list[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/type_constants.asm
        return [
            self._read_m(addr)
            for addr in [
                0xD170,
                0xD171,
                0xD19C,
                0xD19D,
                0xD1C8,
                0xD1C9,
                0xD1F4,
                0xD1F5,
                0xD220,
                0xD221,
                0xD24C,
                0xD24D,
            ]
        ]

    def _read_party_level(self) -> list[int]:
        return [
            self._read_m(addr)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]

    def _read_party_status(self) -> list[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/status_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]
        ]

    def _read_party_hp(self) -> dict[str, list[int]]:
        hp = [
            self._read_hp(addr)
            for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ]
        max_hp = [
            self._read_hp(addr)
            for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ]
        return {"current": hp, "max": max_hp}

    def _read_party_xp(self) -> list[int]:
        return [
            self._read_triple(addr)
            for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
        ]

    def _read_hp(self, start: int) -> int:
        return 256 * self._read_m(start) + self._read_m(start + 1)

    def _read_caught_pokemon_count(self) -> int:
        return sum(
            list(self._bit_count(self._read_m(i)) for i in range(0xD2F7, 0xD30A))
        )

    def _read_seen_pokemon_count(self) -> int:
        return sum(
            list(self._bit_count(self._read_m(i)) for i in range(0xD30A, 0xD31D))
        )

    def _read_money(self) -> int:
        return (
            100 * 100 * self._read_bcd(self._read_m(0xD347))
            + 100 * self._read_bcd(self._read_m(0xD348))
            + self._read_bcd(self._read_m(0xD349))
        )

    def _read_events(self) -> list[int]:
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        # museum_ticket = (0xD754, 0)
        # base_event_flags = 13
        return [
            self._bit_count(self._read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ]

    def _read_battle_type(self) -> int:
        return self._read_m(0xD057)

    def _read_items(self) -> dict[str, int]:
        # returns a dictionary of owned items
        # BROKEN (needs to be expressed in terms of its max capacity to avoid dictionary changing size and consequently input space)
        total_items = self._read_m(0xD31D)
        if total_items == 0:
            return {}

        addr = 0xD31E
        items = {}

        for i in range(total_items):
            item_id = self._read_m(addr + 2 * i)
            item_count = self._read_m(addr + 2 * i + 1)
            items[f"item_{item_id}"] = item_count

        return items

    def _get_active_pokemon_id(self) -> int:
        return self._read_m(0xD014)

    def _get_enemy_pokemon_health(self) -> int:
        return self._read_hp(0xCFE6)

    def _get_current_pokemon_health(self) -> int:
        return self._read_hp(0xD015)

    def _get_screen_background_tilemap(self):
        # SIMILAR TO CURRENT pyboy.game_wrapper()._game_area_np(), BUT ONLY FOR BACKGROUND TILEMAP, SO NPC ARE SKIPPED
        bsm = self.pyboy.botsupport_manager()
        ((scx, scy), (wx, wy)) = bsm.screen().tilemap_position()
        tilemap = np.array(bsm.tilemap_background()[:, :])
        return np.roll(np.roll(tilemap, -scy // 8, axis=0), -scx // 8, axis=1)[:18, :20]

    def _get_screen_walkable_matrix(self):
        walkable_tiles_indexes = []
        collision_ptr = self.pyboy.get_memory_value(0xD530) + (
            self.pyboy.get_memory_value(0xD531) << 8
        )
        tileset_type = self.pyboy.get_memory_value(0xFFD7)
        if tileset_type > 0:
            grass_tile_index = self.pyboy.get_memory_value(0xD535)
            if grass_tile_index != 0xFF:
                walkable_tiles_indexes.append(grass_tile_index + 0x100)
        for i in range(0x180):
            tile_index = self.pyboy.get_memory_value(collision_ptr + i)
            if tile_index == 0xFF:
                break
            else:
                walkable_tiles_indexes.append(tile_index + 0x100)
        screen_tiles = self._get_screen_background_tilemap()
        bottom_left_screen_tiles = screen_tiles[1 : 1 + screen_tiles.shape[0] : 2, ::2]
        walkable_matrix = np.isin(
            bottom_left_screen_tiles, walkable_tiles_indexes
        ).astype(np.uint8)
        return walkable_matrix

    def game_area_collision(self):
        shape = (20, 18)
        game_area_section = (0, 0) + shape
        width = game_area_section[2]
        height = game_area_section[3]

        game_area = np.ndarray(shape=(height, width), dtype=np.uint32)
        _collision = self._get_screen_walkable_matrix()
        for i in range(height // 2):
            for j in range(width // 2):
                game_area[i * 2][j * 2 : j * 2 + 2] = _collision[i][j]
                game_area[i * 2 + 1][j * 2 : j * 2 + 2] = _collision[i][j]
        return game_area

    ##################################################################################
    ############################### BASE REWARD HELPERS ##############################
    ##################################################################################

    def _buy_pokeball_reward(
        self, new_state: dict[str, any], reward: float = 1
    ) -> float:
        # Does not consider any other method of acquiring pokeballs
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])

        if new_count > previous_count:
            return reward

        return 0

    def _catch_pokemon_reward(
        self, new_state: dict[str, any], reward: float = 1, pokeball_thrown: bool = True
    ) -> float:
        if not pokeball_thrown:
            return 0

        previous_count = self.prior_game_stats["party_size"]
        new_count = new_state["party_size"]

        if new_count > previous_count:
            return reward

        return 0

    def _deal_damage_reward(
        self, new_state: dict[str, any], multiplier: float = 1
    ) -> float:
        damage_dealt = (
            self.prior_game_stats["enemy_pokemon_health"]
            - new_state["enemy_pokemon_health"]
        )

        if new_state["battle_type"] != self.prior_game_stats["battle_type"]:
            return 0

        return max(0, damage_dealt) * multiplier  # avoid punishing for enemy healing

    def _is_in_grass_reward(self, reward: float = 1) -> float:
        if self._is_in_grass_tile():
            return reward
        return 0

    def _levels_increase_reward(
        self, new_state: dict[str, any], multiplier: float = 1
    ) -> float:
        reward = 0
        prev_levels = self.prior_game_stats["levels"]
        current_levels = new_state["levels"]
        for i, prev_level in enumerate(prev_levels):
            increase = current_levels[i] - prev_level
            if increase == 0 or prev_level == 0:
                break  # no increase or pokemart was not in party before
            ratio = float(increase) / float(prev_level)
            reward += ratio

        return reward * multiplier

    def _start_battle_reward(
        self, new_state: dict[str, any], reward: float = 1, battle_type: int = 1
    ) -> float:
        if (
            new_state["battle_type"] == battle_type
            and self.prior_game_stats["battle_type"] == 0
        ):
            return reward
        return 0

    def _throw_pokeball_reward(
        self, new_state: dict[str, any], reward: float = 1
    ) -> float:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])

        if previous_count > new_count:
            return reward

        return 0

    def _xp_increase_reward(
        self, new_state: dict[str, any], multiplier: float = 1
    ) -> float:
        return (sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])) * multiplier

    ##################################################################################
    ################################# OTHER REWARDS ##################################
    ##################################################################################

    def _seen_reward(self, new_state: dict[str, any]) -> float:
        return new_state["seen_pokemon"] - self.prior_game_stats["seen_pokemon"]

    def _current_pokemon_health_reward(self, new_state: dict[str, any]) -> float:
        return sum(new_state["hp"]["current"]) - sum(
            self.prior_game_stats["hp"]["current"]
        )

    def _leave_battle_reward(self, new_state: dict[str, any]) -> float:
        if new_state["battle_type"] == 0:
            return 1
        return 0

    def _player_defeated_punishment(self, new_state: dict[str, any]) -> float:
        if sum(new_state["hp"]["current"]) == 0:
            return -1
        return 0

    def _current_health_reward(self, new_state: dict[str, any]) -> float:
        return (
            new_state["current_pokemon_health"]
            - self.prior_game_stats["current_pokemon_health"]
        )

    def _own_pokemon_health_decrease_punishment(
        self, new_state: dict[str, any]
    ) -> float:

        if new_state["battle_type"] == 0 or self.prior_game_stats["battle_type"] == 0:
            return 0

        if (
            new_state["current_pokemon_id"]
            != self.prior_game_stats["current_pokemon_id"]
        ):
            return 0

        previous_health = self.prior_game_stats["current_pokemon_health"]
        current_health = new_state["current_pokemon_health"]

        health_decrease = previous_health - current_health

        return -health_decrease  # negative as this is a punishment

    def _badges_reward(self, new_state: dict[str, any]) -> float:
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _money_reward(self, new_state: dict[str, any]) -> float:
        return new_state["money"] - self.prior_game_stats["money"]

    def _event_reward(self, new_state: dict[str, any]) -> float:
        return sum(new_state["events"]) - sum(self.prior_game_stats["events"])
