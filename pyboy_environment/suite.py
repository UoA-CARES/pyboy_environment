from pyboy_environment.environments.environment import PyboyEnvironment
from pyboy_environment.environments.mario.mario_environment import MarioEnvironment
from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)


def make(
    task: str,
    act_freq: int,
    emulation_speed: int = 0,
    headless: bool = False,
) -> PyboyEnvironment:

    if task == "mario":

        env = MarioEnvironment(act_freq, emulation_speed, headless)
    elif task == "pokemon":
        env = PokemonEnvironment(act_freq, emulation_speed, headless)
    else:
        raise ValueError(f"Unkown pyboy environment: {task}")
    return env
