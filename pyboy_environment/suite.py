from pyboy_environment.environments import (
    PyboyEnvironment,
    PokemonEnvironment,
)
from pyboy_environment.environments.mario.mario_run import MarioRun


def make(
    domain: str,
    task: str,
    act_freq: int,
    emulation_speed: int = 0,
    headless: bool = False,
) -> PyboyEnvironment:

    if domain == "mario":
        if task == "run":
            env = MarioRun(act_freq, emulation_speed, headless)
        else:
            raise ValueError(f"Unkown Mario task: {task}")
    elif domain == "pokemon":
        env = PokemonEnvironment(act_freq, emulation_speed, headless)
    else:
        raise ValueError(f"Unkown pyboy environment: {task}")
    return env
