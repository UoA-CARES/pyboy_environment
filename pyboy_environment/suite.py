from pyboy_environment.environments import PyboyEnvironment
from pyboy_environment.environments.mario.mario_run import MarioRun
from pyboy_environment.environments.pokemon.tasks.catch import PokemonCatch
from pyboy_environment.environments.pokemon.tasks.fight import PokemonFight
from pyboy_environment.environments.pokemon.tasks.brock import PokemonBrock


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
            raise ValueError(f"Unknown Mario task: {task}")
    elif domain == "pokemon":
        if task == "catch":
            env = PokemonCatch(act_freq, emulation_speed, headless)
        elif task == "fight":
            env = PokemonFight(act_freq, emulation_speed, headless)
        elif task == "brock":
            env = PokemonBrock(act_freq, emulation_speed, headless)
        else:
            raise ValueError(f"Unknown Pokemon task: {task}")
    else:
        raise ValueError(f"Unknown pyboy environment: {task}")
    return env
