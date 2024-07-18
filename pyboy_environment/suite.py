from pyboy_environment.environments import PyboyEnvironment
from pyboy_environment.environments.mario.mario_run import MarioRun
from pyboy_environment.environments.pokemon.tasks.catch import PokemonCatch
from pyboy_environment.environments.pokemon.tasks.fight import PokemonFight

from pyboy_environment.image_wrapper import ImageWrapper


def create_environment(config, image_observation: bool = True) -> PyboyEnvironment:

    domain = config.domain
    task = config.task
    if domain == "mario":
        if task == "run":
            env = MarioRun(config.act_freq, config.emulation_speed, config.headless)
        else:
            raise ValueError(f"Unknown Mario task: {task}")
    elif domain == "pokemon":
        if task == "catch":
            env = PokemonCatch(config)
        elif task == "fight":
            env = PokemonFight(config)
        else:
            raise ValueError(f"Unknown Pokemon task: {task}")
    else:
        raise ValueError(f"Unknown pyboy environment: {task}")
    if image_observation:
        env = ImageWrapper(config, env)
    return env
