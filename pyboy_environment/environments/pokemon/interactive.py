from pyboy_environment.environments.pokemon.tasks.fight import PokemonFight
import pyboy_environment.suite as Suite

import sys
import termios
import tty
import time


def wait_for_input():
    key_mapping = {
        "\x1b[A": 3,
        "\x1b[B": 0,
        "\x1b[C": 2,
        "\x1b[D": 1,
        "a": 4,
        "b": 5,
    }

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)

        if key == "\x1b":
            key += sys.stdin.read(2)

        time.sleep(0.1)
    except:
        return -1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key_mapping.get(key)


def main(argv):
    if len(argv) < 2:
        print("Usage: interactive.py <domain> <task>")
        sys.exit(1)

    env = Suite.make(argv[0], argv[1], 24, headless=False, discrete=True)

    env.step(3) # initial step to load screen

    while True:
        index = wait_for_input()
        if index == -1:
            continue
        [state, reward, done, truncated] = env.step(index)
        print(
            f"\rReward: {reward}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
