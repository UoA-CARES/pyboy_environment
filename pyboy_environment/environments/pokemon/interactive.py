

from pyboy_environment.environments.pokemon.tasks.flexi import PokemonFlexiEnv
from tasks.fight import PokemonFight

import sys
import termios
import tty
import time

def wait_for_input():
    key_mapping = {
        '\x1b[A': 3,
        '\x1b[B': 0,
        '\x1b[C': 2,
        '\x1b[D': 1,
        'a': 4,
        'b': 5,
    }

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)


    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)

        if key == '\x1b':
            key += sys.stdin.read(2)
        
        time.sleep(0.1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key_mapping.get(key)


def main():

    env = PokemonFlexiEnv(act_freq=10, discrete=True)
    steps = 0

    while(True):
        steps += 1
        index = wait_for_input()
        [state, reward, done, truncated] = env.step(index)
        print(f"\rKey: {index}, Steps: {steps}, Counter: {env.continue_counter}, Rate: {env.continue_subtract_rate}, Reward: {reward} ({env.reward_rolling_average_buffer.average()}), Done/Trunc: {done or truncated}\r")


if __name__ == "__main__":
    main()
