import os
import sys
import termios
import tty
import time

import pyboy_environment.suite as Suite
from pyboy.utils import PyBoyInvalidInputException


def wait_for_input():
    """
    Waits for user input and maps it to a corresponding action or state index.
    Returns a tuple (action_index, state_index).
    """
    key_mapping = {
        "\x1b[A": 3,  # UP arrow
        "\x1b[B": 0,  # DOWN arrow
        "\x1b[C": 2,  # RIGHT arrow
        "\x1b[D": 1,  # LEFT arrow
        "a": 4,       # A button
        "b": 5,       # B button
        "x": 6,       # Load state
        "z": 7,       # Save state
    }

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    state_id = None

    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)

        # Handle arrow keys and special inputs
        if key == "\x1b":
            key += sys.stdin.read(2)
        elif key in ["x", "z"]:
            print('\rPlease enter the state index (0-9): ', end='', flush=True)
            state_id = int(sys.stdin.read(1))

        time.sleep(0.1)
    except Exception:
        return -1, None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    action_index = key_mapping.get(key)
    if action_index is None:
        print(f"\rUnknown key: {key}\r")
        return -1, None

    return action_index, state_id


def main(argv):
    """
    Main function to run the interactive environment.
    """
    actions = {
        0: "DOWN",
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "A",
        5: "B",
    }

    if len(argv) < 2:
        print("Usage: interactive.py <domain> <task>")
        sys.exit(1)

    states_dir = os.path.expanduser("~/cares_rl_configs/pokemon/interactive_states")
    env = Suite.make(argv[0], argv[1], 24, headless=False, discrete=True)

    # Perform an initial step to load the screen
    env.step(3)

    while True:
        action_index, state_index = wait_for_input()

        if action_index == -1:
            # Ignore unrecognized inputs
            continue

        if action_index == 6 and state_index is not None:  # Load state
            state_file = f"{state_index}.state"
            print(f"\rLoading state: {state_file}\r")

            try:
                with open(os.path.join(states_dir, state_file), "rb") as f:
                    env.pyboy.load_state(f)
                    env.pyboy.tick(4)
            except (PyBoyInvalidInputException, FileNotFoundError):
                print(f"{state_file} could not be found!\r")

        elif action_index == 7 and state_index is not None:  # Save state
            state_file = f"{state_index}.state"
            state_file_path = os.path.join(states_dir, state_file)

            print(f"\rSaving state: {state_file_path}\r")

            try:
                os.makedirs(states_dir, exist_ok=True)  # Ensure directory exists
                with open(state_file_path, "wb") as f:
                    env.pyboy.save_state(f)
            except (PyBoyInvalidInputException, IOError) as e:
                print(f"\rError saving state: {e}\r")

        else:
            # Perform an action in the environment
            _, reward, _, _ = env.step(action_index)
            print(f"\rAction: {actions[action_index]:5} | Reward: {reward}\r")


if __name__ == "__main__":
    main(sys.argv[1:])
