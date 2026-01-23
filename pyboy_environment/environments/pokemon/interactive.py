import os
import sys
import termios
import tty
import readline  # DO NOT DELETE - input() needs readline to work properly

from pyboy.utils import WindowEvent, PyBoyInvalidInputException
from pyboy_environment.environments.pyboy_environment import PyboyEnvironment
import pyboy_environment.suite as Suite


def get_action_key() -> str:
    """Captures a single keypress from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)

        if key == "\x1b":
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def manage_state(name: str, dir: os.PathLike, mode: str, env: PyboyEnvironment):
    """Handles saving and loading of PyBoy state."""
    if not name.endswith(".state"):
        name = f"{name}.state"
    state_path = os.path.join(dir, name)

    try:
        with open(state_path, mode) as f:
            if mode == "rb":
                env.pyboy.load_state(f)
                env.pyboy.tick(4)
                print(f"\rLoaded state: {name}\r")
            elif mode == "wb":
                env.pyboy.save_state(f)
                print(f"\rSaved state as: {name}\r")
    except (PyBoyInvalidInputException, IOError) as e:
        print(f"\rError {'loading' if mode == 'rb' else 'saving'} state: {e}\r")


def main(argv: list[str]):
    key_mapping = {
        "\x1b[A": ["up"],
        "8": ["up"],
        "\x1b[B": ["down"],
        "2": ["down"],
        "\x1b[C": ["right"],
        "6": ["right"],
        "\x1b[D": ["left"],
        "4": ["left"],
        "a": ["a"],
        "5": ["a"],
        "s": ["b"],
        "+": ["b"],
        "\b": ["select"],
        "\x7f": ["select"],
        "\r": ["start"],
        "\n": ["start"],
        "7": ["left", "a"],
        "9": ["right", "a"],
        "1": ["left", "b"],
        "3": ["right", "b"],
    }

    if len(argv) < 2:
        print("Usage: interactive.py <domain> <task>")
        sys.exit(1)

    # Set up directory for saving/loading states
    states_dir = os.path.expanduser(f"~/cares_rl_configs/{argv[0]}/interactive_states")
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Set up environment
    env = Suite.make(argv[0], argv[1], 24, headless=False, emulation_speed=1, image_observation=True)

    print("\rEnvironment ready, waiting for user input (Press 'q' to quit)...\r")
    while True:
        key = get_action_key()

        if key in key_mapping.keys():
            action = key_mapping[key]
            action = action[0]

            if action not in env.actions:
                print(
                    f"Failed to execute action: {action}. Valid PyBoy action received but is not a valid environment action\r"
                )
                continue

            _, reward, _, _, _ = env.step(env.actions.index(action))
            print(f"Action: {action} | Reward: {reward}\r")
        elif key in ("x", "z"):
            name = input("Enter name: ")
            mode = "rb" if key == "x" else "wb"
            manage_state(name, states_dir, mode, env)
        elif key == "q":
            print("Exiting...")
            break
        else:
            print(f"Unknown input: {key}\r")


if __name__ == "__main__":
    main(sys.argv[1:])
