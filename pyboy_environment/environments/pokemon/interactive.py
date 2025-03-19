import os
import sys
import termios
import tty
import readline  # DO NOT DELETE - input() needs readline to work properly

from pyboy.utils import WindowEvent, PyBoyInvalidInputException
from pyboy_environment.environments.pyboy_environment import PyboyEnvironment
import pyboy_environment.suite as Suite


def get_action_key() -> str | any:
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
        "\x1b[A": [WindowEvent.PRESS_ARROW_UP, "UP"],
        "\x1b[B": [WindowEvent.PRESS_ARROW_DOWN, "DOWN"],
        "\x1b[C": [WindowEvent.PRESS_ARROW_RIGHT, "RIGHT"],
        "\x1b[D": [WindowEvent.PRESS_ARROW_LEFT, "LEFT"],
        "a": [WindowEvent.PRESS_BUTTON_A, "A"],
        "b": [WindowEvent.PRESS_BUTTON_B, "B"],
        "\b": [WindowEvent.PRESS_BUTTON_SELECT, "SELECT"],
        "\x7f": [WindowEvent.PRESS_BUTTON_SELECT, "SELECT"],
        "\r": [WindowEvent.PRESS_BUTTON_START, "START"],
        "\n": [WindowEvent.PRESS_BUTTON_START, "START"],
    }

    if len(argv) < 2:
        print("Usage: interactive.py <domain> <task>")
        sys.exit(1)

    # Set up directory for saving/loading states
    states_dir = os.path.expanduser(f"~/cares_rl_configs/{argv[0]}/interactive_states")
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Set up environment
    env = Suite.make(argv[0], argv[1], 24, headless=False, discrete=True)
    env.step(env.valid_actions.index(WindowEvent.PRESS_BUTTON_A))

    print("\rEnvironment ready, waiting for user input (Press 'q' to quit)...\r")
    while True:
        key = get_action_key()

        if key in key_mapping.keys():
            action_event, action_name = key_mapping[key]

            if action_event not in env.valid_actions:
                print(
                    f"Failed to execute action: {action_name}. Valid PyBoy action received but is not a valid environment action\r"
                )
                continue

            action_index = env.valid_actions.index(action_event)
            _, reward, _, _ = env.step(action_index)
            print(f"Action: {action_name:5} | Reward: {reward}\r")
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
