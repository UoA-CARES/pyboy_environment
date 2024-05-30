import logging

import cv2

from pyboy_environment.environments.mario.mario_run import MarioRun

logging.basicConfig(level=logging.DEBUG)


def key_to_action(button: int):
    key_map = {
        97: 0,  # a - left
        100: 1,  # d - right
        122: 2,  # z - A
        120: 3,  # x - B
    }
    logging.info(f"Key: {button}")
    if button in key_map:
        logging.info(f"Map: {key_map[button]}")
        return key_map[button]
    return -1


if __name__ == "__main__":
    env = MarioRun(act_freq=6)

    print(f"{env.action_num}")
    state = env.reset()
    image = env.grab_frame()

    while True:
        cv2.imshow("State", image)
        key = cv2.waitKey(0)

        action = [0, 0, 0, 0]
        action[key_to_action(key)] = 1

        state, reward, done, truncated = env.step(action)

        stats = env._generate_game_stats()
        logging.info(f"Stats: {stats}")

        reward = env._calculate_reward(stats)
        logging.info(f"Reward: {reward}")

        score = env._read_m(0x9820)
        logging.info(f"Score: {score}")

        coins = env._get_coins()
        logging.info(f"Coins: {coins}")

        if done or truncated:
            state = env.reset()

        image = env.grab_frame()
        game_area = env.game_area()
