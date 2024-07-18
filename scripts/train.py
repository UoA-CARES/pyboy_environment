"""
This script is used to train reinforcement learning agents in pyboy.
The main function parses command-line arguments, creates the environment, network, 
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import torch
import policy_loop as pbe
import yaml

from pyboy_config import PyboyEnvironmentConfig
from training_config import TrainingConfig

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, Record, RLParser
from cares_reinforcement_learning.util import helpers as hlp

from pyboy_environment.suite import create_environment

logging.basicConfig(level=logging.INFO)


def main():
    """
    The main function that orchestrates the training process.
    """
    parser = RLParser()

    parser.add_configuration("env_config", PyboyEnvironmentConfig)
    parser.add_configuration("training_config", TrainingConfig)

    configurations = parser.parse_args()

    env_config = configurations["env_config"]
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]

    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

    logging.info(
        "\n---------------------------------------------------\n"
        "PYBOY CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(env_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "ALGORITHM CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(alg_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "TRAINING CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(training_config), default_flow_style=False)}")
    logging.info(
        f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    glob_log_dir = os.environ.get("CARES_LOG_DIR", f"{Path.home()}/cares_rl_logs")

    domain = env_config.domain
    task = env_config.task
    run_name_input = input(
        "Double check your experiment configurations and give a name for this run (or just press ENTER)\n"
    )

    run_name = (
        run_name_input
        if run_name_input != ""
        else env_config.run_name if env_config.run_name != "" else "unnamed_run"
    )

    run_name_with_time = f"{run_name}_{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}"

    log_dir = f"{domain}/{task}/{run_name_with_time}"

    if not torch.cuda.is_available():
        no_gpu_answer = input(
            "No cuda detected. Do you still want to continue? Note: Training will be slow. [y/n]"
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info("Terminating Experiment - check CUDA is installed.")
            sys.exit()

    # This line should be here for seed consistency issues
    env = create_environment(env_config, bool(alg_config.image_observation))

    hlp.set_seed(training_config.seed)
    env.set_seed(training_config.seed)

    logging.info(f"Algorithm: {alg_config.algorithm}")
    agent = network_factory.create_network(
        env.observation_space, env.action_num, alg_config
    )

    if agent is None:
        raise ValueError(f"Unknown agent for default algorithms {alg_config.algorithm}")

    memory = memory_factory.create_memory(alg_config)

    # create the record class - standardised results tracking

    record = Record(
        glob_log_dir=glob_log_dir,
        log_dir=log_dir,
        algorithm=alg_config.algorithm,
        task=env_config.task,
        network=agent,
        plot_frequency=training_config.plot_frequency,
        checkpoint_frequency=training_config.checkpoint_frequency,
    )

    record.save_config(env_config, "env_config")
    record.save_config(training_config, "train_config")
    record.save_config(alg_config, "alg_config")

    pbe.policy_based_train(
        env,
        agent,
        memory,
        record,
        training_config,
        alg_config,
        display=env_config.display,
    )

    record.save()


if __name__ == "__main__":
    main()
