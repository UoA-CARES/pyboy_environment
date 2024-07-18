
from typing import List, Optional

from cares_reinforcement_learning.util.configurations import SubscriptableClass

class TrainingConfig(SubscriptableClass):
    """
    Configuration class for training.

    Attributes:
        seeds (List[int]): List of random seeds for reproducibility. Default is [10].
        plot_frequency (Optional[int]): Frequency at which to plot training progress. Default is 100.
        checkpoint_frequency (Optional[int]): Frequency at which to save model checkpoints. Default is 100.
        number_steps_per_evaluation (Optional[int]): Number of steps per evaluation. Default is 10000.
        number_eval_episodes (Optional[int]): Number of episodes to evaluate during training. Default is 10.
    """

    seed: int = 10
    plot_frequency: Optional[int] = 100
    checkpoint_frequency: Optional[int] = 100
    number_steps_per_evaluation: Optional[int] = 10000
    number_eval_episodes: Optional[int] = 10