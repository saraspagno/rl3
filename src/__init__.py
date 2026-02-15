"""
Code package for Deep RL MiniGrid project.

This package contains:
- Environments (SimpleGridEnv, KeyDoorBallEnv)
- Preprocessing utilities
- Neural networks (CNNFeatureExtractor, PolicyNetwork)
- REINFORCE algorithm
- Shared utils (plotting, evaluation, video recording)
"""

from .envs import SimpleGridEnv, KeyDoorBallEnv
from .preprocessing import (
    pre_process,
    preprocess_grayscale,
    preprocess_resize,
    preprocess_crop_and_resize,
    preprocess_normalize,
    get_preprocessed_observation_space,
    visualize_preprocessing,
    visualize_all_preprocessing_methods,
)
from .networks import (
    obs_to_tensor,
    CNNFeatureExtractor,
    PolicyNetwork,
)
from .reinforce import (
    train_reinforce,
    compute_returns,
    TrainedAgent,
)
from .utils import (
    plot_training,
    plot_training_history,   # alias kept for backwards compat
    evaluate_agent,
    print_evaluation_results,
    record_video,
)

__all__ = [
    # Environments
    'SimpleGridEnv',
    'KeyDoorBallEnv',
    # Preprocessing
    'pre_process',
    'preprocess_grayscale',
    'preprocess_resize',
    'preprocess_crop_and_resize',
    'preprocess_normalize',
    'get_preprocessed_observation_space',
    'visualize_preprocessing',
    'visualize_all_preprocessing_methods',
    # Networks
    'obs_to_tensor',
    'CNNFeatureExtractor',
    'PolicyNetwork',
    # REINFORCE
    'train_reinforce',
    'compute_returns',
    'TrainedAgent',
    # Utils
    'plot_training',
    'plot_training_history',
    'evaluate_agent',
    'print_evaluation_results',
    'record_video',
]
