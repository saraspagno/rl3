"""
Actor-Critic (A2C) package.
"""

from .actor_critic import train_actor_critic, TrainedActorCriticAgent

__all__ = [
    'train_actor_critic',
    'TrainedActorCriticAgent',
]
