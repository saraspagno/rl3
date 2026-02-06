"""
REINFORCE (Monte-Carlo Policy Gradient) algorithm.

Algorithm:
    1. Collect full episode trajectories using the current policy
    2. Compute discounted returns G_t for each timestep
    3. Subtract a baseline and normalise advantages
    4. Update policy: theta <- theta + alpha * grad(log pi(a|s)) * A_t
    5. Add entropy bonus to encourage exploration

This implementation uses *batched* updates: it accumulates several episodes
before performing a single gradient step, which reduces variance.
"""

import numpy as np
import torch
import torch.optim as optim

from ..networks import obs_to_tensor, PolicyNetwork
from ..preprocessing import pre_process


def compute_returns(rewards, gamma):
    """
    Compute discounted returns for a single episode.

    Args:
        rewards: list of float — rewards collected during the episode
        gamma: float — discount factor

    Returns:
        list of float — discounted return G_t for each timestep
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()  # O(n) instead of O(n²) from insert(0, ...)
    return returns


def train_reinforce(
    env_class,
    num_episodes=1000,
    max_steps=200,
    lr=0.001,
    gamma=0.99,
    entropy_coef=0.01,
    episodes_per_update=10,
    max_grad_norm=0.5,
    print_every=100,
    device=None,
    tile_size=10,
):
    """
    Train a REINFORCE agent with batched episode updates.

    Args:
        env_class: Environment class (SimpleGridEnv or KeyDoorBallEnv)
        num_episodes: Total number of training episodes
        max_steps: Maximum steps per episode
        lr: Learning rate for Adam optimiser
        gamma: Discount factor
        entropy_coef: Weight for the entropy bonus (higher = more exploration)
        episodes_per_update: Number of episodes to collect before each gradient step
        max_grad_norm: Maximum gradient norm for clipping
        print_every: Print progress every N episodes
        device: torch device (defaults to cuda if available, else cpu)
        tile_size: Tile size for env rendering (smaller = faster; default 10 for
                   training since we resize to 84x84 anyway)

    Returns:
        agent: TrainedAgent wrapper with a select_action(obs) method
        history: dict with keys 'episode_rewards', 'episode_steps', 'running_reward'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- environment & network ----
    env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=tile_size)
    num_actions = env.action_space.n

    policy = PolicyNetwork(num_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ---- bookkeeping ----
    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "running_reward": 0.0,
        "avg_entropy": [],
    }

    # ---- batch storage ----
    batch_log_probs = []
    batch_returns = []
    batch_entropies = []

    print(f"Training REINFORCE on {env_class.__name__}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"Learning rate: {lr}, Gamma: {gamma}")
    print(f"Episodes per update: {episodes_per_update}")
    print("=" * 60)

    for episode in range(num_episodes):
        # ---------- collect one episode ----------
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_log_probs = []
        episode_rewards = []
        episode_entropies = []

        for step in range(max_steps):
            obs_tensor = obs_to_tensor(obs, device)
            logits = policy(obs_tensor)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            episode_log_probs.append(dist.log_prob(action))
            episode_entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_rewards.append(reward)
            episode_reward += reward

            if terminated or truncated:
                break

        # ---------- compute returns ----------
        returns = compute_returns(episode_rewards, gamma)

        batch_log_probs.extend(episode_log_probs)
        batch_returns.extend(returns)
        batch_entropies.extend(episode_entropies)

        # ---------- record history ----------
        history["episode_rewards"].append(episode_reward)
        history["episode_steps"].append(step + 1)

        if history["running_reward"] == 0.0:
            history["running_reward"] = episode_reward
        else:
            history["running_reward"] = (
                0.95 * history["running_reward"] + 0.05 * episode_reward
            )

        # ---------- update policy every N episodes ----------
        if (episode + 1) % episodes_per_update == 0 and len(batch_log_probs) > 0:
            returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
            log_probs_t = torch.stack(batch_log_probs)
            entropies_t = torch.stack(batch_entropies)

            # Advantages = returns with baseline subtraction and normalization
            advantages = returns_t
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # REINFORCE loss with entropy regularisation
            loss = -(log_probs_t * advantages).mean() - entropy_coef * entropies_t.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            # track entropy for debugging
            history["avg_entropy"].append(entropies_t.mean().item())

            # clear batch
            batch_log_probs = []
            batch_returns = []
            batch_entropies = []

        # ---------- logging ----------
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(history["episode_rewards"][-print_every:])
            avg_steps = np.mean(history["episode_steps"][-print_every:])
            print(
                f"Episode {episode + 1:5d} | "
                f"Avg Reward: {avg_reward:8.3f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"Running Reward: {history['running_reward']:8.3f}"
            )

    print("=" * 60)
    print(f"Training complete!")
    recent = min(print_every, len(history["episode_rewards"]))
    print(
        f"Last {recent} episodes — "
        f"Avg Reward: {np.mean(history['episode_rewards'][-recent:]):.3f}, "
        f"Avg Steps: {np.mean(history['episode_steps'][-recent:]):.1f}"
    )

    env.close()

    # wrap policy for easy evaluation
    agent = TrainedAgent(policy, device)
    return agent, history


class TrainedAgent:
    """
    Thin wrapper around a trained PolicyNetwork for evaluation.

    Usage:
        action = agent.select_action(obs)          # greedy (argmax)
        action = agent.select_action(obs, greedy=False)  # stochastic (sample)
    """

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def select_action(self, obs, greedy=True):
        """
        Pick an action for a single observation.

        Args:
            obs: numpy array (84, 84, 1)
            greedy: if True, take argmax; if False, sample from distribution

        Returns:
            action: int
        """
        obs_tensor = obs_to_tensor(obs, self.device)
        logits = self.policy(obs_tensor)

        if greedy:
            return logits.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample().item()
