"""
Advantage Actor-Critic (A2C) algorithm.

Key differences from REINFORCE:
    - Uses a learned value function V(s) as baseline (the "critic")
    - Updates on every step using TD advantage: A(s,a) = r + γV(s') - V(s)
    - n-step returns for lower bias: A = Σ γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t)
    - Much lower variance → learns faster, especially with sparse rewards

The actor and critic share a CNN backbone for sample efficiency.
"""

import numpy as np
import torch
import torch.optim as optim

from ..networks import ActorCriticNetwork, obs_to_tensor
from ..preprocessing import pre_process


def train_actor_critic(
    env_class=None,
    num_episodes=10000,
    max_steps=200,
    lr=0.0007,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    n_steps=5,
    max_grad_norm=0.5,
    print_every=100,
    device=None,
    tile_size=10,
    env=None,
    video_at=None,
):
    """
    Train an Advantage Actor-Critic (A2C) agent.

    Uses n-step returns for advantage estimation:
        A_t = (Σ_{k=0}^{n-1} γ^k r_{t+k}) + γ^n V(s_{t+n}) - V(s_t)

    Args:
        env_class: Environment class (SimpleGridEnv or KeyDoorBallEnv).
                   Ignored if *env* is provided.
        num_episodes: Total training episodes
        max_steps: Max steps per episode
        lr: Learning rate for Adam
        gamma: Discount factor
        entropy_coef: Entropy bonus weight (exploration)
        value_coef: Critic loss weight (relative to actor loss)
        n_steps: Number of steps for n-step returns
        max_grad_norm: Gradient clipping norm
        print_every: Logging frequency
        device: torch device
        tile_size: Tile size for env rendering (smaller = faster)
        env: Pre-built environment instance (use when you need wrappers
             like SmartRewardWrapper). Takes precedence over env_class.

    Returns:
        agent: TrainedActorCriticAgent wrapper
        history: dict with episode_rewards, episode_steps, avg_entropy,
                 actor_losses, critic_losses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- environment & network ----
    if env is None:
        if env_class is None:
            raise ValueError("Either env_class or env must be provided.")
        env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=tile_size)
    num_actions = env.action_space.n

    model = ActorCriticNetwork(num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- bookkeeping ----
    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "running_reward": 0.0,
        "avg_entropy": [],
        "actor_losses": [],
        "critic_losses": [],
    }

    env_name = env_class.__name__ if env_class is not None else type(env).__name__
    print(f"Training A2C on {env_name}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}, n_steps: {n_steps}")
    print(f"LR: {lr}, Gamma: {gamma}, Entropy coef: {entropy_coef}")
    print("=" * 60)

    update_count = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0

        # n-step buffers (collected within the episode, flushed every n_steps)
        step_log_probs = []
        step_values = []
        step_rewards = []
        step_entropies = []

        for step in range(max_steps):
            action, log_prob, entropy, value = model.act(obs, device)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            step_log_probs.append(log_prob)
            step_values.append(value)
            step_rewards.append(reward)
            step_entropies.append(entropy)

            done = terminated or truncated

            # ---- update every n_steps or at episode end ----
            if len(step_rewards) == n_steps or done:
                # Bootstrap value for non-terminal states
                if done:
                    R = 0.0
                else:
                    R = model.get_value(next_obs, device)

                # Compute n-step returns (backward through buffer)
                returns = []
                for r in reversed(step_rewards):
                    R = r + gamma * R
                    returns.append(R)
                returns.reverse()

                returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
                log_probs_t = torch.stack(step_log_probs)
                values_t = torch.stack(step_values).squeeze(-1)
                entropies_t = torch.stack(step_entropies)

                # Advantage = returns - V(s)  (no grad through returns)
                advantages = returns_t - values_t.detach()
                # Normalise so actor gradient isn't dwarfed by entropy term
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Losses
                actor_loss = -(log_probs_t * advantages).mean()
                critic_loss = (returns_t.detach() - values_t).pow(2).mean()
                entropy_bonus = entropies_t.mean()

                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                update_count += 1
                history["actor_losses"].append(actor_loss.item())
                history["critic_losses"].append(critic_loss.item())
                history["avg_entropy"].append(entropy_bonus.item())

                # Clear buffers
                step_log_probs = []
                step_values = []
                step_rewards = []
                step_entropies = []

            obs = next_obs
            if done:
                break

        # ---- record episode stats ----
        history["episode_rewards"].append(episode_reward)
        history["episode_steps"].append(step + 1)

        if history["running_reward"] == 0.0:
            history["running_reward"] = episode_reward
        else:
            history["running_reward"] = (
                0.95 * history["running_reward"] + 0.05 * episode_reward
            )

        # ---- mid-training video snapshot ----
        if video_at and (episode + 1) == video_at["episode"]:
            from ..utils import record_video
            print(f"\n>>> Recording mid-training video at episode {episode + 1} ...")
            mid_agent = TrainedActorCriticAgent(model, device)
            record_video(mid_agent, env_class=env_class,
                         filename=video_at["filename"],
                         num_episodes=1, max_steps=max_steps)
            print()

        # ---- logging ----
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(history["episode_rewards"][-print_every:])
            avg_steps = np.mean(history["episode_steps"][-print_every:])
            recent_entropy = (
                np.mean(history["avg_entropy"][-50:])
                if history["avg_entropy"] else 0.0
            )
            print(
                f"Episode {episode + 1:5d} | "
                f"Avg Reward: {avg_reward:8.3f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"Entropy: {recent_entropy:.3f} | "
                f"Running: {history['running_reward']:8.3f}"
            )

    print("=" * 60)
    print("Training complete!")
    recent = min(print_every, len(history["episode_rewards"]))
    print(
        f"Last {recent} episodes — "
        f"Avg Reward: {np.mean(history['episode_rewards'][-recent:]):.3f}, "
        f"Avg Steps: {np.mean(history['episode_steps'][-recent:]):.1f}"
    )

    env.close()
    agent = TrainedActorCriticAgent(model, device)
    return agent, history


class TrainedActorCriticAgent:
    """
    Wrapper around a trained ActorCriticNetwork for evaluation.

    Usage:
        action = agent.select_action(obs)          # greedy (argmax)
        action = agent.select_action(obs, greedy=False)  # stochastic
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def select_action(self, obs, greedy=True):
        obs_t = obs_to_tensor(obs, self.device)
        logits, _ = self.model(obs_t)

        if greedy:
            return logits.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample().item()
