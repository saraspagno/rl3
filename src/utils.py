"""
Shared utilities for training, evaluation, plotting, and video recording.

All functions are algorithm-agnostic. They depend only on the agent having
a ``select_action(obs)`` method that returns an int action.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

from .preprocessing import pre_process


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot reward and step curves from a training history dict.

    Expects *history* to contain at least:
        - episode_rewards : list[float]
        - episode_steps   : list[int]

    Optionally plots entropy if history contains:
        - avg_entropy : list[float]

    Args:
        history: dict returned by a training function
        title: plot super-title
        save_path: if provided, save the figure to this path instead of plt.show()
    """
    has_entropy = "avg_entropy" in history and len(history["avg_entropy"]) > 0
    ncols = 3 if has_entropy else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    window = min(100, max(1, len(history["episode_rewards"]) // 10))

    # ---- rewards ----
    ax = axes[0]
    ax.plot(history["episode_rewards"], alpha=0.3, label="Per episode")
    if len(history["episode_rewards"]) >= window:
        smoothed = np.convolve(
            history["episode_rewards"],
            np.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            range(window - 1, len(history["episode_rewards"])),
            smoothed,
            label=f"{window}-ep avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{title} — Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- steps ----
    ax = axes[1]
    ax.plot(history["episode_steps"], alpha=0.3, label="Per episode")
    if len(history["episode_steps"]) >= window:
        smoothed = np.convolve(
            history["episode_steps"],
            np.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            range(window - 1, len(history["episode_steps"])),
            smoothed,
            label=f"{window}-ep avg",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(f"{title} — Steps per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- entropy (optional) ----
    if has_entropy:
        ax = axes[2]
        ax.plot(history["avg_entropy"])
        ax.set_xlabel("Update step")
        ax.set_ylabel("Entropy")
        ax.set_title(f"{title} — Policy Entropy")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to: {save_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env_class, num_episodes=100, max_steps=200,
                   tile_size=10):
    """
    Evaluate a trained agent over multiple episodes.

    The agent must expose ``select_action(obs) -> int``.

    Args:
        agent: trained agent (or None for random baseline)
        env_class: environment class (SimpleGridEnv / KeyDoorBallEnv)
        num_episodes: number of evaluation episodes
        max_steps: max steps per episode
        tile_size: tile size for rendering (smaller = faster)

    Returns:
        dict with keys: num_episodes, avg_reward, std_reward, avg_steps,
        std_steps, success_rate, min_steps, max_steps, all_rewards, all_steps
    """
    env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=tile_size)

    episode_rewards = []
    episode_steps = []
    successes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated:
                successes += 1
                break
            if truncated:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)

        # Progress bar
        done = ep + 1
        pct = done / num_episodes
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\rEvaluating: |{bar}| {done}/{num_episodes} "
            f"({pct*100:.0f}%) success so far: {successes}/{done}",
            end="",
            flush=True,
        )

    print()  # newline after progress bar
    env.close()

    return {
        "num_episodes": num_episodes,
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_steps": np.mean(episode_steps),
        "std_steps": np.std(episode_steps),
        "success_rate": successes / num_episodes,
        "min_steps": int(np.min(episode_steps)),
        "max_steps": int(np.max(episode_steps)),
        "all_rewards": episode_rewards,
        "all_steps": episode_steps,
    }


def print_evaluation_results(results, title="Evaluation Results"):
    """Print formatted evaluation metrics."""
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)
    print(f"  Episodes evaluated:  {results['num_episodes']}")
    print(f"  Success rate:        {results['success_rate']*100:.1f}%")
    print("-" * 60)
    print(f"  Average reward:      {results['avg_reward']:.3f} +/- {results['std_reward']:.3f}")
    print(f"  Average steps:       {results['avg_steps']:.1f} +/- {results['std_steps']:.1f}")
    print(f"  Step range:          [{results['min_steps']}, {results['max_steps']}]")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_video(agent, env_class, filename, num_episodes=1, max_steps=200,
                  fps=10):
    """
    Record an agent playing and save as GIF.

    Args:
        agent: trained agent with select_action(obs), or None for random
        env_class: environment class
        filename: output GIF path (e.g. "outputs/agent.gif")
        num_episodes: episodes to record
        max_steps: max steps per episode
        fps: frames per second for the GIF

    Returns:
        (total_reward, total_steps)
    """
    # Use full tile_size=32 for crisp video frames
    env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=32)

    frames = []
    total_reward = 0.0
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))

        for step in range(max_steps):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.array(frame))
            total_reward += reward
            total_steps += 1

            if terminated or truncated:
                break

    env.close()

    if not frames:
        print("Warning: no frames captured, skipping video save.")
        return total_reward, total_steps

    # duration in ms per frame (imageio v2.9+ uses duration instead of fps)
    duration_ms = 1000.0 / fps

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimsave(filename, frames, duration=duration_ms, loop=0)
    print(f"Video saved: {filename}  ({len(frames)} frames, {fps} fps)")
    print(f"  Episodes: {num_episodes}, Total steps: {total_steps}, Total reward: {total_reward:.2f}")

    return total_reward, total_steps
