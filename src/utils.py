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

def plot_training(history, title="Training", save_path=None):
    """
    Unified 2×2 training-curve plot for any algorithm (DQN, A2C, REINFORCE…).

    Top row (always present):
        - Episode rewards (smoothed)
        - Episode steps   (smoothed)

    Bottom row (algorithm-specific, panels hidden when data is absent):
        - Left:  loss curve(s)
              • DQN  → ``losses``
              • A2C  → ``actor_losses`` + ``critic_losses`` overlaid
        - Right: exploration / regularisation schedule
              • DQN  → ``epsilons``
              • A2C  → ``avg_entropy``

    Args:
        history: dict returned by a training function.  Must contain at least
                 ``episode_rewards`` and ``episode_steps``.
        title:     suptitle for the figure
        save_path: if given, save PNG and close; otherwise plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = min(100, max(1, len(history["episode_rewards"]) // 10))
    kernel = np.ones(window) / window if window > 0 else None

    # -- helper ----------------------------------------------------------
    def _smooth(ax, data, color, ylabel, subtitle, xlabel="Episode"):
        ax.plot(data, alpha=0.2, color=color)
        if kernel is not None and len(data) >= window:
            sm = np.convolve(data, kernel, mode="valid")
            ax.plot(range(window - 1, len(data)), sm,
                    color=color, label=f"{window}-ep avg")
            ax.legend()
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {subtitle}"); ax.grid(True, alpha=0.3)

    # ── top row: rewards + steps (always) ──────────────────────────────
    _smooth(axes[0, 0], history["episode_rewards"],
            "tab:blue", "Reward", "Rewards")
    _smooth(axes[0, 1], history["episode_steps"],
            "tab:orange", "Steps", "Steps / Episode")

    # ── bottom-left: loss ──────────────────────────────────────────────
    if history.get("losses"):
        # DQN-style single loss
        _smooth(axes[1, 0], history["losses"],
                "tab:red", "Loss", "Huber Loss")
    elif history.get("actor_losses"):
        # A2C-style actor + critic losses
        ax = axes[1, 0]
        ax.plot(history["actor_losses"], alpha=0.15, color="tab:red")
        ax.plot(history["critic_losses"], alpha=0.15, color="tab:purple")
        if kernel is not None and len(history["actor_losses"]) >= window:
            sm_a = np.convolve(history["actor_losses"], kernel, mode="valid")
            sm_c = np.convolve(history["critic_losses"], kernel, mode="valid")
            x = range(window - 1, len(history["actor_losses"]))
            ax.plot(x, sm_a, color="tab:red",    label=f"Actor  ({window}-avg)")
            ax.plot(x, sm_c, color="tab:purple",  label=f"Critic ({window}-avg)")
        ax.set_xlabel("Update step"); ax.set_ylabel("Loss")
        ax.set_title(f"{title} — Losses"); ax.legend(); ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].set_visible(False)

    # ── bottom-right: epsilon or entropy ──────────────────────────────
    if history.get("epsilons"):
        ax = axes[1, 1]
        ax.plot(history["epsilons"], color="tab:green")
        ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
        ax.set_title(f"{title} — Epsilon Schedule"); ax.grid(True, alpha=0.3)
    elif history.get("avg_entropy"):
        ax = axes[1, 1]
        ax.plot(history["avg_entropy"], color="tab:green")
        ax.set_xlabel("Update step"); ax.set_ylabel("Entropy")
        ax.set_title(f"{title} — Policy Entropy"); ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to: {save_path}")
        plt.close()
    else:
        plt.show()


# Keep the old name as an alias so existing imports don't break
plot_training_history = plot_training


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env_class=None, num_episodes=100, max_steps=200,
                   tile_size=10, env=None):
    """
    Evaluate a trained agent over multiple episodes.

    The agent must expose ``select_action(obs) -> int``.

    Args:
        agent: trained agent (or None for random baseline)
        env_class: environment class (SimpleGridEnv / KeyDoorBallEnv).
                   Ignored if *env* is provided.
        num_episodes: number of evaluation episodes
        max_steps: max steps per episode
        tile_size: tile size for rendering (smaller = faster)
        env: pre-built environment instance (use when you need wrappers).
             Takes precedence over env_class. The env is reset but NOT
             closed at the end so the caller can reuse it.

    Returns:
        dict with keys: num_episodes, avg_reward, std_reward, avg_steps,
        std_steps, success_rate, min_steps, max_steps, all_rewards, all_steps
    """
    close_env = False
    if env is None:
        if env_class is None:
            raise ValueError("Either env_class or env must be provided.")
        env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=tile_size)
        close_env = True

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
    if close_env:
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

def record_video(agent, env_class=None, filename="agent.mp4", num_episodes=1,
                  max_steps=200, fps=10, env=None):
    """
    Record an agent playing and save as MP4 or GIF.

    The format is auto-detected from the filename extension:
        - ``.mp4`` — requires ffmpeg (pre-installed on Colab)
        - ``.gif`` — works everywhere via imageio

    Args:
        agent: trained agent with select_action(obs), or None for random
        env_class: environment class. Ignored if *env* is provided.
        filename: output path (e.g. "outputs/agent.mp4")
        num_episodes: episodes to record
        max_steps: max steps per episode
        fps: frames per second
        env: pre-built environment instance (use when you need wrappers).
             Takes precedence over env_class.

    Returns:
        (total_reward, total_steps)
    """
    close_env = False
    if env is None:
        if env_class is None:
            raise ValueError("Either env_class or env must be provided.")
        # Use full tile_size=32 for crisp video frames
        env = env_class(preprocess=pre_process, max_steps=max_steps, tile_size=32)
        close_env = True

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

    if close_env:
        env.close()

    if not frames:
        print("Warning: no frames captured, skipping video save.")
        return total_reward, total_steps

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if filename.endswith(".mp4"):
        writer = imageio.get_writer(filename, fps=fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
    else:
        duration_ms = 1000.0 / fps
        imageio.mimsave(filename, frames, duration=duration_ms, loop=0)

    print(f"Video saved: {filename}  ({len(frames)} frames, {fps} fps)")
    print(f"  Episodes: {num_episodes}, Total steps: {total_steps}, Total reward: {total_reward:.2f}")

    return total_reward, total_steps
