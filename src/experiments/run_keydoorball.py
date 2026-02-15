"""
Experiment: Actor-Critic (A2C) on KeyDoorBallEnv

Run this script to:
    1. Train an A2C agent on KeyDoorBallEnv with SmartRewardWrapper
    2. Plot training curves (saved to outputs/)
    3. Evaluate over 100 episodes (printed + saved)
    4. Record a video of the trained agent (saved to outputs/)

Usage:
    python -m src.experiments.run_keydoorball          (from project root)
"""

import os
import sys
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work when
# executed with  `python -m src.experiments.run_keydoorball`  from the
# project root, or  `python run_keydoorball.py`  from this directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.envs import KeyDoorBallEnv
from src.reward_shaping import SmartRewardWrapper
from src.preprocessing import pre_process
from src.actor_critic import train_actor_critic
from src.utils import (
    plot_training,
    evaluate_agent,
    print_evaluation_results,
    record_video,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

A2C_KEYDOOR_CONFIG = {
    "num_episodes": 10000,       # A2C is on-policy & sample-hungry; needs many episodes
    "max_steps": 300,            # KeyDoorBall needs more steps (4-step sequence)
    "lr": 0.0005,               # moderate LR
    "gamma": 0.99,              # longer horizon — rewards come late
    "entropy_coef": 0.15,       # was 0.08 — aggressive exploration for sequential task
    "value_coef": 0.5,          # critic loss weight
    "n_steps": 10,              # faster updates
    "max_grad_norm": 0.5,
    "print_every": 200,         # log every 200 eps
    "tile_size": 10,            # small for fast training
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_keydoor_env(max_steps, tile_size):
    """Create a KeyDoorBallEnv wrapped with SmartRewardWrapper."""
    env = KeyDoorBallEnv(
        preprocess=pre_process,
        max_steps=max_steps,
        tile_size=tile_size,
    )
    return SmartRewardWrapper(env)


def make_keydoor_env_video(max_steps):
    """Create a KeyDoorBallEnv wrapped with SmartRewardWrapper (tile_size=32 for video)."""
    env = KeyDoorBallEnv(
        preprocess=pre_process,
        max_steps=max_steps,
        tile_size=32,
    )
    return SmartRewardWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print(" A2C on KeyDoorBallEnv (with SmartRewardWrapper)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")
    print(f"Config: {A2C_KEYDOOR_CONFIG}")
    print()

    # ---- reproducibility ----
    set_seed(SEED)

    # ---- build the wrapped environment ----
    train_env = make_keydoor_env(
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
        tile_size=A2C_KEYDOOR_CONFIG["tile_size"],
    )

    # ---- train ----
    agent, history = train_actor_critic(
        env=train_env,
        device=device,
        num_episodes=A2C_KEYDOOR_CONFIG["num_episodes"],
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
        lr=A2C_KEYDOOR_CONFIG["lr"],
        gamma=A2C_KEYDOOR_CONFIG["gamma"],
        entropy_coef=A2C_KEYDOOR_CONFIG["entropy_coef"],
        value_coef=A2C_KEYDOOR_CONFIG["value_coef"],
        n_steps=A2C_KEYDOOR_CONFIG["n_steps"],
        max_grad_norm=A2C_KEYDOOR_CONFIG["max_grad_norm"],
        print_every=A2C_KEYDOOR_CONFIG["print_every"],
    )

    # ---- plot training curves ----
    plot_path = os.path.join(OUTPUTS_DIR, "keydoorball_a2c_training.png")
    plot_training(history, title="A2C on KeyDoorBallEnv", save_path=plot_path)

    # ---- evaluate over 100 episodes ----
    # Evaluate with the shaped reward wrapper (same env the agent trained on)
    print()
    eval_env = make_keydoor_env(
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
        tile_size=A2C_KEYDOOR_CONFIG["tile_size"],
    )
    results = evaluate_agent(
        agent=agent,
        env=eval_env,
        num_episodes=100,
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
    )
    print_evaluation_results(results, title="A2C on KeyDoorBallEnv — Evaluation")
    eval_env.close()

    # ---- record video ----
    print()
    video_env = make_keydoor_env_video(
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
    )
    video_path = os.path.join(OUTPUTS_DIR, "keydoorball_a2c_trained.mp4")
    record_video(
        agent=agent,
        env=video_env,
        filename=video_path,
        num_episodes=1,
        max_steps=A2C_KEYDOOR_CONFIG["max_steps"],
    )
    video_env.close()

    # ---- summary ----
    print()
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    n = len(history["episode_rewards"])
    last100 = min(100, n)
    print(f"  Total episodes trained: {n}")
    print(f"  Last {last100} eps — Avg reward: {np.mean(history['episode_rewards'][-last100:]):.3f}")
    print(f"  Last {last100} eps — Avg steps:  {np.mean(history['episode_steps'][-last100:]):.1f}")
    if history.get("avg_entropy"):
        ent = history["avg_entropy"]
        print(f"  Final entropy:         {ent[-1]:.4f}  (start: {ent[0]:.4f})")
    print(f"  Eval success rate:     {results['success_rate']*100:.1f}%")
    print(f"  Outputs saved to:      {OUTPUTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
