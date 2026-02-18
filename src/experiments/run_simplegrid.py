"""
Experiment: Actor-Critic (A2C) on SimpleGridEnv

Run this script to:
    1. Train an A2C agent on the SimpleGridEnv
    2. Plot training curves (saved to outputs/)
    3. Evaluate over 100 episodes (printed + saved)
    4. Record a video of the trained agent (saved to outputs/)

Usage:
    python -m src.experiments.run_simplegrid          (from project root)
"""

import os
import sys
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work when
# executed with  `python -m src.experiments.run_simplegrid`  from the
# project root, or  `python run_simplegrid.py`  from this directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.envs import SimpleGridEnv
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

A2C_CONFIG = {
    "num_episodes": 30000,
    "max_steps": 100,          # random agent finds goal ~17 % of the time
    "lr": 0.0003,              # lower LR for stability (A2C is noisy)
    "gamma": 0.95,
    "entropy_coef": 0.02,      # lighter entropy to let policy commit
    "value_coef": 0.5,         # critic loss weight
    "n_steps": 20,             # longer horizon covers more of the episode
    "max_grad_norm": 0.5,
    "print_every": 500,
    "tile_size": 10,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print(" A2C on SimpleGridEnv")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")
    print(f"Config: {A2C_CONFIG}")
    print()

    # ---- reproducibility ----
    set_seed(SEED)

    # ---- train ----
    agent, history = train_actor_critic(
        env_class=SimpleGridEnv,
        device=device,
        **A2C_CONFIG,
    )

    # ---- plot training curves ----
    plot_path = os.path.join(OUTPUTS_DIR, "simplegrid_a2c_training.png")
    plot_training(history, title="A2C on SimpleGridEnv", save_path=plot_path)

    # ---- evaluate over 100 episodes ----
    print()
    results = evaluate_agent(
        agent=agent,
        env_class=SimpleGridEnv,
        num_episodes=100,
        max_steps=A2C_CONFIG["max_steps"],
    )
    print_evaluation_results(results, title="A2C on SimpleGridEnv — Evaluation")

    # ---- record video ----
    print()
    video_path = os.path.join(OUTPUTS_DIR, "simplegrid_a2c_trained.mp4")
    record_video(
        agent=agent,
        env_class=SimpleGridEnv,
        filename=video_path,
        num_episodes=1,
        max_steps=A2C_CONFIG["max_steps"],
    )

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
