"""
Test script for networks.

Run this to verify that CNNFeatureExtractor and PolicyNetwork work correctly.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from networks import obs_to_tensor, CNNFeatureExtractor, PolicyNetwork
from envs import SimpleGridEnv, KeyDoorBallEnv
from preprocessing import pre_process

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def test_obs_to_tensor():
    """Test obs_to_tensor conversion."""
    print("=" * 60)
    print("Testing obs_to_tensor")
    print("=" * 60)

    obs = np.random.randint(0, 255, size=(84, 84, 1), dtype=np.uint8)
    t = obs_to_tensor(obs, device)

    assert t.shape == (1, 1, 84, 84), f"Expected (1, 1, 84, 84), got {t.shape}"
    assert t.dtype == torch.float32
    assert t.device.type == device.type
    assert t.is_contiguous()
    print(f"   Input: {obs.shape} uint8  ->  Output: {tuple(t.shape)} float32")
    print("   ✓ Passed\n")


def test_cnn_feature_extractor():
    """Test CNNFeatureExtractor."""
    print("=" * 60)
    print("Testing CNNFeatureExtractor")
    print("=" * 60)

    cnn = CNNFeatureExtractor(input_channels=1, feature_dim=256).to(device)

    x = torch.randn(1, 1, 84, 84, device=device)
    with torch.no_grad():
        features = cnn(x)

    print(f"   Input:  {tuple(x.shape)}")
    print(f"   Output: {tuple(features.shape)}")
    assert features.shape == (1, 256), f"Expected (1, 256), got {features.shape}"

    # Batch of 4
    x_batch = torch.randn(4, 1, 84, 84, device=device)
    with torch.no_grad():
        features_batch = cnn(x_batch)
    assert features_batch.shape == (4, 256)
    print(f"   Batch:  {tuple(x_batch.shape)} -> {tuple(features_batch.shape)}")
    print("   ✓ Passed\n")


def test_policy_network():
    """Test PolicyNetwork with both environments."""
    print("=" * 60)
    print("Testing PolicyNetwork")
    print("=" * 60)

    # --- SimpleGridEnv (3 actions) ---
    print("\n   SimpleGridEnv (3 actions):")
    simple_env = SimpleGridEnv(preprocess=pre_process, max_steps=100)
    obs, _ = simple_env.reset(seed=42)

    policy = PolicyNetwork(num_actions=3, feature_dim=256).to(device)

    obs_t = obs_to_tensor(obs, device)
    with torch.no_grad():
        probs = policy(obs_t)

    print(f"   Obs shape: {obs.shape}  ->  Tensor: {tuple(obs_t.shape)}")
    print(f"   Probs: {probs.squeeze().cpu().numpy()}  sum={probs.sum().item():.4f}")
    assert probs.shape == (1, 3)
    assert abs(probs.sum().item() - 1.0) < 1e-5

    action, log_prob = policy.select_action(obs, device)
    print(f"   Action: {action}  log_prob: {log_prob.item():.4f}")
    assert 0 <= action < 3
    assert log_prob.requires_grad
    print("   ✓ Passed")

    # --- KeyDoorBallEnv (5 actions) ---
    print("\n   KeyDoorBallEnv (5 actions):")
    kd_env = KeyDoorBallEnv(preprocess=pre_process, max_steps=100)
    obs_kd, _ = kd_env.reset(seed=42)

    policy_kd = PolicyNetwork(num_actions=5, feature_dim=256).to(device)

    action_kd, log_prob_kd = policy_kd.select_action(obs_kd, device)
    print(f"   Obs shape: {obs_kd.shape}")
    print(f"   Action: {action_kd}  log_prob: {log_prob_kd.item():.4f}")
    assert 0 <= action_kd < 5
    assert log_prob_kd.requires_grad
    print("   ✓ Passed\n")


def test_gradient_flow():
    """Verify gradients flow through the policy network."""
    print("=" * 60)
    print("Testing gradient flow")
    print("=" * 60)

    policy = PolicyNetwork(num_actions=3, feature_dim=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    obs = np.random.randint(0, 255, size=(84, 84, 1), dtype=np.uint8)
    action, log_prob = policy.select_action(obs, device)

    loss = -log_prob  # simple REINFORCE-style loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check at least one parameter got a gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert has_grad, "No gradients flowed through the network"
    print("   ✓ Gradients flow correctly\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NETWORK TESTS")
    print("=" * 60 + "\n")

    test_obs_to_tensor()
    test_cnn_feature_extractor()
    test_policy_network()
    test_gradient_flow()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")
