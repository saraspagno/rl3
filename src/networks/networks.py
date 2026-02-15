"""
Neural network architectures for Deep RL.

This module contains:
- CNNFeatureExtractor: CNN backbone for processing image observations
- PolicyNetwork: Policy network for REINFORCE (outputs action probabilities)
- ActorCriticNetwork: Shared-backbone network with policy + value heads (A2C)

Expected input convention:
    Observations arrive as numpy arrays of shape (84, 84, 3) from preprocessing.
    Call obs_to_tensor() to convert to (1, 3, 84, 84) float tensor before forward().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def obs_to_tensor(obs, device):
    """
    Convert a preprocessed numpy observation to a batched tensor in (N, C, H, W) format.
    Normalises uint8 pixels from [0, 255] to [0, 1] — critical for CNN convergence
    (Kaiming-initialised weights assume ~unit-variance inputs).

    Args:
        obs: numpy array of shape (H, W, C), e.g. (84, 84, 3), dtype uint8
        device: torch device

    Returns:
        Tensor of shape (1, C, H, W) on the given device, values in [0, 1]
    """
    # Wrap the numpy array (zero-copy), permute to (C, H, W) as a view,
    # then do a single contiguous copy when converting uint8→float32.
    # Avoids the intermediate numpy float32 array of the old path.
    t = (
        torch.from_numpy(obs)          # (H,W,C) uint8 — zero-copy wrap
        .permute(2, 0, 1)              # (C,H,W) — view, no copy
        .unsqueeze(0)                  # (1,C,H,W) — view
        .to(dtype=torch.float32, device=device)  # single contiguous copy
    )
    t.div_(255.0)                      # in-place: [0,255] -> [0,1]
    return t


class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features from preprocessed grid-world images.

    Uses small 3×3 kernels with stride 2 to capture fine spatial detail
    (agent triangle, goal square) in the sparse 84×84 grayscale images.
    Inputs are expected in [0, 1] (via obs_to_tensor normalisation).

    Input:  (batch, 3, 84, 84)  — channels-first RGB, values in [0, 1]
    Output: (batch, feature_dim)

    Architecture:
        Conv1 + ReLU: 32 × 3×3, stride 2, pad 1 → 42×42
        Conv2 + ReLU: 64 × 3×3, stride 2, pad 1 → 21×21
        Conv3 + ReLU: 64 × 3×3, stride 2, pad 1 → 11×11
        Conv4 + ReLU: 64 × 3×3, stride 2, pad 1 →  6×6
        Flatten → 2304
        FC + ReLU → feature_dim
    """

    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        # 64 * 6 * 6 = 2304  (84 → 42 → 21 → 11 → 6)
        self.fc = nn.Linear(64 * 6 * 6, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Float tensor of shape (batch, C, H, W).

        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities (for REINFORCE).

    Architecture:
        CNNFeatureExtractor (feature_dim) -> Linear -> softmax

    Input:  (batch, 3, 84, 84)  — channels-first RGB float tensor
    Output: (batch, num_actions) — action probabilities
    """

    def __init__(self, num_actions, feature_dim=256):
        super().__init__()

        self.cnn = CNNFeatureExtractor(input_channels=3, feature_dim=feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)

    def forward(self, x):
        """
        Args:
            x: Float tensor of shape (batch, 1, 84, 84).

        Returns:
            logits: (batch, num_actions) — raw logits (unnormalised).
                    Use Categorical(logits=...) to sample, which is faster
                    and more numerically stable than softmax → Categorical(probs).
        """
        features = self.cnn(x)
        return self.policy_head(features)

    def select_action(self, obs, device):
        """
        Sample an action from the policy given a single numpy observation.

        Args:
            obs: numpy array of shape (84, 84, 3)
            device: torch device (model must already be on this device)

        Returns:
            action:   int — sampled action index
            log_prob: scalar tensor shape () — log-probability (keeps grad)
        """
        obs_tensor = obs_to_tensor(obs, device)
        logits = self.forward(obs_tensor)

        # squeeze(0) removes batch dim so log_prob is a true scalar ()
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with a shared CNN backbone and two heads:
        - Actor  (policy head): outputs action logits
        - Critic (value head):  outputs scalar state value V(s)

    Sharing the backbone means the CNN learns features useful for both
    action selection and value estimation, which is more sample-efficient.

    Input:  (batch, 3, 84, 84)
    Output: logits (batch, num_actions), value (batch, 1)
    """

    def __init__(self, num_actions, feature_dim=256):
        super().__init__()

        self.cnn = CNNFeatureExtractor(input_channels=3, feature_dim=feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        Returns:
            logits: (batch, num_actions) — raw unnormalised action scores
            value:  (batch, 1) — estimated state value V(s)
        """
        features = self.cnn(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(self, obs, device):
        """
        Select an action for a single observation (keeps grad for training).

        Returns:
            action: int
            log_prob: scalar tensor shape () (with grad)
            entropy: scalar tensor shape ()
            value: scalar tensor shape () (with grad)
        """
        obs_t = obs_to_tensor(obs, device)
        logits, value = self.forward(obs_t)

        # squeeze(0) removes the batch dim so Categorical has batch_shape ()
        # → log_prob, entropy, sample all return true scalars (shape ())
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action),   # shape ()
            dist.entropy(),          # shape ()
            value.squeeze(),         # (1,1) -> scalar ()
        )

    @torch.no_grad()
    def get_value(self, obs, device):
        """Get V(s) for a single observation (no grad, for bootstrapping)."""
        obs_t = obs_to_tensor(obs, device)
        _, value = self.forward(obs_t)
        return value.squeeze().item()
