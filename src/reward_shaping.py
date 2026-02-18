"""
Reward shaping wrappers for MiniGrid environments.

SmartRewardWrapper: Phase-aware dense reward shaping for KeyDoorBallEnv.
Written by Dor — ported from the shared notebook.
"""

import gymnasium as gym


class SmartRewardWrapper(gym.Wrapper):
    """
    Phase-aware reward shaping for KeyDoorBallEnv.

    The default KeyDoorBallEnv rewards are too sparse for pixel-based RL.
    This wrapper provides dense feedback by:
      - Tracking which phase the agent is in (key → door → ball → goal)
      - Giving milestone bonuses when sub-goals are completed
      - Providing a distance-based shaping signal toward the current target
      - Rewarding proximity + correct facing toward the current target
      - Penalising wall bumps and each time step

    Reward budget (approximate):
      Key pickup:   +2.0
      Door open:    +3.0
      Ball pickup:  +2.0
      Goal reached: +10.0
      Step penalty:  -0.02 per step
      Distance:     ±0.3 per step (closer/farther from target)
      Proximity:    +0.05 when adjacent and facing the target
      Wall bump:    -0.03
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_dist = None
        self._phase = 0

    def _manhattan(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_target_pos(self):
        uw = self.env.unwrapped
        if self._phase == 0:
            return uw.key_pos
        elif self._phase == 1:
            return uw.door_pos
        elif self._phase == 2:
            return uw.ball_pos
        else:
            return uw.goal_pos

    def _update_phase(self):
        """Advance phase and return milestone bonus (if any)."""
        uw = self.env.unwrapped
        bonus = 0.0
        if self._phase == 0 and uw.is_carrying_key():
            self._phase = 1
            bonus = 2.0
        if self._phase == 1 and uw.is_door_open():
            self._phase = 2
            bonus += 3.0
        if self._phase == 2 and uw.is_carrying_ball():
            self._phase = 3
            bonus += 2.0
        return bonus

    def _facing_target(self) -> bool:
        """Check if agent's front_pos matches the current phase target."""
        uw = self.env.unwrapped
        target = self._get_target_pos()
        front = uw.front_pos
        return (front[0] == target[0] and front[1] == target[1])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phase = 0
        self._prev_dist = self._manhattan(
            self.env.unwrapped.agent_pos, self._get_target_pos()
        )
        return obs, info

    def step(self, action):
        pos_before = tuple(self.env.unwrapped.agent_pos)

        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        # Phase transitions
        milestone_bonus = self._update_phase()

        # Terminal — big reward so it dominates everything
        if terminated:
            return obs, 10.0 + milestone_bonus, terminated, truncated, info

        # Dense distance signal (±0.3 per step)
        target = self._get_target_pos()
        cur_dist = self._manhattan(self.env.unwrapped.agent_pos, target)
        dist_delta = 0.0
        if self._prev_dist is not None:
            dist_delta = 0.3 * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # Proximity bonus: reward being adjacent + facing the target
        proximity = 0.05 if self._facing_target() else 0.0

        # Wall bump penalty
        wall_bump = -0.03 if (action == 2 and tuple(self.env.unwrapped.agent_pos) == pos_before) else 0.0

        # Step penalty: -0.02 so timeout(300) = -6.0, clearly worse than partial progress
        shaped = -0.02 + dist_delta + milestone_bonus + proximity + wall_bump
        return obs, shaped, terminated, truncated, info
