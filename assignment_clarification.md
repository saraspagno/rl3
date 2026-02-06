Clarification regarding Reward Shaping and auxiliary variables – Final Project

Following a question raised by students, I would like to clarify the policy regarding Reward Shaping and the use of additional internal variables in the provided MiniGrid environments.

You are not limited to pixels only.
The provided environments intentionally include internal state indicators (e.g., is_carrying_key, is_door_open, is_carrying_ball, and the prev_* flags), which you are allowed to use for Reward Shaping.

You are allowed to store additional internal variables in __init__, reset, or step, as long as they represent discrete task-related events or logical states, such as:

picking up the key
opening the door
picking up the ball
progressing between task stages
In addition, it is allowed to use a negative reward per step (step penalty) as part of Reward Shaping, in order to encourage faster solutions (fewer steps).

However, it is not allowed to use variables or reward signals that make the task trivial or bypass the intended learning challenge, including (but not limited to):

distance to the goal / key / door / ball
minimum distance tracked during an episode
any continuous or geometric distance-based signal between entities
Such information significantly simplifies the problem and allows solving it with minimal reliance on the observation, which is not the goal of this assignment.

Recommendation:
The state variables already provided in the environments are sufficient to implement effective and meaningful Reward Shaping.
If you choose to add additional variables, they must describe events and not encode the solution directly.

As always, any Reward Shaping logic and auxiliary variables (existing or added) must be clearly explained in the code and in the report.