# MDP Terms

This directory contains the Isaac Lab manager terms used by the teleop task.

## Files

- `actions.py`: action application and teleop disturbance model
- `commands.py`: leader joint trajectory generation and leader visualization driving
- `observations.py`: policy observation terms
- `rewards.py`: training rewards
- `terminations.py`: failure conditions used by training and counted during evaluation
- `curriculum.py`: disturbance curriculum logic
- `adaptive_curriculum_utils.py`: helper math for curriculum and trajectory shaping

## Mental model

Training and evaluation both use the same environment, but these files define what the environment means:
- what the leader does
- what the follower observes
- how actions are applied
- what counts as success or failure

The teleop disturbance model delays the full joint command, but Gaussian command noise is masked to joints 1-4 only. The wrist roll and gripper joints are not noised. The same mask is used for reset-time joint perturbations in simulation.
