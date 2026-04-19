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
