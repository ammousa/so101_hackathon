# Simulation

This package contains simulator-facing logic that students usually do not need to touch first.

## Main responsibilities

- robot kinematics
- robot asset configuration
- MDP terms for actions, observations, rewards, commands, and terminations

## Key submodules

- `mdp/`: Isaac Lab manager terms and task logic
- `robots/`: the shared internalized SO101 follower robot configuration and metadata
- `kinematics.py`: end-effector pose helpers used by evaluation and metrics

## Notes

- The repo now uses a single shared SO101 follower robot config derived from the internalized follower asset.
- Environment-specific scene and teleop logic lives under `so101_hackathon/envs/`, not here.

See [mdp/README.md](mdp/README.md) for the term-by-term breakdown.
