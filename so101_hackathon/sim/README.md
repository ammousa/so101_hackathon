# Simulation

This package contains simulator-facing logic that students usually do not need to touch first.

## Main responsibilities

- robot kinematics
- robot asset configuration
- MDP terms for actions, observations, rewards, commands, and terminations

## Key submodules

- `mdp/`: Isaac Lab manager terms and task logic
- `robots/`: SO101 robot configuration and USD/URDF assets
- `kinematics.py`: end-effector pose helpers used by evaluation and metrics

See [mdp/README.md](mdp/README.md) for the term-by-term breakdown.
