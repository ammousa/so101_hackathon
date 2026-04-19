# Training

This package contains the PPO baseline wiring.

## Key files

- `ppo_config.py`: small, explicit PPO config surface for this repo
- `on_policy_runner.py`: thin wrapper around upstream `rsl_rl`
- `rsl_rl_wrapper.py`: environment wrapper for RSL-RL
- `runtime_utils.py`: runtime-safe device and renderer helpers

## What students usually need to know

- PPO training logs go under `logs/rsl_rl/...`
- evaluation of PPO checkpoints nests under the training run directory
- custom controllers do not need this package unless they build on PPO
