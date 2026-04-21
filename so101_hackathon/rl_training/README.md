# Training

This package contains the PPO baseline wiring.

## Key files

- `ppo_config.py`: small, explicit PPO config surface for this repo
- `on_policy_runner.py`: thin wrapper around upstream `rsl_rl`
- `rsl_rl_wrapper.py`: environment wrapper for RSL-RL
- `runtime_utils.py`: runtime-safe device and renderer helpers

## What students usually need to know

- The PPO baseline is trained as a residual compensator, not as a direct absolute teleop imitator.
- The training observation matches the deployable real-robot teleop observation layout.
- The policy action is interpreted as a bounded residual correction that is added to the live leader command.
- PPO training logs go under `logs/rsl_rl/...`
- evaluation of PPO checkpoints nests under the training run directory
- custom controllers do not need this package unless they build on PPO

## Practical note

- Older PPO checkpoints from before the residual-action update are not expected to deploy correctly with the current runtime semantics.
