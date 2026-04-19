# Scripts

This directory contains the command-line entrypoints students actually run.

## Files

- `evaluate.py`: run evaluation, record videos, and save JSON + TensorBoard artifacts
- `play.py`: run one controller rollout in real time with the same evaluation stack
- `train_rl.py`: train or resume the PPO baseline training
- `list_controllers.py`: print the currently registered controllers

## Typical flow

1. `python scripts/list_controllers.py`
2. `python scripts/evaluate.py --controller pd --headless`
3. `python scripts/play.py --controller pd`
4. `python scripts/train_rl.py --headless`

Use `--help` on each script to see argument descriptions and defaults.
