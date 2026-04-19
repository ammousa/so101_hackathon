# Utilities

This package holds small helpers that keep the scripts and core modules readable.

## Files

- `eval_utils.py`: evaluation rollout, artifact writing, TensorBoard logging, and output-dir resolution
- `train_utils.py`: training log-dir helpers
- `checkpoints.py`: checkpoint path resolution
- `obs_utils.py`: observation parsing helpers for controller authors
- `action_utils.py`: action formatting helpers

If a script starts feeling too large, this is where shared helper logic should move.
