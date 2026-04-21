# Task Scripts

This directory holds task-specific entrypoints that sit beside the shared controller scripts.

## Current task scripts

- `pick_orange/teleop.py`: launch and teleoperate the internal `PickOrange` task with an SO101 leader arm

## Scope

- These scripts are separate from the public controller workflow in `scripts/evaluate.py` and `scripts/train_rl.py`.
- They are useful when a task needs a custom teleop loop or a different observation structure.
