# PickOrange Script

This folder contains the teleop entrypoint for the internal `PickOrange` kitchen environment.

## File

- `teleop.py`: connect to an SO101 leader, build the PickOrange env config directly, and step it in real time

## Runtime behavior

- Press `B` to start teleoperation.
- Press `R` to reset the task.
- Press `N` to mark success and reset the task.

This script does not include training or evaluation wrappers. Those remain on the legacy public stack.
