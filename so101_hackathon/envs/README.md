# Environments

This package owns the simulator-facing environment code for the repo.

For a beginner, there are only three files to care about:

- `teleop_env.py`: the training/evaluation env used by the controller pipeline
- `pick_orange_env.py`: the kitchen env used by the PickOrange teleop script
- `common.py`: the shared pieces both envs build on

## Key responsibility

- create the Isaac Lab config
- launch the Isaac application
- apply evaluation-only options such as video capture and timeout-only termination
- keep shared env pieces in one obvious place

## Key files

- `common.py`: shared builder base, single-arm config, teleop action helpers, and scene helpers
- `teleop_env.py`: legacy controller-training environment assembly and launch
- `pick_orange_env.py`: teleop kitchen environment config, builder, and task-specific helper functions

The controller code is still the student-facing API, and this package is the shared simulator-facing glue.
