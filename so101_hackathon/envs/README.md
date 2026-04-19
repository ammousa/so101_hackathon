# Environments

This package builds and launches the single public teleoperation environment used by all controllers.

## Key responsibility

- create the Isaac Lab config
- launch the Isaac application
- apply evaluation-only options such as video capture and timeout-only termination

## Key file

- `teleop_env.py`: environment assembly, app launch, and runtime options

The controller code is the student-facing API, and this package is the simulator-facing glue.
