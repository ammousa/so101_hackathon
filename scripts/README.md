# Scripts

This directory contains the command-line entrypoints students actually run.

## Files

- `evaluate.py`: run evaluation, record videos, and save JSON + TensorBoard artifacts
- `train_rl.py`: train or resume the PPO baseline training
- `list_controllers.py`: print the currently registered controllers
- `calibrate_hardware.py`: interactively calibrate one real SO101 leader or follower arm
- `deploy/deploy_traj.py`: run a CSV leader trajectory on real follower hardware through any registered controller
- `deploy/sim_pick_orange/teleop.py`: teleoperate the internal PickOrange kitchen env with an SO101 leader arm and a registered controller
- `deploy/sim_pick_orange/traj.py`: run the internal PickOrange kitchen env with a CSV leader trajectory instead of an SO101 leader arm

## Typical flow

1. `python scripts/list_controllers.py`
2. `python scripts/evaluate.py --controller pd --headless`
3. `python scripts/train_rl.py --headless`
4. `python scripts/deploy/deploy_traj.py --controller raw --trajectory-config config/traj.yaml`

## Notes

- The legacy student workflow still goes through `evaluate.py` and `train_rl.py`.
- The CSV trajectory scripts are scenario entrypoints; they do not add a `traj` controller.
- The `scripts/tasks/` subtree is for internal env-specific entrypoints that do not affect the public controller API.

Use `--help` on each script to see argument descriptions and defaults.
