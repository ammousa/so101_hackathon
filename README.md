# so101_hackathon

`so101_hackathon` is a simplified SO101 teleoperation repo for students.

The repo is centered on three ideas:

1. one shared teleop environment
2. one shared evaluation path
3. controllers are the part students modify

## Student Workflow

1. Open `so101_hackathon/controllers/templates/my_controller.py`
2. Implement `act(obs) -> action`
3. Register the controller in `so101_hackathon/registry.py`
4. Run `python scripts/evaluate.py --controller <name>`

## Built-in Baselines

- `pd`: simple proportional-derivative controller
- `ppo`: trained PPO checkpoint loaded through RSL-RL

## Main Commands

- `python scripts/list_controllers.py`
- `python scripts/train_rl.py --headless`
- `python scripts/evaluate.py --controller pd`
- `python scripts/play.py --controller pd`

## Notes

- The public API is intentionally small.
- Isaac Lab specifics live under `so101_hackathon/sim/`.
- The single environment keeps the same teleop observation ordering, action
  semantics, joint ordering, and disturbance model as the source repo.
