# Deploy

This package contains the real-hardware deployment helpers for the SO101 teleoperation controllers.

## Key responsibility

- connect the leader and follower hardware through `lerobot`
- rebuild the standard teleop observation used by the hackathon controllers
- run the generic deploy loop for any registered controller
- record deploy artifacts such as `config.json`, `summary.json`, and `timeseries.csv`

## Key files

- `hardware.py`: leader/follower dependency loading and robot construction
- `runtime.py`: observation reconstruction, joint-limit handling, action blending, and artifact writing
- `metrics.py`: deploy-time tracking metrics and time-series summaries
- `session.py`: the controller-agnostic real-time deploy loop

## How it fits together

The public CLI lives in `scripts/deploy.py`.

That script:

- loads the controller through `so101_hackathon/registry.py`
- creates the real hardware pair
- uses `LiveTeleopObservationBuilder` to feed the same observation layout that controllers see in simulation
- blends controller output with direct leader teleop using `--controller-coeff` / `--rl_coeff`
- writes deploy outputs under `logs/<controller>/deploy/...` or under the PPO training run when a checkpoint is used

## PPO semantics

- The current PPO controller is a residual-compensation policy.
- It does not output a full absolute robot target by itself.
- During deployment, the PPO output is added to the live leader joint command, then clamped and sent to the follower.
- This keeps the policy deployable using only robot-available observations.

## Calibration

Use the repo-local calibration script before the first deploy or after changing motors, wiring, or arm mechanics.

Calibrate the leader:

```bash
python scripts/calibrate_hardware.py \
  --role leader \
  --port /dev/ttyACM1 \
  --id my_awesome_leader_arm
```

Calibrate the follower:

```bash
python scripts/calibrate_hardware.py \
  --role follower \
  --port /dev/ttyACM0 \
  --id my_awesome_follower_arm
```

Follower without gripper:

```bash
python scripts/calibrate_hardware.py \
  --role follower \
  --port /dev/ttyACM0 \
  --id my_awesome_follower_arm \
  --disable-gripper
```

Interactive calibration flow:

1. Run the script for the arm you want to calibrate.
2. Move the arm to the middle of its range of motion and press `Enter`.
3. Move every joint through its full range of motion.
4. Press `Enter` again to finish and save the calibration JSON.

By default, calibration files are written to `~/.cache/huggingface/lerobot/calibration/`.
The deploy and teleop paths in this repo read from that directory.

## How to deploy PD

Deploy the built-in PD controller with the default hardware ports:

```bash
python scripts/deploy.py --controller pd
```

Use custom serial ports and stop automatically after 30 seconds:

```bash
python scripts/deploy.py \
  --controller pd \
  --leader-port /dev/ttyACM1 \
  --follower-port /dev/ttyACM0 \
  --teleop-time-s 30
```

Tune the PD gains from a YAML config:

```bash
python scripts/deploy.py \
  --controller pd \
  --controller-config path/to/pd.yaml
```

Example `pd.yaml`:

```yaml
kp: 0.8
kd: 0.1
max_action: 0.5
```

Reduce the controller influence while keeping partial direct leader teleop:

```bash
python scripts/deploy.py \
  --controller pd \
  --controller-coeff 0.5
```

`--controller-coeff 0.0` means pure leader teleop, and `--controller-coeff 1.0` means full controller output.

## Common arguments

- `--controller`: registered controller name, such as `pd`, `ppo`, or `raw`
- `--controller-config`: optional YAML file with controller-specific settings
- `--checkpoint-path`: checkpoint path forwarded to learned controllers such as `ppo`
- `--leader-port`, `--follower-port`: serial ports for the leader and follower arms
- `--leader-id`, `--follower-id`: hardware IDs passed into `lerobot`
- `--fps`: target control frequency for the deploy loop
- `--teleop-time-s`: optional runtime limit in seconds
- `--print-every`: print one live status line every N control steps
- `--device`: runtime device for learned controllers, for example `cuda:0` or `cpu`
- `--controller-coeff`: generic blend coefficient between direct teleop and controller output
- `--rl_coeff`: compatibility alias for `--controller-coeff`
- `--output-dir`: explicit artifact directory instead of the default `logs/.../deploy/...` location

## PPO example

Deploy a trained PPO checkpoint:

```bash
python scripts/deploy.py \
  --controller ppo \
  --checkpoint-path /full/path/to/model_1500.pt \
  --device cuda:0
```

## Design goal

Future controllers should become deployable automatically as long as they:

- use the shared teleop observation layout
- return actions whose runtime semantics are clearly defined in the deploy loop
- are registered in `so101_hackathon/registry.py`
