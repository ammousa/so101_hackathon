# Task Scripts

This directory holds task-specific entrypoints that sit beside the shared controller scripts.

## Current task scripts

- `sim_pick_orange/teleop.py`: launch the internal `PickOrange` task, read an SO101 leader arm, and run a registered controller in the loop

## PickOrange teleop arguments

The PickOrange script now supports the same built-in controller names as the rest of the repo:

```bash
python scripts/deploy/sim_pick_orange/teleop.py \
  --controller pd \
  --port /dev/ttyACM1 \
  --device cuda \
  --enable_cameras
```

- `--controller`: registered controller to deploy in sim, such as `raw`, `pd`, or `ppo`. Default: `raw`.
- `--controller-config`: optional YAML file with controller-specific overrides.
- `--checkpoint-path`: PPO or learned-controller checkpoint path.
- `--controller-coeff`: blend between direct leader teleop and controller output. `0.0` keeps pure leader control; `1.0` uses the full controller output.
- `--teleop_device`: teleop device name. Only `so101leader` is currently supported.
- `--port`: serial port for the SO101 leader arm.
- `--num_envs`: number of parallel PickOrange environments. Use `1` for interactive teleop.
- `--seed`: optional environment seed.
- `--step_hz`: target sim control frequency.
- `--delay-steps`: fixed post-controller command delay in control steps.
- `--noise-std`: Gaussian joint-space command noise in radians for joints 1-4 only. Delay still applies to all joints.
- `--disturbance-channel`: disturbance source. Use `fixed` for the built-in delay/noise model or `ultrazohm` to route commands through UltraZohm.
- `--uzohm-can-iface`: SocketCAN interface for UltraZohm, for example `can0`.
- `--uzohm-timeout-s`: UltraZohm manipulated-output timeout in seconds.
- `--recalibrate`: remove the cached leader calibration before connecting.
- `--device`: Torch/Isaac device string.
- `--enable_cameras`: enable RGB camera rendering.
- `--headless`: run without the viewer.

UltraZohm example:

```bash
python scripts/deploy/sim_pick_orange/teleop.py \
  --controller raw \
  --port /dev/ttyACM1 \
  --num_envs 1 \
  --device cuda \
  --enable_cameras \
  --disturbance-channel ultrazohm \
  --uzohm-can-iface can0
```

Use the UltraZohm noise panel separately to change noise and delay while the teleop script runs:

```bash
python external/ultrazohm/sebi-scripts/09_noise_control_panel.py --can-iface can0
```

## Scope

- These scripts are separate from the public controller workflow in `scripts/evaluate.py` and `scripts/train_rl.py`.
- They are useful when a task needs a custom teleop loop or a different observation structure.
