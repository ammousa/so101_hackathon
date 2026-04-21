# Robots

This package defines the single shared SO101 robot used across the repo.

## Files

- `so101_follower_spec.py`: robot metadata, joint names, body names, joint limits, motor limits, rest-pose ranges, and motor/joint conversion helpers
- `so101_follower_cfg.py`: Isaac Lab articulation config pointing at the vendored `so101_follower.usd`

## Design

- `SO101_FOLLOWER_CFG` is the only supported robot config export.
- Legacy teleop, evaluation, deploy helpers, and the internal task stack all read from the same follower spec.
- `trs_so101/` is retained only as leftover upstream asset history and license context, not as an active config source.
