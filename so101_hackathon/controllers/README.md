# Controllers

This package is the main student workspace.

## What belongs here

- built-in baselines such as `rule_based_pd.py` and `rl_ppo.py`
- new student controllers
- the `raw.py` baseline, which you can use as the starter template

## Key files

- `base.py`: tiny controller interface
- `raw.py`: minimal pass-through starter controller
- `rule_based_pd.py`: simple baseline to beat first
- `rl_ppo.py`: checkpoint-backed PPO controller

## Typical change

Copy `raw.py`, implement `act(obs) -> action`, then register the controller in `so101_hackathon/registry.py`.
