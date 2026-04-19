# Controllers

This package is the main student workspace.

## What belongs here

- built-in baselines such as `rule_based_pd.py` and `rl_ppo.py`
- new student controllers
- the starter template in `templates/my_controller.py`

## Key files

- `base.py`: tiny controller interface
- `rule_based_pd.py`: simple baseline to beat first
- `rl_ppo.py`: checkpoint-backed PPO controller
- `templates/my_controller.py`: best place to start a custom policy

## Typical change

Implement `act(obs) -> action`, then register the controller in `so101_hackathon/registry.py`.
