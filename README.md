# Physical AI Hackathon - Graz 2026

Build a controller, run it in Isaac Lab, and see how well your follower arm tracks the leader.

This repo is the hackathon playground for the SO101 Robot Arm:
- one shared teleoperation environment
- one shared evaluation pipeline
- a small controller API that is easy to modify


## Quick Install

### 1) Create the environment
```bash
conda create -n hack python=3.11 -y
conda activate hack
```

### 2) Install PyTorch
```bash
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3) Install Isaac Sim and Isaac Lab

Isaacsim installation:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
isaacsim --headless
```
And IsaacLab:
```bash
mkdir -p external
cd external
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
./isaaclab.sh --install
```

### 4) Install this repo
```bash
cd /path/to/so101_hackathon
pip install -e .
```

## Quick Start

### List available controllers
```bash
python scripts/list_controllers.py
```

### Run the PD baseline
```bash
python scripts/evaluate.py --controller pd --headless
```

Key args:
- `--controller`: controller to run, such as `pd`, `ppo` as RL controller or `raw` as a starting template.
- `--headless`: disable the interactive viewer 
- `--num-episodes`: number of evaluation episodes to aggregate
- `--num-envs`: number of parallel Isaac environments

### Record an evaluation video
```bash
python scripts/evaluate.py --controller pd --headless --video --num-episodes 5
```

Key args:
- `--video`: record an MP4
- `--video-length`: cap the MP4 length in steps
- `--hide-leader-ghost`: hide the leader visualization robot

### Evaluate a trained PPO checkpoint
```bash
python scripts/evaluate.py \
  --controller ppo \
  --checkpoint-path /full/path/to/model_1500.pt \
  --headless \
  --video
```

Key args:
- `--checkpoint-path`: full path to the PPO checkpoint to load
- `--delay-steps`: override teleop action delay during evaluation
- `--noise-std`: override teleop action noise during evaluation
- `--output-dir`: force a custom artifact directory

### Train the PPO baseline
```bash
python scripts/train_rl.py --headless
```

Useful args:
- `--num-envs`: override the number of parallel training envs
- `--max-iterations`: override the training budget
- `--device`: choose `cuda:0`, `cuda:1`, or `cpu`
- `--resume`: continue from a previous training run
- `--load-run`: regex used to select the run when resuming
- `--checkpoint`: checkpoint name or regex used when resuming

Use `--help` on every script to see defaults and argument descriptions:
```bash
python scripts/evaluate.py --help
python scripts/train_rl.py --help
```

## Dev guide

1. Start from [so101_hackathon/controllers/raw.py](so101_hackathon/controllers/raw.py)
2. Implement `act(obs) -> action`
3. Register your controller in [so101_hackathon/registry.py](so101_hackathon/registry.py)
4. Run evaluation and inspect the outputs

Built-in baselines:
- `raw`: minimal pass-through controller that does nothing but returning the leader joint command directly
- `pd`: rule-based proportional-derivative controller (not tuned!)
- `ppo`: learned baseline loaded from an RSL-RL checkpoint

## Outputs and Artifacts

Evaluation runs always save:
- `config.json`
- `summary.json`
- `tensorboard/`
- `videos/` when `--video` is enabled

Output locations:
- `ppo`: nested under the training run, in `logs/rsl_rl/<experiment-name>/<train_run>/evaluation/<timestamp>/`
- non-RL controllers: `logs/<controller>/evaluation/<timestamp>/`

Training logs go under:
- `logs/rsl_rl/<experiment-name>/<timestamp>[_run-name]/`

## Repo Tour

Start with these nested READMEs:
- [scripts/README.md](scripts/README.md)
- [so101_hackathon/controllers/README.md](so101_hackathon/controllers/README.md)
- [so101_hackathon/envs/README.md](so101_hackathon/envs/README.md)
- [so101_hackathon/evaluation/README.md](so101_hackathon/evaluation/README.md)
- [so101_hackathon/sim/README.md](so101_hackathon/sim/README.md)
- [so101_hackathon/sim/mdp/README.md](so101_hackathon/sim/mdp/README.md)
- [so101_hackathon/training/README.md](so101_hackathon/training/README.md)
- [so101_hackathon/utils/README.md](so101_hackathon/utils/README.md)

## Short Module Map

- `scripts/`: CLI entrypoints
- `so101_hackathon/controllers/`: student controllers and baselines
- `so101_hackathon/envs/`: environment assembly and Isaac app launch
- `so101_hackathon/evaluation/`: evaluation-facing wrappers and metrics
- `so101_hackathon/sim/`: robot config, MDP terms, and simulator-facing logic
- `so101_hackathon/training/`: PPO config and RSL-RL integration
- `so101_hackathon/utils/`: helper functions for checkpoints, observations, training, and evaluation

## A Good First Loop

```bash
python scripts/list_controllers.py
python scripts/evaluate.py --controller raw --headless --num-episodes 3 --video
tensorboard --logdir logs/raw/evaluation
```
Then copy `raw.py`, register your controller name, and try to beat the baseline.
