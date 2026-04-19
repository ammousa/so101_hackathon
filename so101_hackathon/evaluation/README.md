# Evaluation

This package contains evaluation-facing wrappers and metrics.

## Key files

- `metrics.py`: shared rollout metrics such as joint RMSE, smoothness, failure counts, and episode summaries

## What evaluation produces

- `config.json`
- `summary.json`
- `tensorboard/`
- `videos/` when enabled

Evaluation logic itself lives in `so101_hackathon/utils/eval_utils.py`.
