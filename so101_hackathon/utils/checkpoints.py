"""Checkpoint resolution helpers.

The hackathon repo keeps checkpoint loading explicit because students often need
to understand where a baseline model is coming from before they modify it.
"""

from __future__ import annotations

import os
import re


def resolve_checkpoint_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    """Resolve a checkpoint from a log root or accept a direct path."""

    if os.path.isfile(load_checkpoint):
        return os.path.abspath(load_checkpoint)

    if not os.path.isdir(log_root_path):
        raise FileNotFoundError(f"Log root does not exist: {log_root_path}")

    run_pattern = re.compile(load_run)
    checkpoint_pattern = re.compile(load_checkpoint)

    candidate_runs = []
    for run_name in os.listdir(log_root_path):
        run_path = os.path.join(log_root_path, run_name)
        if os.path.isdir(run_path) and run_pattern.fullmatch(run_name):
            candidate_runs.append(run_name)

    if not candidate_runs:
        raise FileNotFoundError(f"No run matched pattern '{load_run}' under {log_root_path}")

    selected_run = sorted(candidate_runs)[-1]
    run_dir = os.path.join(log_root_path, selected_run)

    candidate_checkpoints = []
    for file_name in os.listdir(run_dir):
        file_path = os.path.join(run_dir, file_name)
        if os.path.isfile(file_path) and checkpoint_pattern.fullmatch(file_name):
            candidate_checkpoints.append(file_name)

    if not candidate_checkpoints:
        raise FileNotFoundError(
            f"No checkpoint matched pattern '{load_checkpoint}' under {run_dir}"
        )

    selected_checkpoint = sorted(candidate_checkpoints)[-1]
    return os.path.join(run_dir, selected_checkpoint)
