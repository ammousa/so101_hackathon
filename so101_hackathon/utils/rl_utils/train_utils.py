"""Shared helpers for the training entrypoint."""

from __future__ import annotations

import os
from datetime import datetime


def build_training_log_dir(log_root: str, run_name: str) -> str:
    """Build the per-run training log directory."""

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_dir = f"{log_dir}_{run_name}"
    return os.path.join(log_root, log_dir)
