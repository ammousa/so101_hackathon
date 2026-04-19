"""Pure helpers shared by SO101 teleop runtime code and unit tests."""

from __future__ import annotations

import torch


def apply_delay_sequence(sequence: torch.Tensor, delay_steps: int) -> torch.Tensor:
    """Return a delayed copy of a step-major sequence."""
    if delay_steps < 0:
        raise ValueError(f"delay_steps must be >= 0, received {delay_steps}")
    if sequence.ndim < 2:
        raise ValueError(f"Expected sequence shape (T, ...), received {tuple(sequence.shape)}")

    delayed = sequence.clone()
    for index in range(sequence.shape[0]):
        delayed[index] = sequence[max(index - delay_steps, 0)]
    return delayed


def compose_residual_joint_commands(
    target_positions: torch.Tensor,
    residual_actions: torch.Tensor,
    action_scale: torch.Tensor | float,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose and clamp teleoperation joint commands."""
    commands = target_positions + residual_actions * action_scale
    if noise is not None:
        commands = commands + noise
    return torch.clamp(commands, min=lower_limits, max=upper_limits)
