"""Pure helpers for the adaptive teleoperation curriculum.

These helpers deliberately avoid Isaac Lab imports so the sampling and level
update logic can be unit-tested in a lightweight environment.
"""

from __future__ import annotations

from typing import Final

import torch


_EPS: Final[float] = 1.0e-6


def update_difficulty_levels(
    levels: torch.Tensor,
    episode_joint_rmse: torch.Tensor,
    sample_count: torch.Tensor,
    good_threshold: float,
    bad_threshold: float,
    max_level: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ANYmal-style promotion/demotion with hysteresis.

    Environments with no completed episode samples keep their current level.
    """

    valid_mask = sample_count > 0
    promoted = valid_mask & (episode_joint_rmse < good_threshold)
    demoted = valid_mask & (episode_joint_rmse > bad_threshold)

    next_levels = levels + promoted.to(levels.dtype) - demoted.to(levels.dtype)
    next_levels = torch.clamp(next_levels, min=0, max=max_level)
    return next_levels, promoted, demoted


def compute_episode_joint_rmse(
    sq_error_sum: torch.Tensor,
    sample_count: torch.Tensor,
) -> torch.Tensor:
    """Convert accumulated mean-squared joint error into episode RMSE."""

    safe_count = torch.clamp(sample_count, min=1.0)
    return torch.sqrt(sq_error_sum / safe_count)


def sample_waypoint_targets(
    start_pos: torch.Tensor,
    safe_lower: torch.Tensor,
    safe_upper: torch.Tensor,
    active_joint_count_low: torch.Tensor,
    active_joint_count_high: torch.Tensor,
    position_span_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample waypoint targets inside a level-dependent local joint window."""

    num_envs, num_joints = start_pos.shape
    usable_range = torch.clamp(safe_upper - safe_lower, min=0.0)
    span_scale = torch.clamp(position_span_scale, min=0.0, max=1.0).unsqueeze(-1)
    half_window = 0.5 * usable_range * span_scale
    low = torch.clamp(active_joint_count_low.to(dtype=torch.int), min=1, max=num_joints)
    high = torch.clamp(active_joint_count_high.to(dtype=torch.int), min=1, max=num_joints)
    high = torch.maximum(high, low)
    sampled_count = low + torch.floor(
        torch.rand(num_envs, device=start_pos.device) * (high - low + 1).to(dtype=torch.float32)
    ).to(dtype=torch.int)

    joint_scores = torch.rand(num_envs, num_joints, device=start_pos.device, dtype=start_pos.dtype)
    joint_rank = joint_scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    active_mask = joint_rank < sampled_count.unsqueeze(-1)

    local_lower = torch.maximum(start_pos - half_window, safe_lower)
    local_upper = torch.minimum(start_pos + half_window, safe_upper)
    local_upper = torch.maximum(local_upper, local_lower + _EPS)
    sampled_targets = local_lower + (local_upper - local_lower) * torch.rand_like(start_pos)
    end_pos = torch.where(active_mask, sampled_targets, start_pos)
    return end_pos, active_mask


def sample_duration_range(
    duration_min: torch.Tensor,
    duration_max: torch.Tensor,
) -> torch.Tensor:
    """Sample one segment duration per environment from `[min, max]`."""

    duration_max = torch.maximum(duration_max, duration_min + _EPS)
    duration_noise = torch.empty_like(duration_min).uniform_(0.0, 1.0)
    return duration_min + (duration_max - duration_min) * duration_noise


def sample_episode_disturbance(
    delay_max: int,
    noise_std_max: float,
    batch_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample delay and Gaussian noise magnitude for one curriculum level."""

    delay_steps = torch.randint(0, int(delay_max) + 1, size=(batch_size,), device=device, dtype=torch.int)
    noise_std = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(0.0, float(noise_std_max))
    return delay_steps, noise_std


def sample_episode_disturbance_per_env(
    delay_max: torch.Tensor,
    noise_std_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample delay and noise with per-environment upper bounds."""

    delay_max = torch.clamp(delay_max.to(dtype=torch.int), min=0)
    noise_std_max = torch.clamp(noise_std_max.to(dtype=torch.float32), min=0.0)
    delay_noise = torch.rand(delay_max.shape, device=delay_max.device, dtype=torch.float32)
    delay_steps = torch.floor(delay_noise * (delay_max.to(dtype=torch.float32) + 1.0)).to(dtype=torch.int)
    noise_std = torch.rand(noise_std_max.shape, device=noise_std_max.device, dtype=torch.float32) * noise_std_max
    return delay_steps, noise_std


def resolve_disturbance_reset_values(
    batch_size: int,
    device: torch.device | str,
    delay_range: tuple[int, int],
    noise_std_range: tuple[float, float],
    fixed_delay_steps: int | None,
    fixed_noise_std: float | None,
    has_curriculum_sample: torch.Tensor,
    curriculum_delay_steps: torch.Tensor,
    curriculum_noise_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve disturbance reset values with fixed > curriculum > random precedence."""

    if fixed_delay_steps is not None:
        delay_steps = torch.full((batch_size,), int(fixed_delay_steps), device=device, dtype=torch.int)
    else:
        delay_low, delay_high = delay_range
        delay_steps = torch.randint(delay_low, delay_high + 1, size=(batch_size,), device=device, dtype=torch.int)
        delay_steps = torch.where(has_curriculum_sample, curriculum_delay_steps.to(torch.int), delay_steps)

    if fixed_noise_std is not None:
        noise_std = torch.full((batch_size,), float(fixed_noise_std), device=device, dtype=torch.float32)
    else:
        noise_low, noise_high = noise_std_range
        noise_std = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(noise_low, noise_high)
        noise_std = torch.where(has_curriculum_sample, curriculum_noise_std.to(torch.float32), noise_std)

    return delay_steps, noise_std
