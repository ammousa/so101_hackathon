"""Termination helpers for SO101 teleoperation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_error_too_large(
    env: "ManagerBasedRLEnv",
    command_name: str = "leader_joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_abs_error: float = 0.75,
) -> torch.Tensor:
    """Terminate when any controlled joint tracking error exceeds the configured threshold."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_positions = command[:, : command.shape[-1] // 2]
    joint_error = torch.abs(target_positions - asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.any(joint_error > max_abs_error, dim=-1)


def joint_limit_violation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_tolerance: float = 0.02,
) -> torch.Tensor:
    """Terminate when any controlled joint drifts outside the soft joint limits by a tolerance."""
    asset = env.scene[asset_cfg.name]
    lower_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] - position_tolerance
    upper_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1] + position_tolerance
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.any((joint_pos < lower_limits) | (joint_pos > upper_limits), dim=-1)


def unstable_joint_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_velocity: float = 2.0,
) -> torch.Tensor:
    """Terminate on excessive joint velocity or invalid numeric state."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    invalid_state = ~torch.isfinite(joint_vel).all(dim=-1) | ~torch.isfinite(asset.data.joint_pos[:, asset_cfg.joint_ids]).all(dim=-1)
    return torch.any(torch.abs(joint_vel) > max_velocity, dim=-1) | invalid_state
