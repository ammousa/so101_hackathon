"""Reward helpers for SO101 teleoperation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .observations import command_joint_positions, command_joint_velocities

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _split_command(command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = command.shape[-1] // 2
    return command[:, :half], command[:, half:]


def joint_position_tracking_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "leader_joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command_pos = command_joint_positions(env, command_name=command_name)
    return -torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - command_pos), dim=-1)


def joint_velocity_tracking_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "leader_joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command_vel = command_joint_velocities(env, command_name=command_name)
    return -torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] - command_vel), dim=-1)


def joint_acceleration_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)
