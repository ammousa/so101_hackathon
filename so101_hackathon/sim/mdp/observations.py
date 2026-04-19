"""Custom observation helpers for SO101 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_tasks.manager_based.manipulation.reach import mdp as reach_mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_position(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """Return only xyz target position from a pose command."""
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "target_ee_position"):
        return command_term.target_ee_position
    return reach_mdp.generated_commands(env, command_name=command_name)[:, :3]


def command_joint_positions(env: "ManagerBasedRLEnv", command_name: str = "leader_joints") -> torch.Tensor:
    """Return target joint positions from a teleop joint command."""
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "target_joint_positions"):
        return command_term.target_joint_positions
    command = env.command_manager.get_command(command_name)
    half = command.shape[-1] // 2
    return command[:, :half]


def command_joint_velocities(env: "ManagerBasedRLEnv", command_name: str = "leader_joints") -> torch.Tensor:
    """Return target joint velocities from a teleop joint command."""
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "target_joint_velocities"):
        return command_term.target_joint_velocities
    command = env.command_manager.get_command(command_name)
    half = command.shape[-1] // 2
    return command[:, half:]


def joint_tracking_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "leader_joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return target minus actual joint positions for the configured joints."""
    asset = env.scene[asset_cfg.name]
    target_positions = command_joint_positions(env, command_name=command_name)
    return target_positions - asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_velocity_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "leader_joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return target minus actual joint velocities for the configured joints."""
    asset = env.scene[asset_cfg.name]
    target_velocities = command_joint_velocities(env, command_name=command_name)
    return target_velocities - asset.data.joint_vel[:, asset_cfg.joint_ids]


def applied_action(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    """Return the latest applied joint command for the configured action term."""
    return env.action_manager.get_term(action_name).applied_actions
