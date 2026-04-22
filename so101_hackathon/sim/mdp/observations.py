"""Custom observation helpers for SO101 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach import mdp as reach_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

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
    target_velocities = command_joint_velocities(
        env, command_name=command_name)
    return target_velocities - asset.data.joint_vel[:, asset_cfg.joint_ids]


def applied_action(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    """Return the latest applied joint command for the configured action term."""
    return env.action_manager.get_term(action_name).applied_actions


def controller_action(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    """Return the latest pre-disturbance controller command."""
    action_term = env.action_manager.get_term(action_name)
    if hasattr(action_term, "controller_actions"):
        return action_term.controller_actions
    return action_term.applied_actions


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the end-effector frame state in robot coordinates."""
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)


def joint_pos_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the joint position targets for the configured asset joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]
