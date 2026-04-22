"""Hand-written SO101 forward kinematics and Jacobian helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from so101_hackathon.sim.robots.so101_follower_spec import (
    SO101_ARM_JOINT_NAMES,
    SO101_KINEMATIC_ARM_JOINTS,
    SO101_KINEMATICS_ORIGINS_RPY,
    SO101_KINEMATICS_ORIGINS_XYZ,
)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Handle rpy to matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )


def _rotation_about_z(angles: torch.Tensor) -> torch.Tensor:
    """Handle rotation about z."""
    cos_angle = torch.cos(angles)
    sin_angle = torch.sin(angles)
    rotation = torch.zeros((angles.shape[0], 3, 3), dtype=angles.dtype, device=angles.device)
    rotation[:, 0, 0] = cos_angle
    rotation[:, 0, 1] = -sin_angle
    rotation[:, 1, 0] = sin_angle
    rotation[:, 1, 1] = cos_angle
    rotation[:, 2, 2] = 1.0
    return rotation


def _quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to quaternions in (w, x, y, z)."""
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    quat = torch.zeros((matrix.shape[0], 4), dtype=matrix.dtype, device=matrix.device)

    positive_trace = trace > 0.0
    if positive_trace.any():
        s = torch.sqrt(trace[positive_trace] + 1.0) * 2.0
        quat[positive_trace, 0] = 0.25 * s
        quat[positive_trace, 1] = (matrix[positive_trace, 2, 1] - matrix[positive_trace, 1, 2]) / s
        quat[positive_trace, 2] = (matrix[positive_trace, 0, 2] - matrix[positive_trace, 2, 0]) / s
        quat[positive_trace, 3] = (matrix[positive_trace, 1, 0] - matrix[positive_trace, 0, 1]) / s

    mask_x = (~positive_trace) & (matrix[:, 0, 0] > matrix[:, 1, 1]) & (matrix[:, 0, 0] > matrix[:, 2, 2])
    if mask_x.any():
        s = torch.sqrt(1.0 + matrix[mask_x, 0, 0] - matrix[mask_x, 1, 1] - matrix[mask_x, 2, 2]) * 2.0
        quat[mask_x, 0] = (matrix[mask_x, 2, 1] - matrix[mask_x, 1, 2]) / s
        quat[mask_x, 1] = 0.25 * s
        quat[mask_x, 2] = (matrix[mask_x, 0, 1] + matrix[mask_x, 1, 0]) / s
        quat[mask_x, 3] = (matrix[mask_x, 0, 2] + matrix[mask_x, 2, 0]) / s

    mask_y = (~positive_trace) & (~mask_x) & (matrix[:, 1, 1] > matrix[:, 2, 2])
    if mask_y.any():
        s = torch.sqrt(1.0 + matrix[mask_y, 1, 1] - matrix[mask_y, 0, 0] - matrix[mask_y, 2, 2]) * 2.0
        quat[mask_y, 0] = (matrix[mask_y, 0, 2] - matrix[mask_y, 2, 0]) / s
        quat[mask_y, 1] = (matrix[mask_y, 0, 1] + matrix[mask_y, 1, 0]) / s
        quat[mask_y, 2] = 0.25 * s
        quat[mask_y, 3] = (matrix[mask_y, 1, 2] + matrix[mask_y, 2, 1]) / s

    mask_z = (~positive_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + matrix[mask_z, 2, 2] - matrix[mask_z, 0, 0] - matrix[mask_z, 1, 1]) * 2.0
        quat[mask_z, 0] = (matrix[mask_z, 1, 0] - matrix[mask_z, 0, 1]) / s
        quat[mask_z, 1] = (matrix[mask_z, 0, 2] + matrix[mask_z, 2, 0]) / s
        quat[mask_z, 2] = (matrix[mask_z, 1, 2] + matrix[mask_z, 2, 1]) / s
        quat[mask_z, 3] = 0.25 * s

    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)


_SO101_ARM_JOINTS: tuple[str, ...] = SO101_KINEMATIC_ARM_JOINTS
_SO101_LOCAL_Z_AXIS = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
_SO101_ORIGIN_ROTATIONS = tuple(_rpy_to_matrix(*rpy) for rpy in SO101_KINEMATICS_ORIGINS_RPY)
_SO101_ORIGIN_TRANSLATIONS = tuple(torch.tensor(xyz, dtype=torch.float32) for xyz in SO101_KINEMATICS_ORIGINS_XYZ)


def _prepare_arm_joint_positions(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> tuple[torch.Tensor, list[str]]:
    """Handle prepare arm joint positions."""
    if joint_positions.ndim != 2:
        raise ValueError(f"Expected joint_positions shape (N, D), received {tuple(joint_positions.shape)}")

    names = list(joint_names)
    if joint_positions.shape[1] != len(names):
        raise ValueError(
            f"Joint name count ({len(names)}) does not match tensor width ({joint_positions.shape[1]})"
        )

    arm_joint_pos = torch.zeros(
        (joint_positions.shape[0], len(_SO101_ARM_JOINTS)),
        dtype=joint_positions.dtype,
        device=joint_positions.device,
    )
    for index, name in enumerate(names):
        if name in _SO101_ARM_JOINTS:
            arm_joint_pos[:, _SO101_ARM_JOINTS.index(name)] = joint_positions[:, index]
    return arm_joint_pos, names


def _forward_so101_arm(
    arm_joint_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Handle forward so101 arm."""
    batch_size = arm_joint_positions.shape[0]
    rotation = torch.eye(3, dtype=arm_joint_positions.dtype, device=arm_joint_positions.device)
    rotation = rotation.unsqueeze(0).repeat(batch_size, 1, 1)
    position = torch.zeros((batch_size, 3), dtype=arm_joint_positions.dtype, device=arm_joint_positions.device)
    joint_origins = []
    joint_axes = []

    for joint_index, joint_angles in enumerate(torch.unbind(arm_joint_positions, dim=1)):
        origin_rotation = _SO101_ORIGIN_ROTATIONS[joint_index].to(
            device=arm_joint_positions.device, dtype=arm_joint_positions.dtype
        )
        origin_translation = _SO101_ORIGIN_TRANSLATIONS[joint_index].to(
            device=arm_joint_positions.device, dtype=arm_joint_positions.dtype
        )

        position = position + torch.matmul(rotation, origin_translation)
        rotation = torch.matmul(rotation, origin_rotation.unsqueeze(0).expand(batch_size, -1, -1))
        joint_origins.append(position)
        joint_axes.append(torch.matmul(rotation, _SO101_LOCAL_Z_AXIS.to(
            device=arm_joint_positions.device, dtype=arm_joint_positions.dtype)))
        rotation = torch.matmul(rotation, _rotation_about_z(joint_angles))

    return position, rotation, torch.stack(joint_origins, dim=1), torch.stack(joint_axes, dim=1)


def compute_so101_ee_position(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> torch.Tensor:
    """Return SO101 gripper-link xyz positions for batched joint positions."""
    arm_joint_positions, _ = _prepare_arm_joint_positions(joint_positions, joint_names)
    position, _, _, _ = _forward_so101_arm(arm_joint_positions)
    return position


def compute_so101_ee_pose(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SO101 gripper-link xyz position and quaternion."""
    arm_joint_positions, _ = _prepare_arm_joint_positions(joint_positions, joint_names)
    position, rotation, _, _ = _forward_so101_arm(arm_joint_positions)
    return position, _quat_from_matrix(rotation)


def compute_so101_chain_points(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> torch.Tensor:
    """Return SO101 joint-origin points followed by the gripper-link point."""
    arm_joint_positions, _ = _prepare_arm_joint_positions(joint_positions, joint_names)
    position, _, joint_origins, _ = _forward_so101_arm(arm_joint_positions)
    return torch.cat((joint_origins, position.unsqueeze(1)), dim=1)


def compute_so101_ee_jacobian(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> torch.Tensor:
    """Return the analytic gripper-link position Jacobian for the provided joint order."""
    arm_joint_positions, names = _prepare_arm_joint_positions(joint_positions, joint_names)
    ee_position, _, joint_origins, joint_axes = _forward_so101_arm(arm_joint_positions)
    arm_jacobian = torch.cross(joint_axes, ee_position.unsqueeze(1) - joint_origins, dim=-1)
    arm_jacobian = torch.transpose(arm_jacobian, dim0=1, dim1=2)

    jacobian = torch.zeros(
        (joint_positions.shape[0], 3, len(names)),
        dtype=joint_positions.dtype,
        device=joint_positions.device,
    )
    for output_index, name in enumerate(names):
        if name in _SO101_ARM_JOINTS:
            jacobian[:, :, output_index] = arm_jacobian[:, :, _SO101_ARM_JOINTS.index(name)]
    return jacobian
