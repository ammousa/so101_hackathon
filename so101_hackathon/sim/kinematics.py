"""Lightweight URDF kinematics helpers used by teleop evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import math
import xml.etree.ElementTree as ET

import torch


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
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


def _rotation_about_axis(axis: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.norm(axis)
    x, y, z = axis.tolist()
    c = torch.cos(angles)
    s = torch.sin(angles)
    one_c = 1.0 - c
    out = torch.zeros((angles.shape[0], 3, 3), dtype=angles.dtype, device=angles.device)
    out[:, 0, 0] = c + x * x * one_c
    out[:, 0, 1] = x * y * one_c - z * s
    out[:, 0, 2] = x * z * one_c + y * s
    out[:, 1, 0] = y * x * one_c + z * s
    out[:, 1, 1] = c + y * y * one_c
    out[:, 1, 2] = y * z * one_c - x * s
    out[:, 2, 0] = z * x * one_c - y * s
    out[:, 2, 1] = z * y * one_c + x * s
    out[:, 2, 2] = c + z * z * one_c
    return out


def _quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to quaternions in (w, x, y, z)."""
    m = matrix
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = torch.zeros((m.shape[0], 4), dtype=m.dtype, device=m.device)

    positive_trace = trace > 0.0
    if positive_trace.any():
        s = torch.sqrt(trace[positive_trace] + 1.0) * 2.0
        quat[positive_trace, 0] = 0.25 * s
        quat[positive_trace, 1] = (m[positive_trace, 2, 1] - m[positive_trace, 1, 2]) / s
        quat[positive_trace, 2] = (m[positive_trace, 0, 2] - m[positive_trace, 2, 0]) / s
        quat[positive_trace, 3] = (m[positive_trace, 1, 0] - m[positive_trace, 0, 1]) / s

    mask_x = (~positive_trace) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if mask_x.any():
        s = torch.sqrt(1.0 + m[mask_x, 0, 0] - m[mask_x, 1, 1] - m[mask_x, 2, 2]) * 2.0
        quat[mask_x, 0] = (m[mask_x, 2, 1] - m[mask_x, 1, 2]) / s
        quat[mask_x, 1] = 0.25 * s
        quat[mask_x, 2] = (m[mask_x, 0, 1] + m[mask_x, 1, 0]) / s
        quat[mask_x, 3] = (m[mask_x, 0, 2] + m[mask_x, 2, 0]) / s

    mask_y = (~positive_trace) & (~mask_x) & (m[:, 1, 1] > m[:, 2, 2])
    if mask_y.any():
        s = torch.sqrt(1.0 + m[mask_y, 1, 1] - m[mask_y, 0, 0] - m[mask_y, 2, 2]) * 2.0
        quat[mask_y, 0] = (m[mask_y, 0, 2] - m[mask_y, 2, 0]) / s
        quat[mask_y, 1] = (m[mask_y, 0, 1] + m[mask_y, 1, 0]) / s
        quat[mask_y, 2] = 0.25 * s
        quat[mask_y, 3] = (m[mask_y, 1, 2] + m[mask_y, 2, 1]) / s

    mask_z = (~positive_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + m[mask_z, 2, 2] - m[mask_z, 0, 0] - m[mask_z, 1, 1]) * 2.0
        quat[mask_z, 0] = (m[mask_z, 1, 0] - m[mask_z, 0, 1]) / s
        quat[mask_z, 1] = (m[mask_z, 0, 2] + m[mask_z, 2, 0]) / s
        quat[mask_z, 2] = (m[mask_z, 1, 2] + m[mask_z, 2, 1]) / s
        quat[mask_z, 3] = 0.25 * s

    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)
    return quat


@dataclass(frozen=True)
class JointTransform:
    name: str
    joint_type: str
    axis: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]


class UrdfKinematicChain:
    """Forward-kinematics chain built from a URDF path."""

    def __init__(self, urdf_path: str | Path, end_link: str):
        self.urdf_path = Path(urdf_path)
        self.end_link = end_link
        self.joints = self._load_chain(self.urdf_path, end_link)
        self.joint_names = [joint.name for joint in self.joints if joint.joint_type != "fixed"]
        self._joint_name_to_index = {name: idx for idx, name in enumerate(self.joint_names)}

    def forward_kinematics(
        self,
        joint_positions: torch.Tensor,
        joint_names: Iterable[str] | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Return end-link xyz positions for the provided joint batch."""
        if joint_positions.ndim != 2:
            raise ValueError(f"Expected joint_positions shape (N, D), received {tuple(joint_positions.shape)}")

        if joint_names is None:
            if joint_positions.shape[1] != len(self.joint_names):
                raise ValueError(
                    f"Expected {len(self.joint_names)} joint values, received {joint_positions.shape[1]}"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(self.joint_names)}
        else:
            names = list(joint_names)
            if joint_positions.shape[1] != len(names):
                raise ValueError(
                    f"Joint name count ({len(names)}) does not match tensor width ({joint_positions.shape[1]})"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(names)}

        device = device or joint_positions.device
        batch = joint_positions.shape[0]
        transform = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
        zeros = torch.zeros(batch, dtype=joint_positions.dtype, device=device)

        for joint in self.joints:
            origin = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            origin[:, :3, :3] = _rpy_to_matrix(*joint.origin_rpy).to(device=device, dtype=joint_positions.dtype)
            origin[:, :3, 3] = torch.tensor(joint.origin_xyz, dtype=joint_positions.dtype, device=device)
            transform = torch.bmm(transform, origin)

            if joint.joint_type == "fixed":
                continue

            angle = value_by_name.get(joint.name, zeros)
            rot = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            rot[:, :3, :3] = _rotation_about_axis(
                torch.tensor(joint.axis, dtype=joint_positions.dtype, device=device), angle
            )
            transform = torch.bmm(transform, rot)

        return transform[:, :3, 3]

    def forward_pose(
        self,
        joint_positions: torch.Tensor,
        joint_names: Iterable[str] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return end-link position and orientation quaternion in the base frame."""
        if joint_positions.ndim != 2:
            raise ValueError(f"Expected joint_positions shape (N, D), received {tuple(joint_positions.shape)}")

        if joint_names is None:
            if joint_positions.shape[1] != len(self.joint_names):
                raise ValueError(
                    f"Expected {len(self.joint_names)} joint values, received {joint_positions.shape[1]}"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(self.joint_names)}
        else:
            names = list(joint_names)
            if joint_positions.shape[1] != len(names):
                raise ValueError(
                    f"Joint name count ({len(names)}) does not match tensor width ({joint_positions.shape[1]})"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(names)}

        device = device or joint_positions.device
        batch = joint_positions.shape[0]
        transform = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
        zeros = torch.zeros(batch, dtype=joint_positions.dtype, device=device)

        for joint in self.joints:
            origin = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            origin[:, :3, :3] = _rpy_to_matrix(*joint.origin_rpy).to(device=device, dtype=joint_positions.dtype)
            origin[:, :3, 3] = torch.tensor(joint.origin_xyz, dtype=joint_positions.dtype, device=device)
            transform = torch.bmm(transform, origin)

            if joint.joint_type == "fixed":
                continue

            angle = value_by_name.get(joint.name, zeros)
            rot = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            rot[:, :3, :3] = _rotation_about_axis(
                torch.tensor(joint.axis, dtype=joint_positions.dtype, device=device), angle
            )
            transform = torch.bmm(transform, rot)

        return transform[:, :3, 3], _quat_from_matrix(transform[:, :3, :3])

    def forward_chain_points(
        self,
        joint_positions: torch.Tensor,
        joint_names: Iterable[str] | None = None,
        device: str | torch.device | None = None,
        include_end_link: bool = True,
    ) -> torch.Tensor:
        """Return joint-origin positions, and optionally the end-link position, in the base frame."""
        if joint_positions.ndim != 2:
            raise ValueError(f"Expected joint_positions shape (N, D), received {tuple(joint_positions.shape)}")

        if joint_names is None:
            if joint_positions.shape[1] != len(self.joint_names):
                raise ValueError(
                    f"Expected {len(self.joint_names)} joint values, received {joint_positions.shape[1]}"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(self.joint_names)}
        else:
            names = list(joint_names)
            if joint_positions.shape[1] != len(names):
                raise ValueError(
                    f"Joint name count ({len(names)}) does not match tensor width ({joint_positions.shape[1]})"
                )
            value_by_name = {name: joint_positions[:, idx] for idx, name in enumerate(names)}

        device = device or joint_positions.device
        batch = joint_positions.shape[0]
        transform = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
        zeros = torch.zeros(batch, dtype=joint_positions.dtype, device=device)
        points: list[torch.Tensor] = []

        for joint in self.joints:
            origin = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            origin[:, :3, :3] = _rpy_to_matrix(*joint.origin_rpy).to(device=device, dtype=joint_positions.dtype)
            origin[:, :3, 3] = torch.tensor(joint.origin_xyz, dtype=joint_positions.dtype, device=device)
            transform = torch.bmm(transform, origin)

            if joint.joint_type == "fixed":
                continue

            points.append(transform[:, :3, 3].clone())
            angle = value_by_name.get(joint.name, zeros)
            rot = torch.eye(4, dtype=joint_positions.dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
            rot[:, :3, :3] = _rotation_about_axis(
                torch.tensor(joint.axis, dtype=joint_positions.dtype, device=device), angle
            )
            transform = torch.bmm(transform, rot)

        if include_end_link:
            points.append(transform[:, :3, 3].clone())

        return torch.stack(points, dim=1)

    @staticmethod
    def _load_chain(urdf_path: Path, end_link: str) -> list[JointTransform]:
        root = ET.fromstring(urdf_path.read_text())
        child_to_joint = {}
        child_to_parent = {}
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            parent = joint.find("parent").get("link")
            child_to_joint[child] = joint
            child_to_parent[child] = parent

        if end_link not in child_to_parent:
            raise KeyError(f"Link '{end_link}' is not reachable from URDF joints in {urdf_path}")

        chain_xml = []
        cursor = end_link
        while cursor in child_to_parent:
            chain_xml.append(child_to_joint[cursor])
            cursor = child_to_parent[cursor]
        chain_xml.reverse()

        chain = []
        for joint in chain_xml:
            origin = joint.find("origin")
            axis = joint.find("axis")
            origin_xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
            origin_rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
            joint_axis = tuple(float(v) for v in axis.get("xyz", "0 0 1").split()) if axis is not None else (0.0, 0.0, 1.0)
            chain.append(
                JointTransform(
                    name=joint.get("name"),
                    joint_type=joint.get("type", "fixed"),
                    axis=joint_axis,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                )
            )
        return chain


@lru_cache(maxsize=1)
def so101_gripper_kinematics() -> UrdfKinematicChain:
    urdf_path = (
        Path(__file__).resolve().parent
        / "robots"
        / "trs_so101"
        / "urdf"
        / "so_arm101.urdf"
    )
    return UrdfKinematicChain(urdf_path=urdf_path, end_link="gripper_link")


def compute_so101_ee_position(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> torch.Tensor:
    """Return SO101 gripper xyz positions for batched joint positions."""
    return so101_gripper_kinematics().forward_kinematics(joint_positions, joint_names=joint_names)


def compute_so101_ee_pose(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SO101 gripper xyz position and quaternion for batched joint positions."""
    return so101_gripper_kinematics().forward_pose(joint_positions, joint_names=joint_names)


def compute_so101_chain_points(
    joint_positions: torch.Tensor,
    joint_names: Iterable[str],
) -> torch.Tensor:
    """Return SO101 joint-origin and end-effector positions in the base frame."""
    return so101_gripper_kinematics().forward_chain_points(joint_positions, joint_names=joint_names)
