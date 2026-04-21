"""Shared metadata for the internalized SO101 follower robot."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from so101_hackathon.paths import ASSETS_ROOT


SO101_FOLLOWER_ASSET_PATH = (ASSETS_ROOT / "robots" / "so101_follower.usd").resolve()

SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_ARM_JOINT_NAMES: tuple[str, ...] = SO101_JOINT_NAMES[:-1]

SO101_BASE_BODY_NAME = "base"
SO101_EE_BODY_NAME = "gripper"
SO101_JAW_BODY_NAME = "jaw"
SO101_CONTACT_BODY_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "jaw",
)
SO101_CONTACT_SENSOR_BODY_NAMES: tuple[str, ...] = (
    "gripper",
    "jaw",
)

# Joint limits authored in the internalized USD asset, in degrees.
SO101_FOLLOWER_USD_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# Physical motor ranges exposed by the SO101 hardware interface.
SO101_FOLLOWER_MOTOR_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

SO101_FOLLOWER_REST_POSE_RANGE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-30.0, 30.0),
    "shoulder_lift": (-130.0, -70.0),
    "elbow_flex": (60.0, 120.0),
    "wrist_flex": (20.0, 80.0),
    "wrist_roll": (-30.0, 30.0),
    "gripper": (-40.0, 20.0),
}

# Legacy FK metadata retained for the flat teleop stack.
SO101_KINEMATIC_ARM_JOINTS: tuple[str, ...] = SO101_ARM_JOINT_NAMES
SO101_KINEMATICS_ORIGINS_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.0388353, -8.97657e-09, 0.0624),
    (-0.0303992, -0.0182778, -0.0542),
    (-0.11257, -0.028, 1.73763e-16),
    (-0.1349, 0.0052, 3.62355e-17),
    (5.55112e-17, -0.0611, 0.0181),
)
SO101_KINEMATICS_ORIGINS_RPY: tuple[tuple[float, float, float], ...] = (
    (3.14159, 4.18253e-17, -3.14159),
    (-1.5708, -1.5708, 0.0),
    (-3.63608e-16, 8.74301e-16, 1.5708),
    (4.02456e-15, 8.67362e-16, -1.5708),
    (1.5708, 0.0486795, 3.14159),
)


def degrees_to_radians(value_deg: float) -> float:
    return math.radians(float(value_deg))


def radians_to_degrees(value_rad: float) -> float:
    return math.degrees(float(value_rad))


def follower_joint_limits_rad_map() -> dict[str, tuple[float, float]]:
    return {
        joint_name: (
            degrees_to_radians(lower_deg),
            degrees_to_radians(upper_deg),
        )
        for joint_name, (lower_deg, upper_deg) in SO101_FOLLOWER_USD_JOINT_LIMITS_DEG.items()
    }


def follower_joint_limit_vectors_rad(
    joint_names: Iterable[str] = SO101_JOINT_NAMES,
) -> tuple[list[float], list[float]]:
    joint_limits = follower_joint_limits_rad_map()
    lower = [float(joint_limits[joint_name][0]) for joint_name in joint_names]
    upper = [float(joint_limits[joint_name][1]) for joint_name in joint_names]
    return lower, upper


def rest_pose_range_rad_map() -> dict[str, tuple[float, float]]:
    return {
        joint_name: (
            degrees_to_radians(lower_deg),
            degrees_to_radians(upper_deg),
        )
        for joint_name, (lower_deg, upper_deg) in SO101_FOLLOWER_REST_POSE_RANGE_DEG.items()
    }


def motor_value_to_joint_radians(
    joint_name: str,
    motor_value: float,
    *,
    motor_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    if joint_name not in SO101_FOLLOWER_USD_JOINT_LIMITS_DEG:
        raise KeyError(f"Unknown SO101 joint `{joint_name}`")
    motor_lower, motor_upper = (motor_limits or SO101_FOLLOWER_MOTOR_LIMITS)[joint_name]
    joint_lower_deg, joint_upper_deg = SO101_FOLLOWER_USD_JOINT_LIMITS_DEG[joint_name]
    bounded_value = min(float(motor_upper), max(float(motor_lower), float(motor_value)))
    motor_span = float(motor_upper) - float(motor_lower)
    if motor_span <= 0.0:
        raise ValueError(f"Invalid motor limits for `{joint_name}`: {motor_lower}, {motor_upper}")
    alpha = (bounded_value - float(motor_lower)) / motor_span
    joint_value_deg = float(joint_lower_deg) + alpha * (float(joint_upper_deg) - float(joint_lower_deg))
    return degrees_to_radians(joint_value_deg)


def joint_radians_to_motor_value(
    joint_name: str,
    joint_value_rad: float,
    *,
    motor_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    if joint_name not in SO101_FOLLOWER_USD_JOINT_LIMITS_DEG:
        raise KeyError(f"Unknown SO101 joint `{joint_name}`")
    motor_lower, motor_upper = (motor_limits or SO101_FOLLOWER_MOTOR_LIMITS)[joint_name]
    joint_lower_deg, joint_upper_deg = SO101_FOLLOWER_USD_JOINT_LIMITS_DEG[joint_name]
    joint_value_deg = radians_to_degrees(joint_value_rad)
    bounded_joint = min(float(joint_upper_deg), max(float(joint_lower_deg), float(joint_value_deg)))
    joint_span = float(joint_upper_deg) - float(joint_lower_deg)
    if joint_span <= 0.0:
        raise ValueError(f"Invalid joint limits for `{joint_name}`: {joint_lower_deg}, {joint_upper_deg}")
    alpha = (bounded_joint - float(joint_lower_deg)) / joint_span
    return float(motor_lower) + alpha * (float(motor_upper) - float(motor_lower))


def convert_motor_observation_to_joint_positions(
    joint_state: dict[str, float],
    *,
    joint_names: Iterable[str] = SO101_JOINT_NAMES,
    motor_limits: dict[str, tuple[float, float]] | None = None,
) -> list[float]:
    return [
        motor_value_to_joint_radians(joint_name, float(joint_state[joint_name]), motor_limits=motor_limits)
        for joint_name in joint_names
    ]
