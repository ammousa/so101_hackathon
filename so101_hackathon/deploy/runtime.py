"""Pure helpers shared by teleop deploy code and tests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

from so101_hackathon.sim.robots.so101_follower_spec import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    follower_joint_limit_vectors_rad,
    follower_joint_limits_rad_map,
    joint_radians_to_motor_value,
    motor_value_to_joint_radians,
)
from so101_hackathon.utils.eval_utils import checkpoint_run_dir
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES

DEFAULT_FPS = 60
DEFAULT_TELEOP_TIME_S = None
DEFAULT_FOLLOWER_PORT = "/dev/ttyACM0"
DEFAULT_FOLLOWER_ID = "my_awesome_follower_arm"
DEFAULT_LEADER_PORT = "/dev/ttyACM1"
DEFAULT_LEADER_ID = "my_awesome_leader_arm"
DEFAULT_PRINT_EVERY = 20
DEFAULT_DELAY_STEPS = 0
DEFAULT_NOISE_STD = 0.0
DEFAULT_NOISE_JOINT_INDICES = (0, 1, 2, 3)

_JOINT_KEY_SUFFIX = ".pos"
_GRIPPER_JOINT_NAME = "gripper"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def joint_field_name(joint_name: str) -> str:
    return f"{joint_name}{_JOINT_KEY_SUFFIX}"


def _gripper_joint_limits_rad() -> tuple[float, float]:
    joint_limits = parse_joint_limits_from_urdf()
    lower, upper = joint_limits[_GRIPPER_JOINT_NAME]
    return float(lower), float(upper)


def _gripper_percent_to_radians(value_percent: float) -> float:
    lower, upper = _gripper_joint_limits_rad()
    bounded_value = min(100.0, max(0.0, float(value_percent)))
    alpha = bounded_value / 100.0
    return lower + alpha * (upper - lower)


def _gripper_radians_to_percent(value_rad: float) -> float:
    lower, upper = _gripper_joint_limits_rad()
    if upper <= lower:
        raise ValueError(
            f"Invalid gripper joint limits: lower={lower}, upper={upper}")
    bounded_value = min(upper, max(lower, float(value_rad)))
    alpha = (bounded_value - lower) / (upper - lower)
    return 100.0 * alpha


def _coerce_vector(values: Any, *, joint_dim: int = len(TELEOP_JOINT_NAMES)) -> list[float]:
    if isinstance(values, list):
        if len(values) != joint_dim:
            raise ValueError(
                f"Expected {joint_dim} values, received {len(values)}")
        return [float(value) for value in values]
    if hasattr(values, "detach") and hasattr(values, "cpu") and hasattr(values, "tolist"):
        data = values.detach().cpu().tolist()
        if isinstance(data, list) and data and isinstance(data[0], list):
            if len(data) != 1:
                raise ValueError(
                    f"Expected a single action vector, received batch shape {len(data)}")
            data = data[0]
        if len(data) != joint_dim:
            raise ValueError(
                f"Expected {joint_dim} values, received {len(data)}")
        return [float(value) for value in data]
    data = list(values)
    if len(data) != joint_dim:
        raise ValueError(f"Expected {joint_dim} values, received {len(data)}")
    return [float(value) for value in data]


def degrees_to_radians(values: Iterable[float]) -> list[float]:
    return [math.radians(float(value)) for value in values]


def radians_to_degrees(values: Iterable[float]) -> list[float]:
    return [math.degrees(float(value)) for value in values]


def hardware_obs_to_joint_positions(
    observation: dict[str, float],
    *,
    allowed_missing_joint_names: Iterable[str] = (),
    fallback_joint_positions_rad: Iterable[float] | None = None,
) -> list[float]:
    allowed_missing = set(allowed_missing_joint_names)
    fallback_joint_positions = (
        _coerce_vector(
            fallback_joint_positions_rad) if fallback_joint_positions_rad is not None else None
    )
    values: list[float] = []
    for index, joint_name in enumerate(TELEOP_JOINT_NAMES):
        field_name = joint_field_name(joint_name)
        if field_name in observation:
            raw_value = float(observation[field_name])
            values.append(motor_value_to_joint_radians(joint_name, raw_value))
            continue
        if joint_name in allowed_missing and fallback_joint_positions is not None:
            values.append(float(fallback_joint_positions[index]))
            continue
        raise KeyError(f"Missing joint observation field `{field_name}`")
    return values


def build_follower_action(
    joint_positions_rad: Iterable[float],
    *,
    active_joint_names: Iterable[str] | None = None,
) -> dict[str, float]:
    joint_positions = _coerce_vector(joint_positions_rad)
    active_joint_name_set = set(
        active_joint_names) if active_joint_names is not None else set(TELEOP_JOINT_NAMES)
    action: dict[str, float] = {}
    for index, joint_name in enumerate(TELEOP_JOINT_NAMES):
        if joint_name not in active_joint_name_set:
            continue
        value_rad = float(joint_positions[index])
        action[joint_field_name(joint_name)] = joint_radians_to_motor_value(
            joint_name, value_rad)
    return action


def normalize_controller_action(action: Any) -> list[float]:
    return _coerce_vector(action)


def blend_with_leader(
    leader_joint_pos: Iterable[float],
    controller_joint_pos: Iterable[float],
    coeff: float,
) -> list[float]:
    if coeff < 0.0 or coeff > 1.0:
        raise ValueError(
            f"controller coefficient must be within [0, 1], received {coeff}")
    leader = _coerce_vector(leader_joint_pos)
    controller = _coerce_vector(controller_joint_pos)
    return [
        float(leader_value) + float(coeff) *
        (float(controller_value) - float(leader_value))
        for leader_value, controller_value in zip(leader, controller)
    ]


def clamp_joint_positions(
    joint_positions: Iterable[float],
    lower_limits: Iterable[float],
    upper_limits: Iterable[float],
) -> list[float]:
    positions = _coerce_vector(joint_positions)
    lower = _coerce_vector(lower_limits)
    upper = _coerce_vector(upper_limits)
    return [
        max(float(lower_value), min(float(upper_value), float(position)))
        for position, lower_value, upper_value in zip(positions, lower, upper)
    ]


@dataclass
class FixedDisturbanceChannel:
    """Apply fixed command delay and masked Gaussian noise in joint space."""

    delay_steps: int = DEFAULT_DELAY_STEPS
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = 0
    noise_joint_indices: tuple[int, ...] = DEFAULT_NOISE_JOINT_INDICES

    def __post_init__(self) -> None:
        if self.delay_steps < 0:
            raise ValueError(
                f"delay_steps must be >= 0, received {self.delay_steps}")
        if self.noise_std < 0.0:
            raise ValueError(
                f"noise_std must be >= 0.0, received {self.noise_std}")
        self.noise_joint_indices = tuple(int(index)
                                         for index in self.noise_joint_indices)
        self._buffer: list[list[float]] = []
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        self._buffer = []
        self._rng = random.Random(self.seed)

    def apply(self, joint_positions: Iterable[float]) -> list[float]:
        command = _coerce_vector(joint_positions)
        self._buffer.append(list(command))
        if len(self._buffer) <= self.delay_steps:
            delayed = list(self._buffer[0])
        else:
            delayed = list(self._buffer[-(self.delay_steps + 1)])
        if self.noise_std <= 0.0:
            return delayed
        disturbed = list(delayed)
        for joint_index in self.noise_joint_indices:
            if 0 <= joint_index < len(disturbed):
                disturbed[joint_index] = float(
                    disturbed[joint_index]) + self._rng.gauss(0.0, self.noise_std)
        return disturbed


def parse_joint_limits_from_urdf(urdf_path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    del urdf_path
    return follower_joint_limits_rad_map()


def get_joint_limit_vectors() -> tuple[list[float], list[float]]:
    return follower_joint_limit_vectors_rad(TELEOP_JOINT_NAMES)


@dataclass
class LiveTeleopObservation:
    observation: list[float]
    leader_joint_pos: list[float]
    follower_joint_pos: list[float]
    leader_joint_vel: list[float]
    joint_error: list[float]
    joint_error_vel: list[float]
    previous_action: list[float]


class LiveTeleopObservationBuilder:
    """Reconstruct the flat teleop observation used during hackathon evaluation."""

    def __init__(self, *, missing_follower_joint_names: Iterable[str] = ()):
        self._previous_leader_joint_pos: list[float] | None = None
        self._previous_joint_error: list[float] | None = None
        self._previous_action: list[float] = [0.0] * len(TELEOP_JOINT_NAMES)
        self._missing_follower_joint_names = set(missing_follower_joint_names)

    def reset(self) -> None:
        self._previous_leader_joint_pos = None
        self._previous_joint_error = None
        self._previous_action = [0.0] * len(TELEOP_JOINT_NAMES)

    def set_previous_action(self, action: Iterable[float]) -> None:
        self._previous_action = _coerce_vector(action)

    def build(
        self,
        *,
        leader_observation: dict[str, float],
        follower_observation: dict[str, float],
        dt: float,
    ) -> LiveTeleopObservation:
        leader_joint_pos = hardware_obs_to_joint_positions(leader_observation)
        follower_joint_pos = hardware_obs_to_joint_positions(
            follower_observation,
            allowed_missing_joint_names=self._missing_follower_joint_names,
            fallback_joint_positions_rad=leader_joint_pos,
        )
        joint_error = [
            float(leader_value) - float(follower_value)
            for leader_value, follower_value in zip(leader_joint_pos, follower_joint_pos)
        ]
        if self._previous_leader_joint_pos is None or dt <= 0.0:
            leader_joint_vel = [0.0] * len(TELEOP_JOINT_NAMES)
            joint_error_vel = [0.0] * len(TELEOP_JOINT_NAMES)
        else:
            leader_joint_vel = [
                (float(current) - float(previous)) / dt
                for current, previous in zip(leader_joint_pos, self._previous_leader_joint_pos)
            ]
            joint_error_vel = [
                (float(current) - float(previous)) / dt
                for current, previous in zip(joint_error, self._previous_joint_error or joint_error)
            ]

        observation = (
            list(leader_joint_pos)
            + list(leader_joint_vel)
            + list(joint_error)
            + list(joint_error_vel)
            + list(self._previous_action)
        )
        self._previous_leader_joint_pos = list(leader_joint_pos)
        self._previous_joint_error = list(joint_error)
        return LiveTeleopObservation(
            observation=observation,
            leader_joint_pos=list(leader_joint_pos),
            follower_joint_pos=list(follower_joint_pos),
            leader_joint_vel=list(leader_joint_vel),
            joint_error=list(joint_error),
            joint_error_vel=list(joint_error_vel),
            previous_action=list(self._previous_action),
        )


def resolve_deploy_output_dir(
    *,
    controller_name: str,
    requested_output_dir: str | None,
    checkpoint_path: str | None,
) -> str:
    if requested_output_dir:
        return os.path.abspath(requested_output_dir)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if controller_name == "ppo":
        run_dir = checkpoint_run_dir(checkpoint_path)
        if run_dir is not None:
            return os.path.join(run_dir, "deploy", stamp)
    return os.path.abspath(os.path.join("logs", controller_name, "deploy", stamp))


def build_deploy_config(
    *,
    args: Any,
    controller_name: str,
    controller_config: dict[str, Any],
    checkpoint_path: str | None,
    output_dir: str,
    lower_limits: Iterable[float],
    upper_limits: Iterable[float],
) -> dict[str, Any]:
    return {
        "controller": controller_name,
        "args": vars(args).copy(),
        "controller_config": dict(controller_config),
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "joint_names": list(TELEOP_JOINT_NAMES),
        "joint_lower_limits_rad": _coerce_vector(lower_limits),
        "joint_upper_limits_rad": _coerce_vector(upper_limits),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_timeseries_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    path = Path(path)
    field_names = sorted(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_deploy_artifacts(
    *,
    output_dir: str | Path,
    config_payload: dict[str, Any],
    metrics: Any,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    artifact_paths = {
        "config": str(output_dir / "config.json"),
        "summary": str(output_dir / "summary.json"),
        "timeseries": str(output_dir / "timeseries.csv"),
    }
    write_json(artifact_paths["config"], config_payload)
    summary_payload = metrics.summary_payload() if hasattr(
        metrics, "summary_payload") else metrics.summary()
    write_json(artifact_paths["summary"], summary_payload)
    write_timeseries_csv(
        artifact_paths["timeseries"], metrics.timeseries_rows())
    return artifact_paths
