"""Shared teleop evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable


def _as_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


@dataclass
class TeleopMetricAccumulator:
    """Accumulate controller-agnostic teleop metrics episode by episode."""

    compute_pose_fn: Callable[[Iterable[float]],
                              tuple[list[float], list[float]]] | None = None
    failure_threshold: float = 0.75
    joint_count: int = 5
    _episodes: list[dict[str, float]] = field(
        default_factory=list, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _sum_joint_mse: float = field(default=0.0, init=False, repr=False)
    _max_joint_error: float = field(default=0.0, init=False, repr=False)
    _sum_action_rate: float = field(default=0.0, init=False, repr=False)
    _sum_ee_position_sq: float = field(default=0.0, init=False, repr=False)
    _sum_ee_orientation_sq: float = field(default=0.0, init=False, repr=False)
    _max_ee_position_error: float = field(default=0.0, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _failed: bool = field(default=False, init=False, repr=False)

    def reset_episode(self) -> None:
        self._step_count = 0
        self._sum_joint_mse = 0.0
        self._max_joint_error = 0.0
        self._sum_action_rate = 0.0
        self._sum_ee_position_sq = 0.0
        self._sum_ee_orientation_sq = 0.0
        self._max_ee_position_error = 0.0
        self._failure_count = 0
        self._failed = False

    def add_step(
        self,
        *,
        joint_error: Iterable[float],
        action_rate: float,
        ee_position_error: float = 0.0,
        ee_orientation_error: float = 0.0,
        invalid_state: bool = False,
        failure: bool = False,
    ) -> None:
        errors = _as_list(joint_error)
        if len(errors) != self.joint_count:
            raise ValueError(
                f"Expected {self.joint_count} joint errors, received {len(errors)}")

        self._step_count += 1
        mean_sq_error = sum(
            error * error for error in errors) / self.joint_count
        max_joint_error = max(abs(error) for error in errors)
        self._sum_joint_mse += mean_sq_error
        self._max_joint_error = max(self._max_joint_error, max_joint_error)
        self._sum_action_rate += float(action_rate)
        self._sum_ee_position_sq += float(ee_position_error) ** 2
        self._sum_ee_orientation_sq += float(ee_orientation_error) ** 2
        self._max_ee_position_error = max(
            self._max_ee_position_error, float(ee_position_error))
        failure_event = bool(failure or invalid_state or (
            max_joint_error > self.failure_threshold))
        self._failure_count += int(failure_event)
        self._failed = self._failed or failure_event

    def finish_episode(self) -> dict[str, float]:
        count = max(self._step_count, 1)
        episode = {
            "joint_rmse": math.sqrt(self._sum_joint_mse / count),
            "max_joint_error": self._max_joint_error,
            "ee_position_rmse": math.sqrt(self._sum_ee_position_sq / count),
            "ee_orientation_rmse_rad": math.sqrt(self._sum_ee_orientation_sq / count),
            "max_ee_position_error": self._max_ee_position_error,
            "command_smoothness": self._sum_action_rate / count,
            "num_failures": float(self._failure_count),
            "failure_rate": float(self._failed),
        }
        self._episodes.append(episode)
        self.reset_episode()
        return episode

    def summary(self) -> dict[str, float]:
        if not self._episodes:
            return {
                "eval/joint_rmse": 0.0,
                "eval/max_joint_error": 0.0,
                "eval/ee_position_rmse": 0.0,
                "eval/ee_orientation_rmse_rad": 0.0,
                "eval/max_ee_position_error": 0.0,
                "eval/command_smoothness": 0.0,
                "eval/num_failures": 0.0,
                "eval/num_episodes": 0.0,
            }

        count = float(len(self._episodes))
        return {
            "eval/joint_rmse": sum(ep["joint_rmse"] for ep in self._episodes) / count,
            "eval/max_joint_error": sum(ep["max_joint_error"] for ep in self._episodes) / count,
            "eval/ee_position_rmse": sum(ep["ee_position_rmse"] for ep in self._episodes) / count,
            "eval/ee_orientation_rmse_rad": sum(ep["ee_orientation_rmse_rad"] for ep in self._episodes) / count,
            "eval/max_ee_position_error": sum(ep["max_ee_position_error"] for ep in self._episodes) / count,
            "eval/command_smoothness": sum(ep["command_smoothness"] for ep in self._episodes) / count,
            "eval/num_failures": sum(ep["num_failures"] for ep in self._episodes),
            "eval/num_episodes": count,
        }
