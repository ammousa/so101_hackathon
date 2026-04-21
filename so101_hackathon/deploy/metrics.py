"""Deploy-time metrics and artifact helpers."""

from __future__ import annotations

import math
from typing import Iterable

from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES


def _as_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _euclidean_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


class DeployMetricAccumulator:
    """Accumulate deploy-time tracking metrics and time-series rows."""

    def __init__(self, joint_names: Iterable[str] = TELEOP_JOINT_NAMES, failure_threshold: float = 0.75):
        self.joint_names = list(joint_names)
        self.failure_threshold = float(failure_threshold)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum_joint_mse = 0.0
        self.sum_ee_error_sq = 0.0
        self.sum_command_smoothness = 0.0
        self.max_joint_error = 0.0
        self.failure_count = 0.0
        self.prev_commanded_joint_pos: list[float] | None = None
        self.sum_abs = {joint_name: 0.0 for joint_name in self.joint_names}
        self.sum_sq = {joint_name: 0.0 for joint_name in self.joint_names}
        self.max_abs = {joint_name: 0.0 for joint_name in self.joint_names}
        self.last_abs_err = {
            joint_name: 0.0 for joint_name in self.joint_names}
        self._timeseries_rows: list[dict[str, float]] = []

    def _compute_ee_position_error(
        self,
        leader_joint_pos: list[float],
        follower_joint_pos: list[float],
    ) -> float:
        try:
            import torch
        except ModuleNotFoundError:
            return 0.0

        from so101_hackathon.sim.kinematics import compute_so101_ee_position

        leader_tensor = torch.tensor([leader_joint_pos], dtype=torch.float32)
        follower_tensor = torch.tensor(
            [follower_joint_pos], dtype=torch.float32)
        leader_ee = compute_so101_ee_position(
            leader_tensor, joint_names=self.joint_names)
        follower_ee = compute_so101_ee_position(
            follower_tensor, joint_names=self.joint_names)
        return float(torch.linalg.norm(leader_ee[0] - follower_ee[0]).item())

    def update(
        self,
        *,
        step: int,
        timestamp_s: float,
        leader_joint_pos: Iterable[float],
        follower_joint_pos: Iterable[float],
        commanded_joint_pos: Iterable[float],
    ) -> dict[str, float]:
        leader = _as_list(leader_joint_pos)
        follower = _as_list(follower_joint_pos)
        commanded = _as_list(commanded_joint_pos)
        joint_error = [float(leader_value) - float(follower_value)
                       for leader_value, follower_value in zip(leader, follower)]
        abs_error = [abs(value) for value in joint_error]
        joint_rmse = math.sqrt(
            sum(value * value for value in joint_error) / len(self.joint_names))
        max_joint_error = max(abs_error)
        ee_position_error = self._compute_ee_position_error(leader, follower)
        if self.prev_commanded_joint_pos is None:
            command_smoothness = 0.0
        else:
            command_smoothness = _euclidean_norm(
                float(current) - float(previous)
                for current, previous in zip(commanded, self.prev_commanded_joint_pos)
            )
        failure = float(max_joint_error > self.failure_threshold)

        self.count += 1
        self.sum_joint_mse += joint_rmse * joint_rmse
        self.sum_ee_error_sq += ee_position_error * ee_position_error
        self.sum_command_smoothness += command_smoothness
        self.max_joint_error = max(self.max_joint_error, max_joint_error)
        self.failure_count += failure
        self.prev_commanded_joint_pos = commanded

        row = {
            "step": float(step),
            "timestamp_s": float(timestamp_s),
            "joint_rmse_step": float(joint_rmse),
            "ee_position_error_step": float(ee_position_error),
            "command_smoothness_step": float(command_smoothness),
            "max_joint_error_step": float(max_joint_error),
            "failed_step": float(failure),
        }

        for index, joint_name in enumerate(self.joint_names):
            joint_abs_error = abs_error[index]
            self.sum_abs[joint_name] += joint_abs_error
            self.sum_sq[joint_name] += joint_error[index] * joint_error[index]
            self.max_abs[joint_name] = max(
                self.max_abs[joint_name], joint_abs_error)
            self.last_abs_err[joint_name] = joint_abs_error
            row[f"leader_{joint_name}"] = float(leader[index])
            row[f"follower_{joint_name}"] = float(follower[index])
            row[f"commanded_{joint_name}"] = float(commanded[index])

        self._timeseries_rows.append(row)
        return {
            "joint_rmse": float(joint_rmse),
            "ee_position_error": float(ee_position_error),
            "command_smoothness": float(command_smoothness),
            "max_joint_error": float(max_joint_error),
            "failed": float(failure),
        }

    def summary(self) -> dict[str, float]:
        count = max(self.count, 1)
        return {
            "joint_rmse": math.sqrt(self.sum_joint_mse / count),
            "max_joint_error": self.max_joint_error,
            "ee_position_rmse": math.sqrt(self.sum_ee_error_sq / count),
            "command_smoothness": self.sum_command_smoothness / count,
            "num_failures": self.failure_count,
            "num_steps": float(self.count),
        }

    def per_joint_summary(self) -> dict[str, dict[str, float]]:
        count = max(self.count, 1)
        return {
            joint_name: {
                "mae": self.sum_abs[joint_name] / count,
                "rmse": math.sqrt(self.sum_sq[joint_name] / count),
                "max_abs": self.max_abs[joint_name],
            }
            for joint_name in self.joint_names
        }

    def timeseries_rows(self) -> list[dict[str, float]]:
        return list(self._timeseries_rows)

    def summary_payload(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "per_joint": self.per_joint_summary(),
        }

    def format_status_line(self, *, iter_idx: int, hz: float) -> str:
        summary = self.summary()
        return (
            f"[{iter_idx:06d}] "
            f"RMSE={summary['joint_rmse']:.3f} | "
            f"EE={summary['ee_position_rmse']:.3f} | "
            f"MAX={summary['max_joint_error']:.3f} | "
            f"Hz={hz:.1f}"
        )

    def format_last_joint_errors(self) -> str:
        return " | ".join(f"{joint_name}={self.last_abs_err[joint_name]:.3f}" for joint_name in self.joint_names)

    def format_final_report(self) -> str:
        summary = self.summary()
        lines = ["", "Final deploy metrics:"]
        for joint_name, payload in self.per_joint_summary().items():
            lines.append(
                f"{joint_name:14s} | "
                f"MAE={payload['mae']:.3f} | "
                f"RMSE={payload['rmse']:.3f} | "
                f"MAX={payload['max_abs']:.3f}"
            )
        lines.append(
            f"\nOverall        | "
            f"RMSE={summary['joint_rmse']:.3f} | "
            f"EE={summary['ee_position_rmse']:.3f} | "
            f"MAX={summary['max_joint_error']:.3f} | "
            f"Failures={summary['num_failures']:.0f}"
        )
        return "\n".join(lines)
