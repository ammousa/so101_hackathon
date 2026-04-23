"""CSV trajectory source for deploy scenarios."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from so101_hackathon.deploy.runtime import build_follower_action
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES

_TIME_COLUMNS = {"time", "time_s", "t", "timestamp", "timestamp_s", "step", "timestep"}


@dataclass
class CSVJointTrajectory:
    """Read absolute SO101 joint targets from a CSV file."""

    csv_path: str | None = None
    path: str | None = None
    joint_columns: Sequence[str] | None = None
    frequency_hz: int = 60
    cycles: int = 1
    return_to_start_steps: int = 60

    _targets: list[list[float]] = field(init=False, default_factory=list)
    _step_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Load and validate the configured CSV trajectory."""
        if int(self.frequency_hz) != 60:
            raise ValueError("CSVJointTrajectory expects frequency_hz=60")
        resolved_path = self.csv_path or self.path
        if not resolved_path:
            raise ValueError("csv_path must be provided for CSVJointTrajectory")
        if int(self.cycles) <= 0:
            raise ValueError("cycles must be positive")
        if int(self.return_to_start_steps) < 0:
            raise ValueError("return_to_start_steps must be non-negative")
        self._targets = self._load_targets(Path(resolved_path))
        if not self._targets:
            raise ValueError(f"trajectory CSV contains no targets: {resolved_path}")

    @property
    def start_target(self) -> list[float]:
        """Return the first target in the trajectory."""
        return list(self._targets[0])

    @property
    def completed(self) -> bool:
        """Return whether all configured trajectory targets have been emitted."""
        return self._step_index >= self.total_steps

    @property
    def total_steps(self) -> int:
        """Return the number of emitted targets before completion."""
        return len(self._targets) * int(self.cycles)

    def reset(self) -> None:
        """Restart playback from the first CSV row."""
        self._step_index = 0

    def next_joint_target(self) -> list[float]:
        """Return the next joint target, or raise when the trajectory is done."""
        if self._step_index >= self.total_steps:
            raise StopIteration("trajectory CSV exhausted")
        target = self._targets[self._step_index % len(self._targets)]
        self._step_index += 1
        return list(target)

    def _load_targets(self, path: Path) -> list[list[float]]:
        """Load targets from either a headered or plain numeric CSV."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = [
                row for row in csv.reader(handle)
                if row and any(value.strip() for value in row)
            ]
        if not raw_rows:
            return []

        try:
            return [self._numeric_row_to_target(row) for row in raw_rows]
        except ValueError:
            fieldnames = [value.strip() for value in raw_rows[0]]
            columns = self._resolve_columns(fieldnames)
            return [
                [float(dict(zip(fieldnames, row))[column]) for column in columns]
                for row in raw_rows[1:]
            ]

    def _numeric_row_to_target(self, row: Sequence[str]) -> list[float]:
        """Parse one plain numeric row into six joint targets."""
        values = [float(value) for value in row]
        if len(values) == len(TELEOP_JOINT_NAMES) + 1:
            values = values[1:]
        if len(values) != len(TELEOP_JOINT_NAMES):
            raise ValueError(
                "CSV rows must contain 6 joint targets, or a time/step column plus 6 targets"
            )
        return values

    def _resolve_columns(self, fieldnames: Sequence[str]) -> list[str]:
        """Choose the six joint target columns from a headered CSV."""
        if self.joint_columns is not None:
            columns = list(self.joint_columns)
        elif all(joint_name in fieldnames for joint_name in TELEOP_JOINT_NAMES):
            columns = list(TELEOP_JOINT_NAMES)
        else:
            columns = [
                column for column in fieldnames
                if column.strip().lower() not in _TIME_COLUMNS
            ][:len(TELEOP_JOINT_NAMES)]

        if len(columns) != len(TELEOP_JOINT_NAMES):
            raise ValueError(
                f"trajectory CSV must provide {len(TELEOP_JOINT_NAMES)} joint columns"
            )
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise ValueError(f"trajectory CSV is missing columns: {missing}")
        return columns


class HardwareTrajectoryLeader:
    """Leader-like adapter that emits CSV joint targets as hardware action fields."""

    def __init__(self, trajectory: CSVJointTrajectory):
        """Initialize the adapter."""
        self.trajectory = trajectory

    def connect(self) -> None:
        """No hardware leader is used."""
        return None

    def disconnect(self) -> None:
        """No hardware leader is used."""
        return None

    def get_action(self) -> dict[str, float]:
        """Return the next trajectory target in hardware action format."""
        return build_follower_action(self.trajectory.next_joint_target())
