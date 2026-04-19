"""Simple rule-based baseline for the teleop task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.utils.action_utils import clamp_action
from so101_hackathon.utils.obs_utils import parse_teleop_observation


@dataclass
class TeleopPDController(BaseController):
    """Track the latest leader error with a lightweight PD heuristic.

    The controller stays intentionally small so students can read it end-to-end.
    It relies on the observation parser instead of reaching into Isaac Lab
    internals, which keeps the baseline focused on controller logic.
    """

    kp: float = 1.0
    kd: float = 0.15
    max_action: float = 1.0

    def act(self, obs: Any) -> Any:
        parsed = parse_teleop_observation(obs)
        joint_error = parsed["joint_error"]
        joint_error_vel = parsed["joint_error_vel"]
        if hasattr(joint_error, "shape"):
            action = self.kp * joint_error + self.kd * joint_error_vel
            return clamp_action(action, limit=self.max_action)

        return clamp_action(
            [self.kp * float(err) + self.kd * float(err_vel)
             for err, err_vel in zip(joint_error, joint_error_vel)],
            limit=self.max_action,
        )

    def reset(self) -> None:
        """The PD baseline is stateless across episodes."""

        return None


def default_pd_controller() -> TeleopPDController:
    """Factory used by configs and tests."""

    return TeleopPDController()
