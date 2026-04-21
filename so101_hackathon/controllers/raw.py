"""Reference-tracking baseline that forwards the commanded joints unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.utils.rl_utils import parse_teleop_observation


@dataclass
class RawController(BaseController):
    """Return the latest commanded joint positions directly as the action."""

    def act(self, obs: Any) -> Any:
        parsed = parse_teleop_observation(obs)
        joint_command = parsed["leader_joint_pos"]
        if hasattr(joint_command, "clone"):
            return joint_command.clone()
        return [float(value) for value in joint_command]

    def reset(self) -> None:
        """The baseline keeps no episode state."""

        return None
