"""Starter template for student controllers.

To add a new controller:

1. Copy this file.
2. Rename the class.
3. Implement ``act(obs)``.
4. Register the class in ``so101_hackathon/registry.py``.
"""

from __future__ import annotations

from typing import Any

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.utils.obs_utils import parse_teleop_observation


class MyController(BaseController):
    """Minimal example controller.

    This baseline simply returns zeros, which is usually a bad controller but a
    useful starting point for wiring and debugging.
    """

    def reset(self) -> None:
        return None

    def act(self, obs: Any) -> Any:
        parsed = parse_teleop_observation(obs)
        template = parsed["previous_action"]
        if hasattr(template, "new_zeros"):
            return template.new_zeros(parsed["joint_dim"])
        return [0.0] * parsed["joint_dim"]
