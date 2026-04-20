"""Real-hardware adapter helpers."""

from __future__ import annotations

from pathlib import Path

from so101_hackathon.deploy.runtime import (
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
)

DEFAULT_CALIBRATION_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def load_leader_follower_hardware_dependencies() -> tuple[object, object, object, object, object]:
    from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig
    from lerobot.utils.robot_utils import precise_sleep

    return SOLeader, SOLeaderConfig, SOFollower, SOFollowerConfig, precise_sleep


def create_leader_follower_pair(
    *,
    follower_port: str = DEFAULT_FOLLOWER_PORT,
    follower_id: str = DEFAULT_FOLLOWER_ID,
    leader_port: str = DEFAULT_LEADER_PORT,
    leader_id: str = DEFAULT_LEADER_ID,
    disable_follower_gripper: bool = False,
    SOLeader,
    SOLeaderConfig,
    SOFollower,
    SOFollowerConfig,
):
    leader_cfg = SOLeaderConfig(port=leader_port)
    leader_cfg.id = leader_id
    leader_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR

    follower_cfg = SOFollowerConfig(port=follower_port)
    follower_cfg.id = follower_id
    follower_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR

    leader = SOLeader(leader_cfg)
    follower = SOFollower(follower_cfg)
    if disable_follower_gripper:
        follower.bus.motors.pop("gripper", None)
        follower.bus.calibration.pop("gripper", None)
        if hasattr(follower, "calibration") and isinstance(follower.calibration, dict):
            follower.calibration.pop("gripper", None)

    return leader, follower
