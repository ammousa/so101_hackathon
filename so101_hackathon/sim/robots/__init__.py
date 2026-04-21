"""SO101 robot assets."""

from __future__ import annotations

from .so101_follower_spec import SO101_JOINT_NAMES

__all__ = ["SO101_FOLLOWER_CFG", "SO101_JOINT_NAMES"]


def __getattr__(name: str):
    if name == "SO101_FOLLOWER_CFG":  # pragma: no cover - depends on Isaac Lab runtime
        from .so101_follower_cfg import SO101_FOLLOWER_CFG

        return SO101_FOLLOWER_CFG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
