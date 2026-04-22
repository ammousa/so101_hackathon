"""Observation parsing helpers for the single teleop task.

The environment uses one fixed observation layout:

- leader joint position history
- leader joint velocity history
- joint error history
- joint error velocity history
- previous action history

Each term stores ``history_length`` frames, and each frame stores ``joint_dim``
values in the fixed SO101 joint order.
"""

from __future__ import annotations

from typing import Any

from so101_hackathon.sim.robots.so101_follower_spec import SO101_JOINT_NAMES


TELEOP_JOINT_NAMES: tuple[str, ...] = SO101_JOINT_NAMES
TELEOP_HISTORY_LENGTH = 1
TELEOP_VELOCITY_LIMIT = 100.0
TELEOP_TERM_ORDER: tuple[str, ...] = (
    "leader_joint_pos",
    "leader_joint_vel",
    "joint_error",
    "joint_error_vel",
    "previous_action",
)


def _slice_values(values: Any, start: int, stop: int) -> Any:
    """Handle slice values."""
    if isinstance(values, list):
        if values and isinstance(values[0], list):
            return [row[start:stop] for row in values]
        return values[start:stop]
    return values[..., start:stop]


def _vector_length(values: Any) -> int:
    """Handle vector length."""
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(values, list) and values and isinstance(values[0], list):
        return len(values[0])
    return len(values)


def _unwrap_policy_observation(obs: Any) -> Any:
    """Extract the flat policy observation from common Isaac Lab containers.

    Raw Isaac environments often return a mapping like ``{"policy": tensor}``,
    while wrapped RL environments usually return the tensor directly. Keeping
    this unwrapping here lets every controller call the same parser.
    """

    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if len(obs) == 1:
            return next(iter(obs.values()))
    return obs


def finite_difference_velocity(
    current: Any,
    previous: Any,
    dt: float,
    *,
    limit: float = TELEOP_VELOCITY_LIMIT,
) -> Any:
    """Compute a clipped finite-difference velocity."""
    safe_dt = max(float(dt), 1.0e-6)
    bound = float(limit)
    if hasattr(current, "shape") and hasattr(current, "dtype") and hasattr(current, "device"):
        return ((current - previous) / safe_dt).clamp(min=-bound, max=bound)
    return [
        max(-bound, min(bound, (float(current_value) - float(previous_value)) / safe_dt))
        for current_value, previous_value in zip(current, previous)
    ]


def parse_teleop_observation(
    obs: Any,
    *,
    joint_dim: int = len(TELEOP_JOINT_NAMES),
    history_length: int = TELEOP_HISTORY_LENGTH,
) -> dict[str, Any]:
    """Parse the flat teleop observation into named slices.

    The returned dictionary exposes both full history segments and the latest
    frame for each term. This keeps student controllers readable while still
    matching the real environment observation tensor exactly.
    """

    obs = _unwrap_policy_observation(obs)
    expected_dim = len(TELEOP_TERM_ORDER) * joint_dim * history_length
    actual_dim = _vector_length(obs)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Expected teleop observation dim {expected_dim}, received {actual_dim}. "
            "This usually means the controller was connected to the wrong environment."
        )

    parsed: dict[str, Any] = {
        "joint_names": TELEOP_JOINT_NAMES,
        "joint_dim": joint_dim,
        "history_length": history_length,
    }

    cursor = 0
    term_size = joint_dim * history_length
    for term_name in TELEOP_TERM_ORDER:
        history_flat = _slice_values(obs, cursor, cursor + term_size)
        parsed[f"{term_name}_history"] = history_flat
        parsed[term_name] = _slice_values(
            history_flat, term_size - joint_dim, term_size)
        cursor += term_size

    return parsed
