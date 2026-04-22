"""Action helpers used by both controllers and tests."""

from __future__ import annotations

from typing import Any, Iterable


TELEOP_RESIDUAL_ACTION_SCALE = 0.25


def _is_torch_tensor(values: Any) -> bool:
    """Return whether torch tensor."""
    return hasattr(values, "shape") and hasattr(values, "dtype") and hasattr(values, "device")


def clamp_action(action: Any, limit: float) -> Any:
    """Clamp an action vector to ``[-limit, limit]``.

    The helper accepts either torch tensors or plain Python sequences so the
    controller logic stays testable without the heavy runtime stack installed.
    """

    if _is_torch_tensor(action):
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
            raise RuntimeError("Torch tensor action received, but torch is not installed.") from exc
        return torch.clamp(action, min=-float(limit), max=float(limit))

    return [max(-float(limit), min(float(limit), float(value))) for value in action]


def zero_action_like(values: Any, joint_dim: int) -> Any:
    """Create a zero action that matches the observation backend."""

    if _is_torch_tensor(values):
        return values.new_zeros(joint_dim)
    return [0.0] * joint_dim
