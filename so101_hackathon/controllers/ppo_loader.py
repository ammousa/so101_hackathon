"""Env-free PPO inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from so101_hackathon.utils.rl_utils import (
    TELEOP_HISTORY_LENGTH,
    TELEOP_JOINT_NAMES,
    TELEOP_TERM_ORDER,
)

PPO_OBS_DIM = len(TELEOP_TERM_ORDER) * \
    len(TELEOP_JOINT_NAMES) * TELEOP_HISTORY_LENGTH
PPO_ACTION_DIM = len(TELEOP_JOINT_NAMES)
_GAUSSIAN_DISTRIBUTION_CLASS = "rsl_rl.modules.distribution:GaussianDistribution"


def _unwrap_policy_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if len(obs) == 1:
            return next(iter(obs.values()))
    return obs


def _import_rsl_rl_inference_deps():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Env-free PPO inference requires `torch` to be installed.") from exc

    try:
        from tensordict import TensorDict
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Env-free PPO inference requires `tensordict` from the RSL-RL stack.") from exc

    try:
        from rsl_rl.models import MLPModel
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Env-free PPO inference requires `rsl_rl` to be installed.") from exc

    return torch, TensorDict, MLPModel


def _load_checkpoint_payload(torch_module: Any, checkpoint_path: str, device: str) -> dict[str, Any]:
    load_kwargs = {"map_location": device}
    try:
        payload = torch_module.load(
            checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:  # pragma: no cover - older torch versions
        payload = torch_module.load(checkpoint_path, **load_kwargs)
    if not isinstance(payload, dict):
        raise TypeError(
            f"PPO checkpoint must deserialize into a dict, received {type(payload)}")
    return payload


def _extract_actor_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    if "actor_state_dict" in payload:
        state_dict = payload["actor_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Actor state dict must be a dict, received {type(state_dict)}")
    return state_dict


@dataclass
class EnvFreePpoPolicy:
    """Small wrapper that accepts flat observations and runs MLPModel inference."""

    actor: Any
    torch_module: Any
    tensor_dict_cls: Any
    device: str

    def __call__(self, obs: Any) -> Any:
        obs = _unwrap_policy_observation(obs)
        if isinstance(obs, list):
            obs_tensor = self.torch_module.tensor(
                obs, dtype=self.torch_module.float32, device=self.device)
        elif hasattr(obs, "to"):
            obs_tensor = obs.to(device=self.device,
                                dtype=self.torch_module.float32)
        else:
            obs_tensor = self.torch_module.tensor(
                list(obs), dtype=self.torch_module.float32, device=self.device)

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.shape[-1] != PPO_OBS_DIM:
            raise ValueError(
                f"Expected PPO deploy observation dim {PPO_OBS_DIM}, received {tuple(obs_tensor.shape)}"
            )

        observation = self.tensor_dict_cls(
            {"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]])
        with self.torch_module.inference_mode():
            action = self.actor(observation)
        if action.ndim > 1 and action.shape[0] == 1:
            return action.squeeze(0)
        return action


def load_env_free_ppo_policy(
    *,
    checkpoint_path: str,
    device: str,
    actor_hidden_dims: Iterable[int],
    empirical_normalization: bool,
) -> EnvFreePpoPolicy:
    """Load PPO inference without constructing an Isaac environment."""

    torch_module, tensor_dict_cls, mlp_model_cls = _import_rsl_rl_inference_deps()
    dummy_obs = tensor_dict_cls(
        {"policy": torch_module.zeros(
            (1, PPO_OBS_DIM), dtype=torch_module.float32, device=device)},
        batch_size=[1],
    )
    actor = mlp_model_cls(
        dummy_obs,
        {"actor": ["policy"], "critic": ["policy"]},
        "actor",
        PPO_ACTION_DIM,
        hidden_dims=list(actor_hidden_dims),
        activation="elu",
        obs_normalization=bool(empirical_normalization),
        distribution_cfg={
            "class_name": _GAUSSIAN_DISTRIBUTION_CLASS,
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ).to(device)

    payload = _load_checkpoint_payload(torch_module, checkpoint_path, device)
    actor_state_dict = _extract_actor_state_dict(payload)
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()
    return EnvFreePpoPolicy(
        actor=actor,
        torch_module=torch_module,
        tensor_dict_cls=tensor_dict_cls,
        device=device,
    )
