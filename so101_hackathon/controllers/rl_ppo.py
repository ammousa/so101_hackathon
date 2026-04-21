"""PPO controller adapter.

This controller keeps the student API tiny while still reusing the real
RSL-RL inference path under the hood when Isaac Lab and RSL-RL are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.controllers.ppo_loader import load_env_free_ppo_policy
from so101_hackathon.rl_training.ppo_config import build_teleop_ppo_runner_cfg
from so101_hackathon.utils.rl_utils import resolve_checkpoint_path


@dataclass
class PPOController(BaseController):
    """Load a trained PPO checkpoint and expose ``act(obs)`` like any controller."""

    checkpoint_path: str | None = None
    log_root: str = "logs/rsl_rl/so101_hackathon_teleop"
    load_run: str = ".*"
    load_checkpoint: str = ".*\\.pt"
    device: str = "cpu"
    seed: int = 42
    logger: str = "tensorboard"
    experiment_name: str = "so101_hackathon_teleop"
    run_name: str = ""
    note: str = ""
    group: str = ""
    action_mode: str = "residual"
    env: Any | None = None
    resolved_checkpoint_path: str | None = field(default=None, init=False)
    _runner_cfg: Any | None = field(default=None, init=False, repr=False)
    _policy: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_checkpoint = self.checkpoint_path or resolve_checkpoint_path(
            log_root_path=self.log_root,
            load_run=self.load_run,
            load_checkpoint=self.load_checkpoint,
        )
        self.resolved_checkpoint_path = resolved_checkpoint
        self._runner_cfg = build_teleop_ppo_runner_cfg(
            seed=self.seed,
            device=self.device,
            logger=self.logger,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            load_run=self.load_run,
            load_checkpoint=self.load_checkpoint,
            note=self.note,
            group=self.group,
        )
        self._policy = self._load_policy(resolved_checkpoint)

    def _load_policy(self, checkpoint_path: str) -> Any:
        if self.env is None:
            return load_env_free_ppo_policy(
                checkpoint_path=checkpoint_path,
                device=self.device,
                actor_hidden_dims=self._runner_cfg.actor_hidden_dims,
                empirical_normalization=bool(
                    self._runner_cfg.empirical_normalization),
            )

        try:
            from so101_hackathon.rl_training.on_policy_runner import OnPolicyRunner
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "PPOController requires `rsl_rl` to be installed. "
                "Install the full runtime dependencies before using the PPO baseline."
            ) from exc

        runner = OnPolicyRunner(
            self.env, self._runner_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint_path)
        return runner.get_inference_policy(device=self.env.unwrapped.device)

    def act(self, obs: Any) -> Any:
        if self._policy is None:
            raise RuntimeError("PPO policy was not initialized correctly.")

        try:
            import torch
        except ModuleNotFoundError:
            return self._policy(obs)

        with torch.inference_mode():
            return self._policy(obs)
