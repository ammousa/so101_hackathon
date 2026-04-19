"""PPO controller adapter.

This controller keeps the student API tiny while still reusing the real
RSL-RL inference path under the hood when Isaac Lab and RSL-RL are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from so101_hackathon.controllers.base import BaseController
from so101_hackathon.training.ppo_config import build_teleop_ppo_runner_cfg
from so101_hackathon.utils.checkpoints import resolve_checkpoint_path


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
    env: Any | None = None
    _runner_cfg: Any | None = field(default=None, init=False, repr=False)
    _policy: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.env is None:
            raise ValueError("PPOController requires a wrapped environment instance.")

        resolved_checkpoint = self.checkpoint_path or resolve_checkpoint_path(
            log_root_path=self.log_root,
            load_run=self.load_run,
            load_checkpoint=self.load_checkpoint,
        )
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
        try:
            from so101_hackathon.training.on_policy_runner import OnPolicyRunner
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "PPOController requires `rsl_rl` to be installed. "
                "Install the full runtime dependencies before using the PPO baseline."
            ) from exc

        runner = OnPolicyRunner(self.env, self._runner_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint_path)
        return runner.get_inference_policy(device=self.env.unwrapped.device)

    def act(self, obs: Any) -> Any:
        if self._policy is None:
            raise RuntimeError("PPO policy was not initialized correctly.")

        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError("PPOController requires `torch` to run inference.") from exc

        with torch.inference_mode():
            return self._policy(obs)
