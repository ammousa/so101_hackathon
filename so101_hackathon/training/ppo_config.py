"""Small PPO config surface for the single teleop baseline.

This module intentionally avoids importing Isaac Lab helper packages so the
training and inference scripts can build their runner config in any environment
where `rsl_rl` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeleopPpoRunnerCfg:
    """Configuration passed to the RSL-RL runner."""

    experiment_name: str = "so101_hackathon_teleop"
    run_name: str = ""
    device: str = "cpu"
    seed: int = 42
    max_iterations: int = 1500
    save_interval: int = 50
    num_steps_per_env: int = 24
    logger: str = "tensorboard"
    note: str = ""
    group: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = ".*\\.pt"
    empirical_normalization: bool = False
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    value_loss_coef: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

    def to_dict(self) -> dict:
        """Return the dictionary expected by ``OnPolicyRunner``."""

        return {
            "seed": self.seed,
            "device": self.device,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "save_interval": self.save_interval,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": self.empirical_normalization,
            "obs_groups": {
                "policy": ["policy"],
                "critic": ["policy"],
            },
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "noise_std_type": "scalar",
                "actor_obs_normalization": self.empirical_normalization,
                "critic_obs_normalization": self.empirical_normalization,
                "actor_hidden_dims": list(self.actor_hidden_dims),
                "critic_hidden_dims": list(self.critic_hidden_dims),
                "activation": "elu",
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": self.value_loss_coef,
                "use_clipped_value_loss": True,
                "clip_param": self.clip_param,
                "entropy_coef": self.entropy_coef,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "learning_rate": self.learning_rate,
                "schedule": "adaptive",
                "gamma": self.gamma,
                "lam": self.lam,
                "desired_kl": self.desired_kl,
                "max_grad_norm": self.max_grad_norm,
            },
            "note": self.note,
            "group": self.group,
        }


def build_teleop_ppo_runner_cfg(
    *,
    seed: int,
    device: str,
    logger: str,
    experiment_name: str,
    run_name: str,
    resume: bool = False,
    load_run: str = ".*",
    load_checkpoint: str = ".*\\.pt",
    note: str = "",
    group: str = "",
) -> TeleopPpoRunnerCfg:
    """Build the single PPO config used by the hackathon repo."""

    return TeleopPpoRunnerCfg(
        seed=seed,
        device=device,
        logger=logger,
        experiment_name=experiment_name,
        run_name=run_name,
        resume=resume,
        load_run=load_run,
        load_checkpoint=load_checkpoint,
        note=note,
        group=group,
    )
