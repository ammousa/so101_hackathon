"""Curriculum helpers for SO101 teleoperation tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from .adaptive_curriculum_utils import sample_episode_disturbance_per_env, update_difficulty_levels


DEFAULT_DISTURBANCE_SCHEDULE = (
    {"max_step": 10000, "stage": 1, "delay_range": (
        0, 0), "noise_range": (0.0, 0.0)},
    {"max_step": 20000, "stage": 2, "delay_range": (
        0, 0), "noise_range": (0.0, 0.005)},
    {"max_step": 400000, "stage": 3, "delay_range": (
        2, 2), "noise_range": (0.0, 0.005)},
    {"max_step": 60000, "stage": 4, "delay_range": (
        0, 3), "noise_range": (0.0, 0.01)},
    {"max_step": None, "stage": 5, "delay_range": (
        0, 8), "noise_range": (0.0, 0.03)},
)


@configclass
class TeleopDifficultyLevelCfg:
    """Trajectory and disturbance settings for a single curriculum level."""

    active_joint_count_range: tuple[int, int] = (5, 5)
    position_span_scale: float = 1.0
    segment_duration_range_s: tuple[float, float] = (4.0, 4.0)
    delay_max: int = 0
    noise_std_max: float = 0.0


@configclass
class AdaptiveTeleopCurriculumParamsCfg:
    """Configuration for the adaptive teleoperation curriculum scheduler."""

    command_name: str = "leader_joints"
    action_name: str = "arm_action"
    good_threshold_rad: float = 0.02
    bad_threshold_rad: float = 0.06
    init_level: int = 0
    max_level: int = 9
    levels: tuple[TeleopDifficultyLevelCfg, ...] = ()


def _env_ids_to_tensor(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    """Convert Isaac Lab env-id selections into a dense index tensor."""

    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class AdaptiveTeleopDifficultyCurriculum(ManagerTermBase):
    """Adaptive per-environment curriculum for teleoperation tracking."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params_cfg = self.cfg.params["params_cfg"]
        max_level = min(int(params_cfg.max_level), max(len(params_cfg.levels) - 1, 0))
        init_level = int(min(params_cfg.init_level, max_level))
        self.current_level = torch.full((env.num_envs,), init_level, device=env.device, dtype=torch.int)
        self.last_episode_joint_rmse = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self.promoted_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.demoted_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._max_level = max_level
        self._level_active_joint_count_low = torch.tensor(
            [level.active_joint_count_range[0] for level in params_cfg.levels],
            device=env.device,
            dtype=torch.int,
        )
        self._level_active_joint_count_high = torch.tensor(
            [level.active_joint_count_range[1] for level in params_cfg.levels],
            device=env.device,
            dtype=torch.int,
        )
        self._level_position_span_scale = torch.tensor(
            [level.position_span_scale for level in params_cfg.levels],
            device=env.device,
            dtype=torch.float32,
        )
        self._level_segment_duration_min = torch.tensor(
            [level.segment_duration_range_s[0] for level in params_cfg.levels],
            device=env.device,
            dtype=torch.float32,
        )
        self._level_segment_duration_max = torch.tensor(
            [level.segment_duration_range_s[1] for level in params_cfg.levels],
            device=env.device,
            dtype=torch.float32,
        )
        self._level_delay_max = torch.tensor(
            [level.delay_max for level in params_cfg.levels],
            device=env.device,
            dtype=torch.int,
        )
        self._level_noise_std_max = torch.tensor(
            [level.noise_std_max for level in params_cfg.levels],
            device=env.device,
            dtype=torch.float32,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Clear one-step promotion/demotion flags after the manager logs them."""

        env_ids_tensor = _env_ids_to_tensor(slice(None) if env_ids is None else env_ids, self._env.num_envs, self._env.device)
        self.promoted_mask[env_ids_tensor] = False
        self.demoted_mask[env_ids_tensor] = False

    def __call__(self, env, env_ids, params_cfg: AdaptiveTeleopCurriculumParamsCfg) -> dict[str, float]:
        """Update per-env levels and sample the next episode configuration.

        Isaac Lab calls curriculum terms before manager resets, which lets this
        scheduler inspect the just-finished episode and configure the next one.
        """

        env_ids_tensor = _env_ids_to_tensor(env_ids, env.num_envs, env.device)
        if env_ids_tensor.numel() == 0:
            return {
                "mean_level": 0.0,
                "min_level": 0.0,
                "max_level": 0.0,
                "mean_episode_joint_rmse": 0.0,
                "promotion_rate": 0.0,
                "demotion_rate": 0.0,
            }

        command_term = env.command_manager.get_term(params_cfg.command_name)
        action_term = env.action_manager.get_term(params_cfg.action_name)
        tracking_stats = command_term.get_episode_tracking_statistics(env_ids_tensor)
        next_level, promoted, demoted = update_difficulty_levels(
            levels=self.current_level[env_ids_tensor],
            episode_joint_rmse=tracking_stats["joint_rmse"],
            sample_count=tracking_stats["sample_count"],
            good_threshold=float(params_cfg.good_threshold_rad),
            bad_threshold=float(params_cfg.bad_threshold_rad),
            max_level=self._max_level,
        )

        self.current_level[env_ids_tensor] = next_level
        self.last_episode_joint_rmse[env_ids_tensor] = tracking_stats["joint_rmse"]
        self.promoted_mask[env_ids_tensor] = promoted
        self.demoted_mask[env_ids_tensor] = demoted
        level_index = next_level.to(dtype=torch.long)
        command_term.set_difficulty_profile(
            env_ids_tensor,
            active_joint_count_low=self._level_active_joint_count_low[level_index],
            active_joint_count_high=self._level_active_joint_count_high[level_index],
            position_span_scale=self._level_position_span_scale[level_index],
            segment_duration_min_s=self._level_segment_duration_min[level_index],
            segment_duration_max_s=self._level_segment_duration_max[level_index],
        )
        delay_steps, noise_std = sample_episode_disturbance_per_env(
            delay_max=self._level_delay_max[level_index],
            noise_std_max=self._level_noise_std_max[level_index],
        )
        action_term.set_episode_disturbance(env_ids_tensor, delay_steps=delay_steps, noise_std=noise_std)

        return {
            "mean_level": float(self.current_level[env_ids_tensor].float().mean().item()),
            "min_level": float(self.current_level[env_ids_tensor].min().item()),
            "max_level": float(self.current_level[env_ids_tensor].max().item()),
            "mean_episode_joint_rmse": float(self.last_episode_joint_rmse[env_ids_tensor].mean().item()),
            "promotion_rate": float(self.promoted_mask[env_ids_tensor].float().mean().item()),
            "demotion_rate": float(self.demoted_mask[env_ids_tensor].float().mean().item()),
        }


def disturbance_curriculum(
    env,
    env_ids,
    action_name: str = "arm_action",
    schedule: Sequence[dict] = DEFAULT_DISTURBANCE_SCHEDULE,
):
    """Update delay/noise ranges according to a configurable schedule."""
    action_term = env.action_manager.get_term(action_name)
    step = int(env.common_step_counter)

    selected = schedule[-1]
    for entry in schedule:
        max_step = entry.get("max_step")
        if max_step is None or step < int(max_step):
            selected = entry
            break

    stage = int(selected["stage"])
    delay_range = tuple(selected["delay_range"])
    noise_values = selected.get("noise_range")
    if noise_values is None:
        noise_values = selected["noise_std_range"]
    noise_range = tuple(noise_values)

    action_term.set_disturbance_ranges(
        delay_range=delay_range, noise_std_range=noise_range)
    return {
        "stage": stage,
        "delay_min": delay_range[0],
        "delay_max": delay_range[1],
        "noise_min": noise_range[0],
        "noise_max": noise_range[1],
    }


fixed_disturbance_curriculum = disturbance_curriculum
