"""Custom command terms for SO101 teleoperation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .adaptive_curriculum_utils import compute_episode_joint_rmse, sample_duration_range, sample_waypoint_targets


def _env_ids_to_tensor(env_ids: Sequence[int] | slice | torch.Tensor, num_envs: int, device: torch.device) -> torch.Tensor:
    """Convert Isaac Lab env-id selections into a dense index tensor."""

    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


@configclass
class TrajectoryDifficultyProfileCfg:
    """Per-episode trajectory difficulty settings used by the curriculum."""

    active_joint_count_range: tuple[int, int] = (5, 5)
    position_span_scale: float = 1.0
    segment_duration_range_s: tuple[float, float] = (4.0, 4.0)


@configclass
class ProceduralJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for procedural SO101 leader joint trajectories."""

    class_type: type[CommandTerm] = MISSING  # type: ignore
    asset_name: str = "robot"
    joint_names: list[str] = MISSING  # type: ignore
    waypoint_limit_margin: float = 0.05
    preserve_order: bool = True
    difficulty_profile: TrajectoryDifficultyProfileCfg = TrajectoryDifficultyProfileCfg()

    def __post_init__(self):
        self.class_type = ProceduralJointTrajectoryCommand


class ProceduralJointTrajectoryCommand(CommandTerm):
    """Generate smooth waypoint-based leader references with per-env difficulty."""

    cfg: ProceduralJointTrajectoryCommandCfg

    def __init__(self, cfg: ProceduralJointTrajectoryCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids, self.joint_names = self.robot.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order)
        self.num_joints = len(self.joint_ids)
        try:
            self.leader: Articulation | None = env.scene["leader_robot"]
        except Exception:
            try:
                self.leader = env.scene["leader_ghost"]
            except Exception:
                self.leader = None
        if self.leader is not None:
            self.leader_joint_ids, _ = self.leader.find_joints(
                self.joint_names, preserve_order=True)
            print(
                "[INFO] Leader visualization robot found and will be driven by the command term.")
            self._bind_leader_visual_override()
        else:
            self.leader_joint_ids = None
            print(
                "[WARN] Leader visualization robot was requested but not found in the scene.")
        self.command_pos = torch.zeros(
            self.num_envs, self.num_joints, device=self.device)
        self.command_vel = torch.zeros_like(self.command_pos)
        self._segment_start = torch.zeros_like(self.command_pos)
        self._segment_end = torch.zeros_like(self.command_pos)
        self._segment_duration = torch.ones(self.num_envs, device=self.device)
        self._segment_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._has_active_segment = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._active_joint_count_low = torch.full(
            (self.num_envs,),
            int(cfg.difficulty_profile.active_joint_count_range[0]),
            device=self.device,
            dtype=torch.int,
        )
        self._active_joint_count_high = torch.full(
            (self.num_envs,),
            int(cfg.difficulty_profile.active_joint_count_range[1]),
            device=self.device,
            dtype=torch.int,
        )
        self._position_span_scale = torch.full(
            (self.num_envs,),
            float(cfg.difficulty_profile.position_span_scale),
            device=self.device,
            dtype=torch.float32,
        )
        self._segment_duration_min = torch.full(
            (self.num_envs,),
            float(cfg.difficulty_profile.segment_duration_range_s[0]),
            device=self.device,
            dtype=torch.float32,
        )
        self._segment_duration_max = torch.full(
            (self.num_envs,),
            float(cfg.difficulty_profile.segment_duration_range_s[1]),
            device=self.device,
            dtype=torch.float32,
        )
        self._episode_sq_joint_error_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_sample_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_max_joint_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _bind_leader_visual_override(self) -> None:
        """Force a strong visual material binding on the spawned leader robot."""
        stage = sim_utils.get_current_stage()
        material_path = "/World/Looks/LeaderRobotMaterial"
        if not stage.GetPrimAtPath(material_path).IsValid():
            leader_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.10, 0.10),
                emissive_color=(0.10, 0.02, 0.02),
                opacity=1.0,
                metallic=0.0,
                roughness=0.15,
            )
            leader_material.func(material_path, leader_material)
        if not hasattr(self.leader, "_prims"):
            self.leader._prims = sim_utils.find_matching_prims(
                self.leader.cfg.prim_path)
        for prim in self.leader._prims:
            sim_utils.bind_visual_material(
                prim.GetPath().pathString, material_path, stage=stage)
        self.leader.set_visibility(True)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        env_ids_tensor = _env_ids_to_tensor(env_ids, self.num_envs, self.device)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        target_joint_pos = self.command_pos[env_ids]
        position_noise = 0.02 * torch.randn_like(target_joint_pos)
        reset_joint_pos = torch.clamp(
            target_joint_pos + position_noise,
            min=self.robot.data.soft_joint_pos_limits[env_ids][:,
                                                               self.joint_ids, 0],
            max=self.robot.data.soft_joint_pos_limits[env_ids][:,
                                                               self.joint_ids, 1],
        )
        default_joint_pos[:, self.joint_ids] = reset_joint_pos
        default_joint_vel[:, self.joint_ids] = 0.0
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(
            default_joint_pos, env_ids=env_ids)

        if self.leader is not None:
            leader_joint_pos = self.leader.data.default_joint_pos[env_ids].clone(
            )
            leader_joint_vel = self.leader.data.default_joint_vel[env_ids].clone(
            )
            leader_joint_pos[:, self.leader_joint_ids] = target_joint_pos
            leader_joint_vel[:, self.leader_joint_ids] = 0.0
            self.leader.write_joint_state_to_sim(
                leader_joint_pos, leader_joint_vel, env_ids=env_ids)
            self.leader.set_joint_position_target(
                leader_joint_pos, env_ids=env_ids)

        self._episode_sq_joint_error_sum[env_ids_tensor] = 0.0
        self._episode_sample_count[env_ids_tensor] = 0.0
        self._episode_max_joint_error[env_ids_tensor] = 0.0

        return extras

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.command_pos, self.command_vel), dim=-1)

    def set_difficulty_profile(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        active_joint_count_range: tuple[int, int] | None = None,
        active_joint_count_low: torch.Tensor | int | None = None,
        active_joint_count_high: torch.Tensor | int | None = None,
        position_span_scale: float | torch.Tensor | None = None,
        segment_duration_range_s: tuple[float, float] | None = None,
        segment_duration_min_s: float | torch.Tensor | None = None,
        segment_duration_max_s: float | torch.Tensor | None = None,
    ) -> None:
        """Set the trajectory profile to be used on the next episode reset."""

        env_ids_tensor = _env_ids_to_tensor(env_ids, self.num_envs, self.device)
        if active_joint_count_range is not None:
            low, high = active_joint_count_range
            self._active_joint_count_low[env_ids_tensor] = int(low)
            self._active_joint_count_high[env_ids_tensor] = int(high)
        if active_joint_count_low is not None:
            if isinstance(active_joint_count_low, torch.Tensor):
                self._active_joint_count_low[env_ids_tensor] = active_joint_count_low.to(device=self.device, dtype=torch.int)
            else:
                self._active_joint_count_low[env_ids_tensor] = int(active_joint_count_low)
        if active_joint_count_high is not None:
            if isinstance(active_joint_count_high, torch.Tensor):
                self._active_joint_count_high[env_ids_tensor] = active_joint_count_high.to(device=self.device, dtype=torch.int)
            else:
                self._active_joint_count_high[env_ids_tensor] = int(active_joint_count_high)
        if position_span_scale is not None:
            if isinstance(position_span_scale, torch.Tensor):
                self._position_span_scale[env_ids_tensor] = position_span_scale.to(device=self.device, dtype=torch.float32)
            else:
                self._position_span_scale[env_ids_tensor] = float(position_span_scale)
        if segment_duration_range_s is not None:
            dur_low, dur_high = segment_duration_range_s
            self._segment_duration_min[env_ids_tensor] = float(dur_low)
            self._segment_duration_max[env_ids_tensor] = float(dur_high)
        if segment_duration_min_s is not None:
            if isinstance(segment_duration_min_s, torch.Tensor):
                self._segment_duration_min[env_ids_tensor] = segment_duration_min_s.to(device=self.device, dtype=torch.float32)
            else:
                self._segment_duration_min[env_ids_tensor] = float(segment_duration_min_s)
        if segment_duration_max_s is not None:
            if isinstance(segment_duration_max_s, torch.Tensor):
                self._segment_duration_max[env_ids_tensor] = segment_duration_max_s.to(device=self.device, dtype=torch.float32)
            else:
                self._segment_duration_max[env_ids_tensor] = float(segment_duration_max_s)

    def get_episode_tracking_statistics(self, env_ids: Sequence[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return episode tracking aggregates for curriculum updates."""

        env_ids_tensor = _env_ids_to_tensor(env_ids, self.num_envs, self.device)
        return {
            "joint_rmse": compute_episode_joint_rmse(
                self._episode_sq_joint_error_sum[env_ids_tensor],
                self._episode_sample_count[env_ids_tensor],
            ),
            "sample_count": self._episode_sample_count[env_ids_tensor],
            "max_joint_error": self._episode_max_joint_error[env_ids_tensor],
        }

    def get_episode_joint_rmse(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return episode RMSE for the requested environments."""

        return self.get_episode_tracking_statistics(env_ids)["joint_rmse"]

    def _update_metrics(self):
        self.metrics["target_joint_speed"] = torch.linalg.norm(self.command_vel, dim=-1)
        joint_error = self.command_pos - self.robot.data.joint_pos[:, self.joint_ids]
        mean_sq_joint_error = torch.mean(torch.square(joint_error), dim=-1)
        max_joint_error = torch.max(torch.abs(joint_error), dim=-1).values
        self._episode_sq_joint_error_sum += mean_sq_joint_error
        self._episode_sample_count += 1.0
        self._episode_max_joint_error = torch.maximum(self._episode_max_joint_error, max_joint_error)
        self.metrics["episode_joint_rmse"] = compute_episode_joint_rmse(
            self._episode_sq_joint_error_sum,
            self._episode_sample_count,
        )
        self.metrics["episode_max_joint_error"] = self._episode_max_joint_error

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
        lower_limits = limits[..., 0] + self.cfg.waypoint_limit_margin
        upper_limits = limits[..., 1] - self.cfg.waypoint_limit_margin
        safe_lower = torch.minimum(lower_limits, upper_limits)
        safe_upper = torch.maximum(lower_limits, upper_limits)

        current_joint_pos = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
        start_pos = torch.where(
            self._has_active_segment[env_ids].unsqueeze(-1),
            self.command_pos[env_ids],
            current_joint_pos,
        )
        end_pos, _ = sample_waypoint_targets(
            start_pos=start_pos,
            safe_lower=safe_lower,
            safe_upper=safe_upper,
            active_joint_count_low=self._active_joint_count_low[env_ids],
            active_joint_count_high=self._active_joint_count_high[env_ids],
            position_span_scale=self._position_span_scale[env_ids],
        )
        sampled_duration = sample_duration_range(
            self._segment_duration_min[env_ids],
            self._segment_duration_max[env_ids],
        )
        sampled_duration = torch.clamp(sampled_duration, min=self._env.step_dt)
        # Keep CommandTerm's internal timer aligned with the per-episode duration
        # so resampling cadence matches the curriculum-selected trajectory speed.
        self.time_left[env_ids] = sampled_duration

        self._segment_start[env_ids] = start_pos
        self._segment_end[env_ids] = end_pos
        self._segment_duration[env_ids] = sampled_duration
        self._segment_elapsed[env_ids] = 0.0
        self._has_active_segment[env_ids] = True

        self.command_pos[env_ids] = start_pos
        self.command_vel[env_ids] = 0.0

    def _update_command(self):
        duration = self._segment_duration.unsqueeze(-1)
        phase = torch.clamp(self._segment_elapsed.unsqueeze(-1) / duration, min=0.0, max=1.0)
        blend = phase * phase * (3.0 - 2.0 * phase)
        blend_rate = (6.0 * phase * (1.0 - phase)) / duration
        delta = self._segment_end - self._segment_start
        self.command_pos[:] = self._segment_start + delta * blend
        self.command_vel[:] = delta * blend_rate
        if self.leader is not None and self.leader_joint_ids is not None:
            self.leader.write_joint_state_to_sim(
                position=self.command_pos,
                velocity=self.command_vel,
                joint_ids=self.leader_joint_ids,
            )
            self.leader.set_joint_position_target(
                self.command_pos, joint_ids=self.leader_joint_ids)
        self._segment_elapsed += self._env.step_dt
