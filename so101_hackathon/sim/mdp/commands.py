"""Task-space leader command terms for SO101 teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from so101_hackathon.sim.kinematics import compute_so101_ee_jacobian, compute_so101_ee_position
from so101_hackathon.sim.robots.so101_follower_spec import SO101_EE_BODY_NAME
from so101_hackathon.deploy.runtime import DEFAULT_NOISE_JOINT_INDICES

from .adaptive_curriculum_utils import compute_episode_joint_rmse, sample_duration_range


def _env_ids_to_tensor(env_ids: Sequence[int] | slice | torch.Tensor, num_envs: int, device: torch.device) -> torch.Tensor:
    """Convert Isaac Lab env-id selections into a dense index tensor."""

    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


@configclass
class TaskSpaceLeaderRangesCfg:
    """Workspace ranges for the end-effector position command."""

    pos_x: tuple[float, float] = (0.05, 0.8)
    pos_y: tuple[float, float] = (-0.8, 0.8)
    pos_z: tuple[float, float] = (0.05, 0.8)


@configclass
class TaskSpaceLeaderCommandCfg(CommandTermCfg):
    """Configuration for task-space SO101 leader trajectories."""

    class_type: type[CommandTerm] = MISSING  # type: ignore
    asset_name: str = "robot"
    body_name: str = SO101_EE_BODY_NAME
    joint_names: list[str] = MISSING  # type: ignore
    gripper_joint_name: str = "gripper"
    preserve_order: bool = True
    waypoint_limit_margin: float = 0.05
    position_span_scale: float = 1.0
    ik_damping: float = 0.05
    ik_fd_epsilon: float = 1.0e-3
    ik_max_iterations: int = 3
    ik_step_size: float = 1.0
    ranges: TaskSpaceLeaderRangesCfg = TaskSpaceLeaderRangesCfg()

    def __post_init__(self):
        """Finalize dataclass initialization."""
        self.class_type = TaskSpaceLeaderCommand


class TaskSpaceLeaderCommand(CommandTerm):
    """Generate smooth task-space leader trajectories and solve them into joint targets.

    The command is sampled in end-effector Cartesian space and converted into a
    6-joint target using a lightweight batched damped-least-squares IK loop.
    """

    cfg: TaskSpaceLeaderCommandCfg

    def __init__(self, cfg: TaskSpaceLeaderCommandCfg, env):
        """Initialize the object."""
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.joint_ids, self.joint_names = self.robot.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self.num_joints = len(self.joint_ids)
        self.gripper_local_idx = self.joint_names.index(cfg.gripper_joint_name)
        self.gripper_joint_id = self.joint_ids[self.gripper_local_idx]
        self.arm_local_indices = [index for index, name in enumerate(
            self.joint_names) if name != cfg.gripper_joint_name]
        self.arm_joint_names = [self.joint_names[index]
                                for index in self.arm_local_indices]

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
            self._bind_leader_visual_override()
        else:
            self.leader_joint_ids = None

        default_joint_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone(
        )
        self.target_joint_pos = default_joint_pos.clone()
        self.target_joint_vel = torch.zeros_like(default_joint_pos)
        self.command_position = compute_so101_ee_position(
            default_joint_pos[:, self.arm_local_indices],
            joint_names=self.arm_joint_names,
        )
        self.command_gripper = default_joint_pos[:,
                                                 self.gripper_local_idx: self.gripper_local_idx + 1].clone()
        self._segment_start_position = self.command_position.clone()
        self._segment_end_position = self.command_position.clone()
        self._segment_start_gripper = self.command_gripper.clone()
        self._segment_end_gripper = self.command_gripper.clone()
        self._segment_start_joint_pos = default_joint_pos.clone()
        self._segment_end_joint_pos = default_joint_pos.clone()
        self._segment_duration = torch.full(
            (self.num_envs,),
            float(cfg.resampling_time_range[0]),
            device=self.device,
            dtype=torch.float32,
        )
        self._segment_elapsed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32)
        self._has_active_segment = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._position_span_scale = torch.full(
            (self.num_envs,),
            float(cfg.position_span_scale),
            device=self.device,
            dtype=torch.float32,
        )
        self._segment_duration_min = torch.full(
            (self.num_envs,),
            float(cfg.resampling_time_range[0]),
            device=self.device,
            dtype=torch.float32,
        )
        self._segment_duration_max = torch.full(
            (self.num_envs,),
            float(cfg.resampling_time_range[1]),
            device=self.device,
            dtype=torch.float32,
        )
        self._workspace_lower = torch.tensor(
            [cfg.ranges.pos_x[0], cfg.ranges.pos_y[0], cfg.ranges.pos_z[0]],
            device=self.device,
            dtype=torch.float32,
        )
        self._workspace_upper = torch.tensor(
            [cfg.ranges.pos_x[1], cfg.ranges.pos_y[1], cfg.ranges.pos_z[1]],
            device=self.device,
            dtype=torch.float32,
        )
        self._episode_sq_joint_error_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_sample_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_max_joint_error = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        """Return the current task-space command `(x, y, z, gripper)`."""

        return torch.cat((self.command_position, self.command_gripper), dim=-1)

    @property
    def target_joint_positions(self) -> torch.Tensor:
        """Run target joint positions."""
        return self.target_joint_pos

    @property
    def target_joint_velocities(self) -> torch.Tensor:
        """Run target joint velocities."""
        return self.target_joint_vel

    @property
    def target_ee_position(self) -> torch.Tensor:
        """Run target ee position."""
        return self.command_position

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
        """Reset internal state."""
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)

        self._segment_start_joint_pos[env_ids_tensor] = self.target_joint_pos[env_ids_tensor]
        self._segment_end_joint_pos[env_ids_tensor] = self.target_joint_pos[env_ids_tensor]

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        target_joint_pos = self.target_joint_pos[env_ids]
        position_noise = torch.zeros_like(target_joint_pos)
        noise_joint_indices = [
            index for index in DEFAULT_NOISE_JOINT_INDICES
            if 0 <= int(index) < position_noise.shape[-1]
        ]
        if noise_joint_indices:
            position_noise[:, noise_joint_indices] = 0.02 * \
                torch.randn_like(position_noise[:, noise_joint_indices])
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
        """Compatibility hook for the adaptive curriculum interface."""

        del active_joint_count_range, active_joint_count_low, active_joint_count_high
        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        if position_span_scale is not None:
            if isinstance(position_span_scale, torch.Tensor):
                self._position_span_scale[env_ids_tensor] = position_span_scale.to(
                    device=self.device, dtype=torch.float32)
            else:
                self._position_span_scale[env_ids_tensor] = float(
                    position_span_scale)
        if segment_duration_range_s is not None:
            duration_min, duration_max = segment_duration_range_s
            self._segment_duration_min[env_ids_tensor] = float(duration_min)
            self._segment_duration_max[env_ids_tensor] = float(duration_max)
        if segment_duration_min_s is not None:
            if isinstance(segment_duration_min_s, torch.Tensor):
                self._segment_duration_min[env_ids_tensor] = segment_duration_min_s.to(
                    device=self.device, dtype=torch.float32)
            else:
                self._segment_duration_min[env_ids_tensor] = float(
                    segment_duration_min_s)
        if segment_duration_max_s is not None:
            if isinstance(segment_duration_max_s, torch.Tensor):
                self._segment_duration_max[env_ids_tensor] = segment_duration_max_s.to(
                    device=self.device, dtype=torch.float32)
            else:
                self._segment_duration_max[env_ids_tensor] = float(
                    segment_duration_max_s)

    def get_episode_tracking_statistics(self, env_ids: Sequence[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return episode tracking aggregates for curriculum updates."""

        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        return {
            "joint_rmse": compute_episode_joint_rmse(
                self._episode_sq_joint_error_sum[env_ids_tensor],
                self._episode_sample_count[env_ids_tensor],
            ),
            "sample_count": self._episode_sample_count[env_ids_tensor],
            "max_joint_error": self._episode_max_joint_error[env_ids_tensor],
        }

    def get_episode_joint_rmse(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return episode joint rmse."""
        return self.get_episode_tracking_statistics(env_ids)["joint_rmse"]

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["target_joint_speed"] = torch.linalg.norm(
            self.target_joint_vel, dim=-1)
        joint_error = self.target_joint_pos - \
            self.robot.data.joint_pos[:, self.joint_ids]
        mean_sq_joint_error = torch.mean(torch.square(joint_error), dim=-1)
        max_joint_error = torch.max(torch.abs(joint_error), dim=-1).values
        self._episode_sq_joint_error_sum += mean_sq_joint_error
        self._episode_sample_count += 1.0
        self._episode_max_joint_error = torch.maximum(
            self._episode_max_joint_error, max_joint_error)
        self.metrics["episode_joint_rmse"] = compute_episode_joint_rmse(
            self._episode_sq_joint_error_sum,
            self._episode_sample_count,
        )
        self.metrics["episode_max_joint_error"] = self._episode_max_joint_error

    def _resample_command(self, env_ids: Sequence[int]):
        """Handle resample command."""
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)

        if self._has_active_segment[env_ids_tensor].any():
            start_position = torch.where(
                self._has_active_segment[env_ids_tensor].unsqueeze(-1),
                self.command_position[env_ids_tensor],
                compute_so101_ee_position(
                    self.target_joint_pos[env_ids_tensor][:,
                                                          self.arm_local_indices],
                    joint_names=self.arm_joint_names,
                ),
            )
            start_gripper = torch.where(
                self._has_active_segment[env_ids_tensor].unsqueeze(-1),
                self.command_gripper[env_ids_tensor],
                self.target_joint_pos[env_ids_tensor][:,
                                                      self.gripper_local_idx: self.gripper_local_idx + 1],
            )
        else:
            start_position = compute_so101_ee_position(
                self.target_joint_pos[env_ids_tensor][:,
                                                      self.arm_local_indices],
                joint_names=self.arm_joint_names,
            )
            start_gripper = self.target_joint_pos[env_ids_tensor][:,
                                                                  self.gripper_local_idx: self.gripper_local_idx + 1]

        end_position = self._sample_position_targets(
            start_position, env_ids_tensor)
        end_gripper = self._sample_gripper_targets(
            start_gripper, env_ids_tensor)
        sampled_duration = sample_duration_range(
            self._segment_duration_min[env_ids_tensor],
            self._segment_duration_max[env_ids_tensor],
        )
        sampled_duration = torch.clamp(sampled_duration, min=self._env.step_dt)
        self.time_left[env_ids_tensor] = sampled_duration

        self._segment_start_position[env_ids_tensor] = start_position
        self._segment_end_position[env_ids_tensor] = end_position
        self._segment_start_gripper[env_ids_tensor] = start_gripper
        self._segment_end_gripper[env_ids_tensor] = end_gripper
        self._segment_duration[env_ids_tensor] = sampled_duration
        self._segment_elapsed[env_ids_tensor] = 0.0
        self._has_active_segment[env_ids_tensor] = True
        self.command_position[env_ids_tensor] = start_position
        self.command_gripper[env_ids_tensor] = start_gripper
        self._segment_start_joint_pos[env_ids_tensor] = self.target_joint_pos[env_ids_tensor]
        self._segment_end_joint_pos[env_ids_tensor] = self._solve_end_joint_targets(
            env_ids_tensor=env_ids_tensor,
            start_joint_pos=self.target_joint_pos[env_ids_tensor],
            desired_position=end_position,
            desired_gripper=end_gripper,
        )

    def _update_command(self):
        """Update command."""
        duration = self._segment_duration.unsqueeze(-1)
        phase = torch.clamp(
            self._segment_elapsed.unsqueeze(-1) / duration, min=0.0, max=1.0)
        blend = phase * phase * (3.0 - 2.0 * phase)
        blend_rate = (6.0 * phase * (1.0 - phase)) / duration
        self.command_position[:] = self._segment_start_position + \
            (self._segment_end_position - self._segment_start_position) * blend
        self.command_gripper[:] = self._segment_start_gripper + \
            (self._segment_end_gripper - self._segment_start_gripper) * blend
        self.target_joint_pos[:] = self._segment_start_joint_pos + \
            (self._segment_end_joint_pos - self._segment_start_joint_pos) * blend
        self.target_joint_vel[:] = (
            self._segment_end_joint_pos - self._segment_start_joint_pos
        ) * blend_rate
        if self.leader is not None and self.leader_joint_ids is not None:
            self.leader.set_joint_position_target(
                self.target_joint_pos, joint_ids=self.leader_joint_ids)
        self._segment_elapsed += self._env.step_dt

    def _sample_position_targets(self, start_position: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample position targets."""
        workspace_lower = self._workspace_lower.unsqueeze(0)
        workspace_upper = self._workspace_upper.unsqueeze(0)
        usable_range = workspace_upper - workspace_lower
        span_scale = torch.clamp(
            self._position_span_scale[env_ids], min=0.0, max=1.0).unsqueeze(-1)
        half_window = 0.5 * usable_range * span_scale
        local_lower = torch.maximum(
            start_position - half_window, workspace_lower)
        local_upper = torch.minimum(
            start_position + half_window, workspace_upper)
        return local_lower + (local_upper - local_lower) * torch.rand_like(start_position)

    def _sample_gripper_targets(self, start_gripper: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample gripper targets."""
        limits = self.robot.data.soft_joint_pos_limits[env_ids,
                                                       self.gripper_joint_id]
        lower_limit = limits[:, 0:1]
        upper_limit = limits[:, 1:2]
        span_scale = torch.clamp(
            self._position_span_scale[env_ids], min=0.0, max=1.0).unsqueeze(-1)
        half_window = 0.5 * (upper_limit - lower_limit) * span_scale
        local_lower = torch.maximum(start_gripper - half_window, lower_limit)
        local_upper = torch.minimum(start_gripper + half_window, upper_limit)
        return local_lower + (local_upper - local_lower) * torch.rand_like(start_gripper)

    def _solve_end_joint_targets(
        self,
        *,
        env_ids_tensor: torch.Tensor,
        start_joint_pos: torch.Tensor,
        desired_position: torch.Tensor,
        desired_gripper: torch.Tensor,
    ) -> torch.Tensor:
        """Handle solve end joint targets."""
        arm_joint_pos = start_joint_pos[:, self.arm_local_indices].clone()
        lower_limits = self.robot.data.soft_joint_pos_limits[env_ids_tensor][:,
                                                                             self.joint_ids, 0][:, self.arm_local_indices]
        upper_limits = self.robot.data.soft_joint_pos_limits[env_ids_tensor][:,
                                                                             self.joint_ids, 1][:, self.arm_local_indices]
        for _ in range(int(self.cfg.ik_max_iterations)):
            current_position = compute_so101_ee_position(
                arm_joint_pos, joint_names=self.arm_joint_names)
            jacobian = self._compute_position_jacobian(
                arm_joint_pos, current_position)
            position_error = desired_position - current_position
            delta_joint_pos = self._solve_dls(position_error, jacobian)
            arm_joint_pos = torch.clamp(
                arm_joint_pos + float(self.cfg.ik_step_size) * delta_joint_pos,
                min=lower_limits,
                max=upper_limits,
            )
        end_joint_pos = start_joint_pos.clone()
        end_joint_pos[:, self.arm_local_indices] = arm_joint_pos
        end_joint_pos[:, self.gripper_local_idx: self.gripper_local_idx +
                      1] = desired_gripper
        return end_joint_pos

    def _compute_position_jacobian(self, arm_joint_pos: torch.Tensor, current_position: torch.Tensor) -> torch.Tensor:
        """Compute position jacobian."""
        del current_position
        return compute_so101_ee_jacobian(arm_joint_pos, joint_names=self.arm_joint_names)

    def _solve_dls(self, position_error: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Handle solve dls."""
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_sq = float(self.cfg.ik_damping) ** 2
        lambda_matrix = lambda_sq * \
            torch.eye(3, device=self.device, dtype=jacobian.dtype).unsqueeze(0)
        delta = jacobian_t @ torch.linalg.solve(
            jacobian @ jacobian_t + lambda_matrix, position_error.unsqueeze(-1))
        return delta.squeeze(-1)
