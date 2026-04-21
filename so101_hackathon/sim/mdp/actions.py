"""Custom action terms for SO101 teleoperation tasks."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg  # type: ignore
from isaaclab.envs.mdp.actions.joint_actions import JointAction  # type: ignore
from isaaclab.managers import ActionTerm  # type: ignore
from isaaclab.utils import DelayBuffer, configclass  # type: ignore

from so101_hackathon.sim.teleop_utils import compose_residual_joint_commands

from .adaptive_curriculum_utils import resolve_disturbance_reset_values


def _env_ids_to_tensor(env_ids: Sequence[int] | slice | torch.Tensor, num_envs: int, device: torch.device) -> torch.Tensor:
    """Convert Isaac Lab env-id selections into a dense index tensor."""

    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Configuration for residual SO101 joint position commands."""

    class_type: type[ActionTerm] = MISSING  # type: ignore
    command_name: str = "leader_joints"
    max_delay: int = 8
    delay_range: tuple[int, int] = (0, 0)
    noise_std_range: tuple[float, float] = (0.0, 0.0)
    noise_joint_indices: tuple[int, ...] = (0, 1, 2, 3)
    fixed_delay_steps: int | None = None
    fixed_noise_std: float | None = None

    def __post_init__(self):
        self.class_type = ResidualJointPositionAction


@configclass
class AbsoluteJointPositionActionCfg(JointActionCfg):
    """Configuration for absolute SO101 joint position commands with delay/noise disturbance."""

    class_type: type[ActionTerm] = MISSING  # type: ignore
    max_delay: int = 8
    delay_range: tuple[int, int] = (0, 0)
    noise_std_range: tuple[float, float] = (0.0, 0.0)
    noise_joint_indices: tuple[int, ...] = (0, 1, 2, 3)
    fixed_delay_steps: int | None = None
    fixed_noise_std: float | None = None

    def __post_init__(self):
        self.class_type = AbsoluteJointPositionAction


class ResidualJointPositionAction(JointAction):
    """Apply policy outputs as residual corrections on top of joint targets."""

    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._offset = 0.0
        self._applied_actions = torch.zeros_like(self.raw_actions)
        self._delay_buffer = DelayBuffer(
            cfg.max_delay, batch_size=self.num_envs, device=self.device)
        self._delay_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)
        self._noise_std = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self._delay_range = cfg.delay_range
        self._noise_std_range = cfg.noise_std_range
        self._fixed_delay_steps = cfg.fixed_delay_steps
        self._fixed_noise_std = cfg.fixed_noise_std
        self._curriculum_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)
        self._curriculum_noise_std = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self._has_curriculum_disturbance = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._noise_mask = torch.zeros(
            (1, self.action_dim), dtype=torch.float32, device=self.device)
        for joint_index in cfg.noise_joint_indices:
            if 0 <= int(joint_index) < self.action_dim:
                self._noise_mask[0, int(joint_index)] = 1.0

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._applied_actions

    @property
    def delay_steps(self) -> torch.Tensor:
        return self._delay_steps

    @property
    def noise_std(self) -> torch.Tensor:
        return self._noise_std

    def set_disturbance_ranges(self, delay_range: tuple[int, int], noise_std_range: tuple[float, float]) -> None:
        self._delay_range = delay_range
        self._noise_std_range = noise_std_range

    def set_disturbance_override(self, delay_steps: int | None, noise_std: float | None) -> None:
        self._fixed_delay_steps = delay_steps
        self._fixed_noise_std = noise_std

    def set_episode_disturbance(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        delay_steps: torch.Tensor | int,
        noise_std: torch.Tensor | float,
    ) -> None:
        """Provide per-env disturbance samples to be consumed on the next reset."""

        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        if isinstance(delay_steps, torch.Tensor):
            self._curriculum_delay_steps[env_ids_tensor] = delay_steps.to(
                device=self.device, dtype=torch.int)
        else:
            self._curriculum_delay_steps[env_ids_tensor] = int(delay_steps)
        if isinstance(noise_std, torch.Tensor):
            self._curriculum_noise_std[env_ids_tensor] = noise_std.to(
                device=self.device, dtype=torch.float32)
        else:
            self._curriculum_noise_std[env_ids_tensor] = float(noise_std)
        self._has_curriculum_disturbance[env_ids_tensor] = True

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        command_term = self._env.command_manager.get_term(
            self.cfg.command_name)
        if hasattr(command_term, "target_joint_positions"):
            target_positions = command_term.target_joint_positions
        else:
            command = self._env.command_manager.get_command(
                self.cfg.command_name)
            target_positions = command[:, : self.action_dim]
        delayed_commands = self._delay_buffer.compute(
            target_positions + self.processed_actions)
        noise = torch.randn_like(delayed_commands) * \
            self._noise_std.unsqueeze(-1) * self._noise_mask
        lower_limits = self._asset.data.soft_joint_pos_limits[:,
                                                              self._joint_ids, 0]
        upper_limits = self._asset.data.soft_joint_pos_limits[:,
                                                              self._joint_ids, 1]
        self._applied_actions = compose_residual_joint_commands(  # type: ignore
            target_positions=delayed_commands,
            residual_actions=torch.zeros_like(self.processed_actions),
            action_scale=1.0,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            noise=noise,
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._applied_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids  # type: ignore
        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        super().reset(env_ids)
        self._applied_actions[env_ids] = 0.0
        self._delay_buffer.reset(env_ids)
        delay_steps, noise_std = resolve_disturbance_reset_values(
            batch_size=env_ids_tensor.numel(),
            device=self.device,
            delay_range=self._delay_range,
            noise_std_range=self._noise_std_range,
            fixed_delay_steps=self._fixed_delay_steps,
            fixed_noise_std=self._fixed_noise_std,
            has_curriculum_sample=self._has_curriculum_disturbance[env_ids_tensor],
            curriculum_delay_steps=self._curriculum_delay_steps[env_ids_tensor],
            curriculum_noise_std=self._curriculum_noise_std[env_ids_tensor],
        )
        self._delay_steps[env_ids_tensor] = delay_steps
        self._noise_std[env_ids_tensor] = noise_std
        self._delay_buffer.set_time_lag(self._delay_steps[env_ids], env_ids)


class AbsoluteJointPositionAction(JointAction):
    """Apply direct absolute joint position targets with shared delay/noise disturbance."""

    cfg: AbsoluteJointPositionActionCfg

    def __init__(self, cfg: AbsoluteJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._offset = 0.0
        self._applied_actions = torch.zeros_like(self.raw_actions)
        self._delay_buffer = DelayBuffer(
            cfg.max_delay, batch_size=self.num_envs, device=self.device)
        self._delay_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)
        self._noise_std = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self._delay_range = cfg.delay_range
        self._noise_std_range = cfg.noise_std_range
        self._fixed_delay_steps = cfg.fixed_delay_steps
        self._fixed_noise_std = cfg.fixed_noise_std
        self._curriculum_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)
        self._curriculum_noise_std = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self._has_curriculum_disturbance = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._noise_mask = torch.zeros(
            (1, self.action_dim), dtype=torch.float32, device=self.device)
        for joint_index in cfg.noise_joint_indices:
            if 0 <= int(joint_index) < self.action_dim:
                self._noise_mask[0, int(joint_index)] = 1.0

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._applied_actions

    @property
    def delay_steps(self) -> torch.Tensor:
        return self._delay_steps

    @property
    def noise_std(self) -> torch.Tensor:
        return self._noise_std

    def set_disturbance_ranges(self, delay_range: tuple[int, int], noise_std_range: tuple[float, float]) -> None:
        self._delay_range = delay_range
        self._noise_std_range = noise_std_range

    def set_disturbance_override(self, delay_steps: int | None, noise_std: float | None) -> None:
        self._fixed_delay_steps = delay_steps
        self._fixed_noise_std = noise_std

    def set_episode_disturbance(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        delay_steps: torch.Tensor | int,
        noise_std: torch.Tensor | float,
    ) -> None:
        """Provide per-env disturbance samples to be consumed on the next reset."""

        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        if isinstance(delay_steps, torch.Tensor):
            self._curriculum_delay_steps[env_ids_tensor] = delay_steps.to(
                device=self.device, dtype=torch.int)
        else:
            self._curriculum_delay_steps[env_ids_tensor] = int(delay_steps)
        if isinstance(noise_std, torch.Tensor):
            self._curriculum_noise_std[env_ids_tensor] = noise_std.to(
                device=self.device, dtype=torch.float32)
        else:
            self._curriculum_noise_std[env_ids_tensor] = float(noise_std)
        self._has_curriculum_disturbance[env_ids_tensor] = True

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        lower_limits = self._asset.data.soft_joint_pos_limits[:,
                                                              self._joint_ids, 0]
        upper_limits = self._asset.data.soft_joint_pos_limits[:,
                                                              self._joint_ids, 1]
        commanded_positions = torch.clamp(
            self.processed_actions,
            min=lower_limits,
            max=upper_limits,
        )
        delayed_commands = self._delay_buffer.compute(commanded_positions)
        noise = torch.randn_like(delayed_commands) * \
            self._noise_std.unsqueeze(-1) * self._noise_mask
        self._applied_actions = torch.clamp(
            delayed_commands + noise, min=lower_limits, max=upper_limits)

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._applied_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids  # type: ignore
        env_ids_tensor = _env_ids_to_tensor(
            env_ids, self.num_envs, self.device)
        super().reset(env_ids)
        self._applied_actions[env_ids] = 0.0
        self._delay_buffer.reset(env_ids)
        delay_steps, noise_std = resolve_disturbance_reset_values(
            batch_size=env_ids_tensor.numel(),
            device=self.device,
            delay_range=self._delay_range,
            noise_std_range=self._noise_std_range,
            fixed_delay_steps=self._fixed_delay_steps,
            fixed_noise_std=self._fixed_noise_std,
            has_curriculum_sample=self._has_curriculum_disturbance[env_ids_tensor],
            curriculum_delay_steps=self._curriculum_delay_steps[env_ids_tensor],
            curriculum_noise_std=self._curriculum_noise_std[env_ids_tensor],
        )
        self._delay_steps[env_ids_tensor] = delay_steps
        self._noise_std[env_ids_tensor] = noise_std
        self._delay_buffer.set_time_lag(self._delay_steps[env_ids], env_ids)
