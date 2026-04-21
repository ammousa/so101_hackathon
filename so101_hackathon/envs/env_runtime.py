"""Lightweight runtime helpers safe to import before Isaac starts."""

from __future__ import annotations


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device: str) -> None:
    if teleop_device != "so101leader":
        return
    write_gripper_effort_limit_sim(env, env.scene["robot"])


def write_gripper_effort_limit_sim(env, env_arm) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError("Runtime gripper effort updates require `torch`.") from exc

    gripper_pos = env_arm.data.body_link_pos_w[:, -1]

    object_positions = []
    object_masses = []
    for obj in env.scene._rigid_objects.values():
        object_positions.append(obj.data.body_link_pos_w[:, 0])
        object_masses.append(obj.data.default_mass)
    if not object_positions:
        return

    object_positions = torch.stack(object_positions)
    object_masses = torch.stack(object_masses)
    distances = torch.sqrt(torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))
    _, min_indices = torch.min(distances, dim=0)
    target_masses = object_masses[min_indices.cpu(), 0, 0]
    target_effort_limits = (target_masses / 0.15).to(env_arm._data.joint_effort_limits.device)

    current_effort_limit_sim = env_arm._data.joint_effort_limits[:, -1]
    need_update = torch.abs(target_effort_limits - current_effort_limit_sim) > 0.1
    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]
        env_arm.write_joint_effort_limit_to_sim(limits=new_limits, joint_ids=[5 for _ in range(env.num_envs)])
