"""Shared helpers for evaluation and play entrypoints."""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from so101_hackathon.evaluation.metrics import TeleopMetricAccumulator


@dataclass
class EvaluationResult:
    """Container returned by the pure rollout helper."""

    metrics: dict[str, float]
    num_steps: int


@contextlib.contextmanager
def evaluation_progress_bar(total_steps: int | None, *, enabled: bool):
    """Create a tqdm progress bar when available."""

    if not enabled:
        yield None
        return

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        yield None
        return

    progress = tqdm(total=total_steps, desc="Evaluating", unit="step")
    try:
        yield progress
    finally:
        progress.close()


def get_initial_observation(env: Any) -> Any:
    """Read the initial observation from either vector or gym-style envs."""

    if hasattr(env, "get_observations"):
        observation = env.get_observations()
    else:
        observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    return observation


def coerce_done_flag(done: Any) -> bool:
    """Convert env-specific done outputs into a plain Python bool."""

    if isinstance(done, bool):
        return done
    if hasattr(done, "numel") and hasattr(done, "any"):
        return bool(done.any().item())
    if hasattr(done, "item"):
        return bool(done.item())
    return bool(done)


def extract_env_step_metrics(env: Any) -> dict[str, Any]:
    """Extract teleop metrics directly from the wrapped Isaac Lab environment."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "Real environment metric extraction requires `torch`."
        ) from exc

    from isaaclab.managers import SceneEntityCfg
    from so101_hackathon.sim import mdp as so101_mdp
    from so101_hackathon.sim.kinematics import compute_so101_ee_pose

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    body_ids, _ = robot.find_bodies("gripper_link")
    ee_body_idx = body_ids[0]
    action_term = base_env.action_manager.get_term("arm_action")
    termination_cfg = getattr(getattr(base_env, "cfg", None), "terminations", None)
    try:
        leader_robot = base_env.scene["leader_robot"]
    except Exception:
        leader_robot = None

    if leader_robot is not None:
        leader_joint_ids, _ = leader_robot.find_joints(
            action_term._joint_names, preserve_order=True
        )
        target_joint_pos = leader_robot.data.joint_pos[:, leader_joint_ids]
    else:
        target_joint_pos = so101_mdp.command_joint_positions(base_env, command_name="leader_joints")
    follower_joint_pos = robot.data.joint_pos[:, action_term._joint_ids]
    joint_error = target_joint_pos - follower_joint_pos

    target_ee_pos, target_ee_quat = compute_so101_ee_pose(
        target_joint_pos, joint_names=action_term._joint_names
    )
    actual_ee_pos = robot.data.body_pos_w[:, ee_body_idx] - base_env.scene.env_origins
    actual_ee_quat = robot.data.body_quat_w[:, ee_body_idx]

    ee_position_error = torch.linalg.norm(target_ee_pos - actual_ee_pos, dim=-1)
    quat_alignment = torch.sum(actual_ee_quat * target_ee_quat, dim=-1).abs().clamp(max=1.0)
    ee_orientation_error = 2.0 * torch.arccos(quat_alignment)
    action_rate = torch.linalg.norm(
        base_env.action_manager.action - base_env.action_manager.prev_action, dim=-1
    )
    invalid_state = ~torch.isfinite(joint_error).all(dim=-1)
    controlled_joint_ids = action_term._joint_ids
    follower_joint_vel = robot.data.joint_vel[:, controlled_joint_ids]
    lower_limits = robot.data.soft_joint_pos_limits[:, controlled_joint_ids, 0] - 0.02
    upper_limits = robot.data.soft_joint_pos_limits[:, controlled_joint_ids, 1] + 0.02
    collision = torch.zeros_like(invalid_state)
    excessive_joint_error = torch.zeros_like(invalid_state)
    joint_limit_violation = torch.zeros_like(invalid_state)
    unstable_joint_velocity = torch.zeros_like(invalid_state)

    if getattr(termination_cfg, "collision", None) is not None:
        collision = so101_mdp.illegal_contact(
            base_env,
            threshold=5.0,
            sensor_cfg=SceneEntityCfg(
                "arm_contact",
                body_names=[
                    "shoulder_link",
                    "upper_arm_link",
                    "lower_arm_link",
                    "wrist_link",
                    "gripper_link",
                    "moving_jaw_so101_v1_link",
                ],
            ),
        )
    if getattr(termination_cfg, "excessive_joint_error", None) is not None:
        excessive_joint_error = torch.any(torch.abs(joint_error) > 0.75, dim=-1)
    if getattr(termination_cfg, "joint_limit_violation", None) is not None:
        joint_limit_violation = torch.any(
            (follower_joint_pos < lower_limits) | (follower_joint_pos > upper_limits), dim=-1
        )
    if getattr(termination_cfg, "unstable_joint_velocity", None) is not None:
        unstable_joint_velocity = torch.any(torch.abs(follower_joint_vel) > 2.0, dim=-1) | (
            ~torch.isfinite(follower_joint_vel).all(dim=-1)
        )
    failure = collision | excessive_joint_error | joint_limit_violation | unstable_joint_velocity

    return {
        "joint_error": joint_error[0].detach().cpu().tolist(),
        "action_rate": float(action_rate[0].detach().cpu().item()),
        "ee_position_error": float(ee_position_error[0].detach().cpu().item()),
        "ee_orientation_error": float(ee_orientation_error[0].detach().cpu().item()),
        "invalid_state": bool(invalid_state[0].detach().cpu().item()),
        "failure": bool(failure[0].detach().cpu().item()),
    }


def evaluate_controller(
    env: Any,
    controller: Any,
    *,
    num_episodes: int,
    real_time: bool = False,
    metric_accumulator: TeleopMetricAccumulator | None = None,
    step_callback: Any | None = None,
    show_progress: bool = True,
    simulation_app: Any | None = None,
) -> EvaluationResult:
    """Run a shared evaluation loop for any controller."""

    metrics = metric_accumulator or TeleopMetricAccumulator()
    metrics.reset_episode()

    vector_env = hasattr(env, "get_observations")
    observation = get_initial_observation(env)
    controller.reset()

    episode_count = 0
    step_count = 0
    max_steps: int | None = None
    dt = getattr(env, "step_dt", getattr(getattr(env, "unwrapped", None), "step_dt", 0.0))
    max_episode_length = getattr(getattr(env, "unwrapped", None), "max_episode_length", None)
    if max_episode_length is not None:
        max_steps = int(max_episode_length) * num_episodes

    with evaluation_progress_bar(max_steps, enabled=show_progress) as progress:
        while (
            episode_count < num_episodes
            and (simulation_app is None or simulation_app.is_running())
            and (max_steps is None or step_count < max_steps)
        ):
            start_time = time.time()
            action = controller.act(observation)
            step_out = env.step(action)

            if vector_env:
                observation, reward, done, info = step_out
                done = coerce_done_flag(done)
            elif len(step_out) == 5:
                observation, reward, terminated, truncated, info = step_out
                done = coerce_done_flag(terminated) or coerce_done_flag(truncated)
            else:
                observation, reward, done, info = step_out
                done = coerce_done_flag(done)

            step_count += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix(episodes=episode_count, steps=step_count)

            step_metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
            if not step_metrics and hasattr(getattr(env, "unwrapped", None), "command_manager"):
                step_metrics = extract_env_step_metrics(env)
            if step_metrics:
                metrics.add_step(
                    joint_error=step_metrics.get("joint_error", [0.0] * metrics.joint_count),
                    action_rate=step_metrics.get("action_rate", 0.0),
                    ee_position_error=step_metrics.get("ee_position_error", 0.0),
                    ee_orientation_error=step_metrics.get("ee_orientation_error", 0.0),
                    invalid_state=bool(step_metrics.get("invalid_state", False)),
                    failure=bool(step_metrics.get("failure", False)),
                )

            if step_callback is not None:
                step_callback(step_count=step_count, reward=reward, done=done, info=info)

            if done:
                metrics.finish_episode()
                episode_count += 1
                if progress is not None:
                    progress.set_postfix(episodes=episode_count, steps=step_count)
                if episode_count >= num_episodes:
                    break
                if not vector_env:
                    observation = get_initial_observation(env)
                controller.reset()

            if real_time and dt > 0.0:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    return EvaluationResult(metrics=metrics.summary(), num_steps=step_count)


def add_app_launcher_args(parser: argparse.ArgumentParser) -> bool:
    """Add Isaac AppLauncher args when the runtime is available."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--headless", action="store_true", default=False)
        return False

    AppLauncher.add_app_launcher_args(parser)
    return True


def default_output_dir() -> str:
    """Return the fallback output dir when no checkpoint-owned run exists."""

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.abspath(os.path.join("logs", "evaluation", stamp))


def checkpoint_run_dir(checkpoint_path: str | None) -> str | None:
    """Return the trained-agent run directory that owns the checkpoint."""

    if not checkpoint_path:
        return None
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        return None
    return os.path.dirname(checkpoint_path)


def resolve_evaluation_output_dir(
    *,
    controller_name: str,
    requested_output_dir: str | None,
    checkpoint_path: str | None,
) -> str:
    """Resolve the evaluation run directory."""

    if requested_output_dir:
        return os.path.abspath(requested_output_dir)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if controller_name == "ppo":
        run_dir = checkpoint_run_dir(checkpoint_path)
        if run_dir is not None:
            return os.path.join(run_dir, "evaluation", stamp)
    return os.path.abspath(os.path.join("logs", controller_name, "evaluation", stamp))


def load_yaml(path: str | None) -> dict[str, Any]:
    """Load an optional YAML config file."""

    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object in {path}, received {type(data)}")
    return data


def tensorboard_log_dir(output_dir: str) -> str:
    """Return the TensorBoard directory for an evaluation run."""

    return os.path.join(output_dir, "tensorboard")


def write_evaluation_config(
    *,
    output_dir: str,
    args: argparse.Namespace,
    env_config: dict[str, Any],
    controller_config: dict[str, Any],
) -> str:
    """Persist the effective evaluation configuration."""

    config_path = os.path.join(output_dir, "config.json")
    payload = {
        "controller": args.controller,
        "args": vars(args).copy(),
        "env_config": env_config,
        "controller_config": controller_config,
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return config_path


def log_evaluation_metrics(
    *,
    controller_name: str,
    output_dir: str,
    result: EvaluationResult,
    controller_config: dict[str, Any],
) -> None:
    """Write summary scalars to TensorBoard."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "Evaluation metric logging requires TensorBoard support from `torch`."
        ) from exc

    checkpoint_path = controller_config.get("checkpoint_path")
    global_step = result.num_steps
    tb_dir = tensorboard_log_dir(output_dir)
    writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
    try:
        for name, value in sorted(result.metrics.items()):
            writer.add_scalar(name, value, global_step)
        writer.add_text("eval/controller", str(controller_name), global_step)
        if checkpoint_path:
            writer.add_text("eval/checkpoint_path", str(checkpoint_path), global_step)
    finally:
        writer.close()

    print(f"[INFO] Evaluation metrics logged to TensorBoard: {tb_dir}")


def build_evaluation_payload(
    *,
    controller_name: str,
    output_dir: str,
    config_path: str,
    result: EvaluationResult,
    video_dir: str,
    include_video: bool,
) -> dict[str, Any]:
    """Build the evaluation summary payload."""

    payload = {
        "controller": controller_name,
        "metrics": result.metrics,
        "num_steps": result.num_steps,
        "output_dir": output_dir,
        "tensorboard_dir": tensorboard_log_dir(output_dir),
        "config_path": config_path,
    }
    if include_video:
        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        payload["video"] = video_files[-1] if video_files else None
    return payload


def write_summary_json(output_dir: str, payload: dict[str, Any]) -> str:
    """Write the evaluation summary file."""

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path
