"""Deploy a registered controller against a CSV leader trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so101_hackathon.deploy.hardware import (
    DEFAULT_CALIBRATION_DIR,
    load_leader_follower_hardware_dependencies,
)
from so101_hackathon.deploy.metrics import DeployMetricAccumulator
from so101_hackathon.deploy.runtime import (
    DEFAULT_DELAY_STEPS,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_FPS,
    DEFAULT_NOISE_STD,
    DEFAULT_PRINT_EVERY,
    DEFAULT_TELEOP_TIME_S,
    LiveTeleopObservationBuilder,
    build_deploy_config,
    build_follower_action,
    clamp_joint_positions,
    get_joint_limit_vectors,
    hardware_obs_to_joint_positions,
    resolve_deploy_output_dir,
    write_deploy_artifacts,
)
from so101_hackathon.deploy.session import run_deploy_session
from so101_hackathon.deploy.trajectory import CSVJointTrajectory, HardwareTrajectoryLeader
from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.rl_training.runtime_utils import normalize_device_for_runtime
from so101_hackathon.utils.eval_utils import load_yaml
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deploy a registered controller with a CSV leader trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controller",
        choices=list_controller_names(),
        default="raw",
        help="Registered controller to test against the trajectory.",
    )
    parser.add_argument(
        "--controller-config",
        type=str,
        default=None,
        help="Optional YAML file with controller-specific overrides.",
    )
    parser.add_argument(
        "--trajectory-config",
        type=str,
        default="config/traj.yaml",
        help="YAML file with csv_path, frequency_hz, cycles, and return_to_start_steps.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint path forwarded into controller config.",
    )
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument(
        "--disable-follower-gripper",
        action="store_true",
        help="Run deploy without follower joint 6/gripper when that motor is absent or offline.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--teleop-time-s", type=float, default=DEFAULT_TELEOP_TIME_S)
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)
    parser.add_argument("--delay-steps", type=int, default=DEFAULT_DELAY_STEPS)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument(
        "--disturbance-channel",
        choices=["fixed", "ultrazohm"],
        default="fixed",
    )
    parser.add_argument("--uzohm-can-iface", default="can0")
    parser.add_argument("--uzohm-timeout-s", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--controller-coeff",
        "--rl_coeff",
        dest="controller_coeff",
        type=float,
        default=1.0,
        help="Blend between direct trajectory tracking and controller output.",
    )
    return parser


def _safe_disconnect(device, label: str) -> None:
    """Disconnect a hardware device if it was created."""
    if device is None:
        return
    try:
        device.disconnect()
    except Exception as exc:
        print(f"[WARN] {label} disconnect failed: {exc}")


def _create_follower(args: argparse.Namespace):
    """Create the follower hardware without constructing a leader arm."""
    _, _, SOFollower, SOFollowerConfig, precise_sleep = load_leader_follower_hardware_dependencies()
    follower_cfg = SOFollowerConfig(port=args.follower_port)
    follower_cfg.id = args.follower_id
    follower_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR
    setattr(follower_cfg, "disable_gripper", bool(args.disable_follower_gripper))
    follower = SOFollower(follower_cfg)
    if args.disable_follower_gripper:
        follower.bus.motors.pop("gripper", None)
        follower.bus.calibration.pop("gripper", None)
        if hasattr(follower, "calibration") and isinstance(follower.calibration, dict):
            follower.calibration.pop("gripper", None)
    return follower, precise_sleep


def _return_to_start(
    *,
    follower,
    trajectory: CSVJointTrajectory,
    start_joint_pos: list[float],
    lower_limits: list[float],
    upper_limits: list[float],
    sleep_fn,
    fps: int,
    active_follower_joint_names: list[str],
) -> int:
    """Command the follower back to its saved start pose without metrics."""
    steps = int(trajectory.return_to_start_steps)
    if steps <= 0:
        return 0
    target = clamp_joint_positions(
        start_joint_pos,
        lower_limits,
        upper_limits,
    )
    follower_action = build_follower_action(
        target,
        active_joint_names=active_follower_joint_names,
    )
    print(f"[INFO] Returning to saved start pose for {steps} step(s).")
    for _ in range(steps):
        follower.send_action(follower_action)
        sleep_fn(1.0 / max(int(fps), 1))
    return steps


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    args.device, _ = normalize_device_for_runtime(
        requested_device=args.device, wants_video=False)

    controller_config = load_yaml(args.controller_config)
    controller_config.setdefault("device", args.device)
    controller_config.setdefault("seed", args.seed)
    if args.checkpoint_path is not None:
        controller_config["checkpoint_path"] = args.checkpoint_path
    trajectory_config = load_yaml(args.trajectory_config)

    controller = create_controller(args.controller, env=None, config=controller_config)
    trajectory = CSVJointTrajectory(**trajectory_config)
    leader = HardwareTrajectoryLeader(trajectory)
    checkpoint_path = getattr(controller, "resolved_checkpoint_path", None) or controller_config.get("checkpoint_path")
    output_dir = resolve_deploy_output_dir(
        controller_name=args.controller,
        requested_output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
    )
    os.makedirs(output_dir, exist_ok=True)

    lower_limits, upper_limits = get_joint_limit_vectors()
    config_payload = build_deploy_config(
        args=args,
        controller_name=args.controller,
        controller_config=controller_config,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )
    config_payload["trajectory_config"] = trajectory_config

    missing_follower_joint_names = {"gripper"} if args.disable_follower_gripper else set()
    active_follower_joint_names = [
        joint_name for joint_name in TELEOP_JOINT_NAMES
        if joint_name not in missing_follower_joint_names
    ]
    observation_builder = LiveTeleopObservationBuilder(
        missing_follower_joint_names=missing_follower_joint_names,
    )
    metrics = DeployMetricAccumulator()
    follower = None
    failed = False
    interrupted = False
    failure_message: str | None = None
    steps_completed = 0
    start_joint_pos: list[float] | None = None

    try:
        follower, precise_sleep = _create_follower(args)
        print(f"[INFO] Controller: {args.controller}")
        print(f"[INFO] Trajectory: {trajectory_config.get('csv_path')}")
        print(f"[INFO] Trajectory steps: {trajectory.total_steps}")
        print(f"[INFO] Output dir: {output_dir}")
        follower.connect()
        start_joint_pos = hardware_obs_to_joint_positions(
            follower.get_observation(),
            allowed_missing_joint_names=missing_follower_joint_names,
            fallback_joint_positions_rad=trajectory.start_target,
        )
        try:
            steps_completed = run_deploy_session(
                args=args,
                leader=leader,
                follower=follower,
                controller=controller,
                observation_builder=observation_builder,
                metrics=metrics,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                sleep_fn=precise_sleep,
                active_follower_joint_names=active_follower_joint_names,
            )
            if trajectory.completed:
                _return_to_start(
                    follower=follower,
                    trajectory=trajectory,
                    start_joint_pos=start_joint_pos,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                    sleep_fn=precise_sleep,
                    fps=int(args.fps),
                    active_follower_joint_names=active_follower_joint_names,
                )
        except KeyboardInterrupt:
            interrupted = True
            steps_completed = int(metrics.summary()["num_steps"])
            print("\n[INFO] Deployment interrupted by user.")
        except (ConnectionError, OSError, TimeoutError, RuntimeError) as exc:
            failed = True
            steps_completed = int(metrics.summary()["num_steps"])
            failure_message = str(exc)
            print(f"[ERROR] Hardware communication failed: {failure_message}")
    except (ConnectionError, OSError, TimeoutError, RuntimeError) as exc:
        failed = True
        failure_message = str(exc)
        print(f"[ERROR] Hardware setup failed: {failure_message}")
    finally:
        _safe_disconnect(follower, "Follower")

    config_payload["status"] = "failed" if failed else (
        "interrupted" if interrupted else "completed")
    if failure_message is not None:
        config_payload["failure_message"] = failure_message
    artifact_paths = write_deploy_artifacts(
        output_dir=output_dir,
        config_payload=config_payload,
        metrics=metrics,
    )
    print(metrics.format_final_report())
    print(f"[INFO] Steps completed: {steps_completed}")
    print(f"[INFO] Config: {artifact_paths['config']}")
    print(f"[INFO] Summary: {artifact_paths['summary']}")
    print(f"[INFO] Timeseries: {artifact_paths['timeseries']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
