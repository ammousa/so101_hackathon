"""Deploy any registered SO101 teleop controller on real hardware."""

from __future__ import annotations
from so101_hackathon.utils.rl_utils import TELEOP_JOINT_NAMES
from so101_hackathon.utils.eval_utils import load_yaml
from so101_hackathon.rl_training.runtime_utils import normalize_device_for_runtime
from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.deploy.session import run_deploy_session
from so101_hackathon.deploy.runtime import (
    DEFAULT_DELAY_STEPS,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_FPS,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
    DEFAULT_NOISE_STD,
    DEFAULT_PRINT_EVERY,
    DEFAULT_TELEOP_TIME_S,
    LiveTeleopObservationBuilder,
    build_deploy_config,
    get_joint_limit_vectors,
    resolve_deploy_output_dir,
    write_deploy_artifacts,
)
from so101_hackathon.deploy.metrics import DeployMetricAccumulator
from so101_hackathon.deploy.hardware import (
    create_leader_follower_pair,
    load_leader_follower_hardware_dependencies,
)

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a registered SO101 hackathon controller on real hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controller",
        choices=list_controller_names(),
        default="pd",
        help="Registered controller to deploy.",
    )
    parser.add_argument(
        "--controller-config",
        type=str,
        default=None,
        help="Optional YAML file with controller-specific overrides.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint path forwarded into controller config.",
    )
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument(
        "--disable-follower-gripper",
        action="store_true",
        help="Run deploy without follower joint 6/gripper when that motor is absent or offline.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Target teleoperation control frequency.",
    )
    parser.add_argument(
        "--teleop-time-s",
        type=float,
        default=DEFAULT_TELEOP_TIME_S,
        help="Optional deployment duration limit in seconds.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
        help="Print one live status line every N control steps.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=DEFAULT_DELAY_STEPS,
        help="Fixed post-controller command delay in control steps.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="Gaussian command noise standard deviation in radians.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Runtime device used by learned controllers, for example `cuda:0` or `cpu`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Controller seed forwarded into controller config when supported.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit directory for deploy artifacts.",
    )
    parser.add_argument(
        "--controller-coeff",
        "--rl_coeff",
        dest="controller_coeff",
        type=float,
        default=1.0,
        help="Blend between direct leader teleop and controller output.",
    )
    return parser


def _print_run_header(
    *,
    args: argparse.Namespace,
    checkpoint_path: str | None,
    output_dir: str,
    lower_limits: list[float],
    upper_limits: list[float],
) -> None:
    print(f"[INFO] Controller: {args.controller}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Controller coefficient: {args.controller_coeff}")
    print(
        f"[INFO] Disturbance: delay_steps={args.delay_steps}, noise_std={args.noise_std}")
    if checkpoint_path:
        print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Joint lower limits (rad): {lower_limits}")
    print(f"[INFO] Joint upper limits (rad): {upper_limits}")


def _safe_disconnect(device, label: str) -> None:
    if device is None:
        return
    try:
        device.disconnect()
    except Exception as exc:
        print(f"[WARN] {label} disconnect failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device, _ = normalize_device_for_runtime(
        requested_device=args.device, wants_video=False)
    controller_config = load_yaml(args.controller_config)
    controller_config.setdefault("device", args.device)
    controller_config.setdefault("seed", args.seed)
    if args.checkpoint_path is not None:
        controller_config["checkpoint_path"] = args.checkpoint_path

    controller = create_controller(
        args.controller, env=None, config=controller_config)
    checkpoint_path = getattr(controller, "resolved_checkpoint_path",
                              None) or controller_config.get("checkpoint_path")
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

    missing_follower_joint_names = {
        "gripper"} if args.disable_follower_gripper else set()
    active_follower_joint_names = [
        joint_name for joint_name in TELEOP_JOINT_NAMES if joint_name not in missing_follower_joint_names
    ]
    observation_builder = LiveTeleopObservationBuilder(
        missing_follower_joint_names=missing_follower_joint_names,
    )
    metrics = DeployMetricAccumulator()
    interrupted = False
    failed = False
    failure_message: str | None = None
    steps_completed = 0
    leader = None
    follower = None
    try:
        SOLeader, SOLeaderConfig, SOFollower, SOFollowerConfig, precise_sleep = (
            load_leader_follower_hardware_dependencies()
        )
        leader, follower = create_leader_follower_pair(
            follower_port=args.follower_port,
            follower_id=args.follower_id,
            leader_port=args.leader_port,
            leader_id=args.leader_id,
            disable_follower_gripper=args.disable_follower_gripper,
            SOLeader=SOLeader,
            SOLeaderConfig=SOLeaderConfig,
            SOFollower=SOFollower,
            SOFollowerConfig=SOFollowerConfig,
        )
        _print_run_header(
            args=args,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        leader.connect()
        follower.connect()
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
        _safe_disconnect(leader, "Leader")
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
    if interrupted:
        print("[INFO] Run ended early due to interrupt.")
    if failed and failure_message is not None:
        print(f"[INFO] Failure: {failure_message}")
    print(f"[INFO] Config: {artifact_paths['config']}")
    print(f"[INFO] Summary: {artifact_paths['summary']}")
    print(f"[INFO] Timeseries: {artifact_paths['timeseries']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
