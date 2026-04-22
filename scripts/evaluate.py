"""Unified evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so101_hackathon.utils.eval_utils import (
    add_app_launcher_args,
    build_evaluation_payload,
    evaluate_controller,
    load_yaml,
    log_evaluation_metrics,
    resolve_evaluation_output_dir,
    write_evaluation_config,
    write_summary_json,
)
from so101_hackathon.rl_training.runtime_utils import (
    apply_video_renderer_fallback,
    normalize_device_for_runtime,
)
from so101_hackathon.registry import create_controller, list_controller_names


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate any registered SO101 hackathon controller.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    has_app_launcher_args = add_app_launcher_args(parser)
    parser.add_argument(
        "--controller",
        choices=list_controller_names(),
        default="pd",
        help="Controller to run during evaluation.",
    )
    parser.add_argument(
        "--controller-config",
        type=str,
        default=None,
        help="Optional YAML file with controller-specific overrides.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=1.0,
        help="Override the PD proportional gain for evaluation. Defaults to 1.0 for a pure proportional controller.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.0,
        help="Override the PD derivative gain for evaluation. Defaults to 0 for a pure proportional controller.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help="Optional YAML file with teleop environment overrides.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to aggregate.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Sleep between steps to roughly match the simulated control rate.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record an MP4 of the first evaluation episode.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=600,
        help="Maximum recorded video length in environment steps.",
    )
    parser.add_argument(
        "--hide-leader-ghost",
        action="store_true",
        default=False,
        help="Disable the leader visualization robot used during teleop evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit directory for evaluation artifacts. If unset, the repo chooses a task-specific location.",
    )
    if not has_app_launcher_args:
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Torch/Isaac device string, for example `cuda:0` or `cpu`.",
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for environment construction and controller setup.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=50,
        help="Number of parallel Isaac environments to launch for evaluation.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=0,
        help="Override the fixed action delay used by the teleop disturbance model.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0,
        help="Override the fixed action noise standard deviation used by the teleop disturbance model for joints 1-4 only.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Full path to the RL checkpoint to load during evaluation.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-form note saved into the evaluation config JSON.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional experiment grouping label saved into the evaluation config JSON.",
    )
    return parser


def _coerce_override_value(raw_value: str) -> Any:
    """Coerce override value."""
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if any(token in raw_value for token in (".", "e", "E")):
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _apply_controller_overrides(
    controller_config: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Apply controller overrides."""
    hydra_key_map = {
        "kp": "kp",
        "kd": "kd",
        "max_action": "max_action",
        "controller.kp": "kp",
        "controller.kd": "kd",
        "controller.max_action": "max_action",
        "controller_config.kp": "kp",
        "controller_config.kd": "kd",
        "controller_config.max_action": "max_action",
    }
    remaining: list[str] = []

    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep:
            remaining.append(override)
            continue
        mapped_key = hydra_key_map.get(key)
        if mapped_key is None:
            remaining.append(override)
            continue
        controller_config[mapped_key] = _coerce_override_value(raw_value)

    if remaining:
        raise SystemExit(
            "Unrecognized arguments: "
            + " ".join(remaining)
            + ". Supported Hydra-style controller overrides are "
            "`kp=...`, `kd=...`, `max_action=...`, `controller.kp=...`, "
            "`controller.kd=...`, `controller.max_action=...`."
        )
    return controller_config


def _run_cli(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    args.device, args.video = normalize_device_for_runtime(
        requested_device=args.device, wants_video=args.video)
    args.enable_cameras = bool(args.video)
    apply_video_renderer_fallback(args)
    env_config = load_yaml(args.env_config)
    controller_config = load_yaml(args.controller_config)
    controller_config = _apply_controller_overrides(
        controller_config, overrides)
    controller_config.setdefault("device", args.device)
    controller_config.setdefault("logger", "tensorboard")
    controller_config.setdefault("seed", args.seed)
    controller_config.setdefault("note", args.note or "")
    controller_config.setdefault("group", args.group or "")
    if args.kp is not None:
        controller_config["kp"] = args.kp
    if args.kd is not None:
        controller_config["kd"] = args.kd
    if args.checkpoint_path is not None:
        controller_config["checkpoint_path"] = args.checkpoint_path

    from so101_hackathon.envs.teleop_env import launch_and_make_teleop_env

    output_dir = resolve_evaluation_output_dir(
        controller_name=args.controller,
        requested_output_dir=args.output_dir,
        checkpoint_path=controller_config.get("checkpoint_path"),
    )
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    config_path = write_evaluation_config(
        output_dir=output_dir,
        args=args,
        env_config=env_config,
        controller_config=controller_config,
    )
    env_options = {
        "headless": args.headless,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "delay_steps": args.delay_steps,
        "noise_std": args.noise_std,
        "enable_cameras": args.video,
        "wrap_for_rl": args.controller == "ppo",
        "record_video": args.video,
        "video_dir": video_dir,
        "video_length": args.video_length,
        "show_leader_ghost": not args.hide_leader_ghost,
        "eval_time_out_only": True,
        "app_launcher_args": args,
    }
    env_options.update(env_config)
    launch = launch_and_make_teleop_env(**env_options)
    simulation_app = launch.simulation_app
    env = launch.env
    env_closed = False
    try:
        controller = create_controller(
            args.controller, env=env, config=controller_config)
        result = evaluate_controller(
            env=env,
            controller=controller,
            num_episodes=args.num_episodes,
            real_time=args.real_time,
            show_progress=True,
            simulation_app=simulation_app,
        )
        try:
            env.close()
            env_closed = True
        except Exception as close_exc:
            print(
                f"[WARN] Environment close failed during evaluation teardown: {close_exc}")

        payload = build_evaluation_payload(
            controller_name=args.controller,
            output_dir=output_dir,
            config_path=config_path,
            result=result,
            video_dir=video_dir,
            include_video=args.video,
        )
        write_summary_json(output_dir, payload)
        log_evaluation_metrics(
            controller_name=args.controller,
            output_dir=output_dir,
            result=result,
            controller_config=controller_config,
        )

        print(json.dumps(payload, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if not env_closed:
            try:
                env.close()
            except Exception as close_exc:
                print(
                    f"[WARN] Environment close failed during evaluation teardown: {close_exc}")
        try:
            simulation_app.close()
        except Exception as close_exc:
            print(
                f"[WARN] Simulation app close failed during evaluation teardown: {close_exc}")


if __name__ == "__main__":
    _run_cli()
