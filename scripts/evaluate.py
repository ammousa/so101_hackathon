"""Unified evaluation and play entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import traceback

from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.training.runtime_utils import (
    apply_video_renderer_fallback,
    normalize_device_for_runtime,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate any registered SO101 hackathon controller.")
    has_app_launcher_args = add_app_launcher_args(parser)
    parser.add_argument(
        "--controller", choices=list_controller_names(), default="pd")
    parser.add_argument("--controller-config", type=str, default=None,
                        help="Optional YAML override for the controller.")
    parser.add_argument("--env-config", type=str, default=None,
                        help="Optional YAML override for the teleop environment.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--real-time", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument(
        "--hide-leader-ghost",
        action="store_true",
        default=False,
        help="Disable the leader visualization robot used during teleop evaluation.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    if not has_app_launcher_args:
        parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--delay-steps", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Full path to the RL checkpoint to load during evaluation.",
    )
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    return parser


def _run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device, args.video = normalize_device_for_runtime(
        requested_device=args.device, wants_video=args.video)
    args.enable_cameras = bool(args.video)
    apply_video_renderer_fallback(args)
    env_config = load_yaml(args.env_config)
    controller_config = load_yaml(args.controller_config)
    controller_config.setdefault("device", args.device)
    controller_config.setdefault("logger", "tensorboard")
    controller_config.setdefault("seed", args.seed)
    controller_config.setdefault("note", args.note or "")
    controller_config.setdefault("group", args.group or "")
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
