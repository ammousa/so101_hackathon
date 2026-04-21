"""Train the single PPO baseline for the hackathon repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so101_hackathon.utils.rl_utils import build_training_log_dir
from so101_hackathon.rl_training.runtime_utils import normalize_device_for_runtime
from so101_hackathon.rl_training.ppo_config import build_teleop_ppo_runner_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SO101 teleop PPO baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Launch Isaac Sim without the interactive viewer.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the number of parallel training environments.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override the PPO training iteration budget.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training and environment initialization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch/Isaac device string, for example `cuda:0` or `cpu`.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=None,
        help="Override the fixed action delay used by the teleop disturbance model.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Override the fixed action noise standard deviation used by the teleop disturbance model for joints 1-4 only.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="so101_hackathon_teleop",
        help="Top-level log directory name under `logs/rsl_rl/`.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional suffix appended to the timestamped training run directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from a previous checkpoint selected by `--load-run` and `--checkpoint`.",
    )
    parser.add_argument(
        "--load-run",
        type=str,
        default=".*",
        help="Regex used to select the training run when `--resume` is enabled.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=".*\\.pt",
        help="Checkpoint filename or regex used when `--resume` is enabled.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-form note stored in the training config.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional experiment grouping label stored in the training config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device, _ = normalize_device_for_runtime(
        requested_device=args.device, wants_video=False)
    try:
        from so101_hackathon.rl_training.on_policy_runner import OnPolicyRunner
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Training requires `rsl_rl`, `torch`, and Isaac Lab to be installed."
        ) from exc

    from so101_hackathon.envs.teleop_env import launch_and_make_teleop_env

    cfg = build_teleop_ppo_runner_cfg(
        seed=args.seed,
        device=args.device,
        logger="tensorboard",
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        resume=args.resume,
        load_run=args.load_run,
        load_checkpoint=args.checkpoint,
        note=args.note or "",
        group=args.group or "",
    )
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations

    launch = launch_and_make_teleop_env(
        headless=args.headless,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        delay_steps=args.delay_steps,
        noise_std=args.noise_std,
        wrap_for_rl=True,
    )
    simulation_app = launch.simulation_app
    env = launch.env
    try:
        log_root = os.path.abspath(os.path.join(
            "logs", "rsl_rl", cfg.experiment_name))
        os.makedirs(log_root, exist_ok=True)
        log_dir = build_training_log_dir(log_root, cfg.run_name)
        print(f"[INFO] Training logger: {cfg.logger}")
        print(f"[INFO] Logging to: {log_dir}")
        runner = OnPolicyRunner(
            env, cfg.to_dict(), log_dir=log_dir, device=cfg.device)
        if cfg.resume:
            from so101_hackathon.utils.rl_utils import resolve_checkpoint_path

            resume_path = resolve_checkpoint_path(
                log_root, cfg.load_run, cfg.load_checkpoint)
            print(f"[INFO] Loading checkpoint: {resume_path}")
            runner.load(resume_path)
        runner.learn(num_learning_iterations=cfg.max_iterations,
                     init_at_random_ep_len=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
