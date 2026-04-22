"""Calibrate one SO101 leader or follower."""

from __future__ import annotations
from so101_hackathon.deploy.hardware import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
    calibrate_so101_arm,
)

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calibrate one SO101 leader or follower arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--role", choices=["leader", "follower"], required=True)
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port for the target arm.")
    parser.add_argument("--id", type=str, default=None,
                        help="Calibration file id.")
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=str(DEFAULT_CALIBRATION_DIR),
        help="Directory where calibration JSON files are stored.",
    )
    parser.add_argument(
        "--disable-gripper",
        action="store_true",
        help="Exclude the gripper motor when calibrating a follower arm.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.role == "leader":
        port = args.port or DEFAULT_LEADER_PORT
        device_id = args.id or DEFAULT_LEADER_ID
    else:
        port = args.port or DEFAULT_FOLLOWER_PORT
        device_id = args.id or DEFAULT_FOLLOWER_ID

    saved_path = calibrate_so101_arm(
        role=args.role,
        port=port,
        device_id=device_id,
        calibration_dir=args.calibration_dir,
        disable_gripper=bool(args.disable_gripper),
    )
    print(f"[INFO] Calibration complete: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
