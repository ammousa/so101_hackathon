"""Run PickOrange with a CSV leader trajectory and a registered controller."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.deploy.sim_pick_orange.teleop import main as teleop_main


def main(argv: list[str] | None = None) -> int:
    """Run PickOrange trajectory playback."""
    args = list(argv if argv is not None else sys.argv[1:])
    if "--trajectory-config" not in args:
        args.extend(["--trajectory-config", "config/traj.yaml"])
    return teleop_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
