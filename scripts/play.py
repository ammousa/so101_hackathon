"""Interactive single-episode play wrapper around the evaluation entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import _run_cli, build_parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = build_parser()
        parser.description = "Play a controller in the SO101 teleop environment."
        parser.print_help()
        return 0

    if "--num-episodes" not in argv:
        argv.extend(["--num-episodes", "1"])
    if "--real-time" not in argv:
        argv.append("--real-time")
    return _run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
