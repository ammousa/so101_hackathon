"""Small controller registry.

This is the only file students need to touch when they add a new controller.
"""

from __future__ import annotations

import argparse
import inspect
from typing import Any

from so101_hackathon.controllers.raw import RawController
from so101_hackathon.controllers.rl_ppo import PPOController
from so101_hackathon.controllers.rule_based_pd import TeleopPDController

CONTROLLERS = {
    "ppo": PPOController,
    "pd": TeleopPDController,
    "raw": RawController,
}


def list_controller_names() -> list[str]:
    """Return the sorted controller names shown in the CLI."""

    return sorted(CONTROLLERS.keys())


def create_controller(name: str, *, env: Any, config: dict[str, Any] | None = None) -> Any:
    """Instantiate one registered controller."""

    config = dict(config or {})
    controller_cls = CONTROLLERS.get(name)
    if controller_cls is None:
        raise KeyError(f"Unknown controller '{name}'. Available controllers: {', '.join(list_controller_names())}")

    # Only PPO needs the environment at construction time so it can load the
    # policy through the real RSL-RL inference API.
    if name == "ppo":
        config["env"] = env

    signature = inspect.signature(controller_cls)
    accepted_kwargs = {
        key: value
        for key, value in config.items()
        if key in signature.parameters
    }
    return controller_cls(**accepted_kwargs)


def cli_main(argv: list[str] | None = None) -> int:
    """Print the available controllers."""

    parser = argparse.ArgumentParser(description="List available hackathon controllers.")
    parser.parse_args(argv)
    for controller_name in list_controller_names():
        print(controller_name)
    return 0
