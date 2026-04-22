"""Lightweight environment builder helpers safe to import before Isaac starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TeleopEnvLaunch:
    simulation_app: Any
    env: Any


class BaseHackathonEnvBuilder:
    """Base builder that centralizes Isaac stack checks and app launch."""

    def require_isaac_stack(self) -> None:
        """Run require isaac stack."""
        missing = []
        for module_name in ("isaaclab", "gymnasium", "isaaclab_rl"):
            try:
                __import__(module_name)
            except ModuleNotFoundError:
                missing.append(module_name)
        if missing:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "The teleop environment requires the Isaac Lab runtime stack. "
                f"Missing modules: {', '.join(missing)}"
            )

    def build_env_cfg(self, **kwargs: Any) -> Any:
        """Build env cfg."""
        raise NotImplementedError

    def make_env(
        self,
        *,
        env_id: str,
        env_cfg: Any,
        enable_cameras: bool = False,
        record_video: bool = False,
        video_dir: str | None = None,
        video_length: int = 600,
        wrap_for_rl: bool = False,
    ) -> Any:
        """Create env."""
        self.require_isaac_stack()

        import gymnasium as gym

        env = gym.make(
            env_id,
            cfg=env_cfg,
            render_mode="rgb_array" if enable_cameras or record_video else None,
        )
        if record_video:
            from so101_hackathon.rl_training.runtime_utils import validate_rgb_rendering

            if video_dir is None:
                raise ValueError(
                    "video_dir must be provided when record_video=True")
            render_ok, render_reason = validate_rgb_rendering(env)
            if not render_ok:
                print(
                    f"[WARN] Video probe reported invalid frames: {render_reason}.")
                print(
                    "[WARN] Continuing video recording anyway because --video was explicitly requested.")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )
        if wrap_for_rl:
            from so101_hackathon.rl_training.rsl_rl_wrapper import RslRlVecEnvWrapper

            env = RslRlVecEnvWrapper(env)
        return env

    def launch_and_make_env(
        self,
        *,
        env_id: str,
        app_launcher_args: Any | None = None,
        headless: bool = False,
        enable_cameras: bool = False,
        device: str = "cpu",
        record_video: bool = False,
        video_dir: str | None = None,
        video_length: int = 600,
        wrap_for_rl: bool = False,
        **build_kwargs: Any,
    ) -> TeleopEnvLaunch:
        """Launch and make env."""
        self.require_isaac_stack()

        import argparse

        from isaaclab.app import AppLauncher

        if app_launcher_args is None:
            parser = argparse.ArgumentParser(add_help=False)
            AppLauncher.add_app_launcher_args(parser)
            launch_args = []
            if headless:
                launch_args.append("--headless")
            if enable_cameras:
                launch_args.append("--enable_cameras")
            if device:
                launch_args.extend(["--device", device])
            parser_args = parser.parse_args(launch_args)
        else:
            parser_args = app_launcher_args

        app_launcher = AppLauncher(parser_args)
        simulation_app = app_launcher.app
        env_cfg = self.build_env_cfg(device=device, **build_kwargs)
        env = self.make_env(
            env_id=env_id,
            env_cfg=env_cfg,
            enable_cameras=enable_cameras,
            record_video=record_video,
            video_dir=video_dir,
            video_length=video_length,
            wrap_for_rl=wrap_for_rl,
        )
        return TeleopEnvLaunch(simulation_app=simulation_app, env=env)
