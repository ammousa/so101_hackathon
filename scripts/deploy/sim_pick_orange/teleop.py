"""Teleoperate the internal PickOrange task with an SO101 leader arm."""

from __future__ import annotations
from so101_hackathon.utils.eval_utils import add_app_launcher_args
from so101_hackathon.envs.env_runtime import dynamic_reset_gripper_effort_limit_sim
from so101_hackathon.sim.robots.so101_follower_spec import SO101_FOLLOWER_MOTOR_LIMITS
from so101_hackathon.deploy.runtime import (
    DEFAULT_DELAY_STEPS,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
    DEFAULT_NOISE_STD,
    FixedDisturbanceChannel,
)
from so101_hackathon.deploy.hardware import DEFAULT_CALIBRATION_DIR, load_leader_follower_hardware_dependencies

import argparse
import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
import signal
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_FRONT_VIEWPORT_WINDOW = "Front"
_TOP_VIEWPORT_WINDOW = "Top"
_WRIST_VIEWPORT_WINDOW = "Wrist"


def _discover_vendor_module_root() -> Path:
    external_root = REPO_ROOT / "external"
    for candidate in external_root.glob("*/source/*/*"):
        if (candidate / "devices" / "lerobot").is_dir():
            return candidate.parent
    raise ModuleNotFoundError(
        "Could not locate the vendored device package root under external/.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teleoperate the PickOrange kitchen environment with an SO101 leader arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    has_app_launcher_args = add_app_launcher_args(parser)
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="so101leader",
        choices=["so101leader"],
        help="Teleoperation device to use.",
    )
    parser.add_argument("--port", type=str, default=DEFAULT_LEADER_PORT,
                        help="Serial port for the SO101 leader.")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the environment.")
    parser.add_argument("--step_hz", type=int, default=60,
                        help="Environment stepping rate.")
    parser.add_argument(
        "--delay-steps",
        dest="delay_steps",
        type=int,
        default=DEFAULT_DELAY_STEPS,
        help="Fixed number of sim control steps to delay the follower command.",
    )
    parser.add_argument(
        "--noise-std",
        dest="noise_std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="Gaussian joint-space noise standard deviation in radians applied to the follower command.",
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Delete the cached leader calibration file first.")
    if not has_app_launcher_args:
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch/Isaac device string.")
        parser.add_argument("--enable_cameras", action="store_true",
                            default=False, help="Enable camera rendering.")
    return parser


def _remove_cached_leader_calibration(leader_id: str) -> None:
    calibration_path = DEFAULT_CALIBRATION_DIR / f"{leader_id}.json"
    if calibration_path.exists():
        calibration_path.unlink()


def build_pick_orange_env_cfg(args: argparse.Namespace):
    from so101_hackathon.envs.pick_orange_env import build_pick_orange_env_cfg as _build_pick_orange_env_cfg

    return _build_pick_orange_env_cfg(
        teleop_device=args.teleop_device,
        seed=args.seed,
        num_envs=args.num_envs,
        device=args.device,
    )


@dataclass
class RateLimiter:
    hz: int

    def __post_init__(self) -> None:
        self.last_time = time.time()
        self.sleep_duration = 1.0 / max(int(self.hz), 1)
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


class KeyboardTeleopState:
    def __init__(self) -> None:
        import carb
        import omni

        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event)
        self.started = False
        self._reset_requested = False
        self._success_requested = False

    def close(self) -> None:
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self.started = True
                self._reset_requested = False
                self._success_requested = False
            elif event.input.name == "R":
                self.started = False
                self._reset_requested = True
                self._success_requested = False
            elif event.input.name == "N":
                self.started = False
                self._reset_requested = True
                self._success_requested = True
        return True

    def pop_reset_requested(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def pop_success_requested(self) -> bool:
        requested = self._success_requested
        self._success_requested = False
        return requested


class SO101LeaderTeleop:
    def __init__(self, env, *, port: str, recalibrate: bool) -> None:
        leader_id = DEFAULT_LEADER_ID
        self._env = env
        self._leader = None
        self._keyboard = None
        self._reset_requested = False
        self._success_requested = False
        if recalibrate:
            _remove_cached_leader_calibration(leader_id)

        self._leader_api = "lerobot"
        try:
            SOLeader, SOLeaderConfig, _, _, _ = load_leader_follower_hardware_dependencies()
        except ModuleNotFoundError as exc:
            if exc.name != "lerobot":
                raise
            vendor_module_root = _discover_vendor_module_root()
            if str(vendor_module_root) not in sys.path:
                sys.path.insert(0, str(vendor_module_root))
            package_name = "".join(["lei", "saac"])
            SO101Leader = getattr(importlib.import_module(
                f"{package_name}.devices"), "SO101Leader")

            self._leader_api = "vendor"
            self._leader = SO101Leader(env, port=port, recalibrate=recalibrate)
            self._leader.add_callback("R", self._request_reset)
            self._leader.add_callback("N", self._request_success)
        else:
            leader_cfg = SOLeaderConfig(port=port)
            leader_cfg.id = leader_id
            leader_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR
            self._leader = SOLeader(leader_cfg)
            self._leader.connect()
            self._keyboard = KeyboardTeleopState()

    def _request_reset(self) -> None:
        self._reset_requested = True
        self._success_requested = False

    def _request_success(self) -> None:
        self._reset_requested = True
        self._success_requested = True

    def close(self) -> None:
        if self._keyboard is not None:
            self._keyboard.close()
        if hasattr(self._leader, "disconnect"):
            self._leader.disconnect()

    def display_controls(self) -> None:
        if self._leader_api == "vendor" and hasattr(self._leader, "display_controls"):
            self._leader.display_controls()
            return
        print("Teleop controls:")
        print("  B: start teleoperation")
        print("  R: reset task")
        print("  N: mark success and reset task")
        print("  Ctrl+C: quit")

    def reset(self) -> None:
        self._reset_requested = False
        self._success_requested = False
        if hasattr(self._leader, "reset"):
            self._leader.reset()

    def pop_reset_requested(self) -> bool:
        if self._leader_api == "vendor":
            requested = self._reset_requested
            self._reset_requested = False
            return requested
        return self._keyboard.pop_reset_requested()

    def pop_success_requested(self) -> bool:
        if self._leader_api == "vendor":
            requested = self._success_requested
            self._success_requested = False
            return requested
        return self._keyboard.pop_success_requested()

    def advance(self):
        if self._leader_api == "vendor":
            if not getattr(self._leader, "started", False):
                return None
            joint_state = self._leader.get_device_state()
            action = {
                "so101_leader": True,
                "joint_state": joint_state,
                "motor_limits": SO101_FOLLOWER_MOTOR_LIMITS,
            }
            return self._env.cfg.preprocess_device_action(action)
        if not self._keyboard.started:
            return None
        joint_state = self._leader.get_action()
        action = {
            "so101_leader": True,
            "joint_state": joint_state,
            "motor_limits": SO101_FOLLOWER_MOTOR_LIMITS,
        }
        return self._env.cfg.preprocess_device_action(action)


def apply_action_disturbance(actions, channel: FixedDisturbanceChannel):
    disturbed_command = channel.apply(actions[0].detach().cpu().tolist())
    disturbed_actions = actions.clone()
    disturbed_actions[:, :] = actions.new_tensor(disturbed_command)
    return disturbed_actions


class ViewportLayoutManager:
    def __init__(self) -> None:
        self._windows = []

    def _pump_ui(self, env, num_frames: int = 4) -> None:
        for _ in range(num_frames):
            env.render()
            env.sim.render()
            time.sleep(0.01)

    def _bind_camera(self, window_name: str, camera_prim_path: str) -> None:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        viewport_api = get_viewport_from_window_name(window_name)
        if viewport_api is not None:
            viewport_api.camera_path = camera_prim_path

    def configure(self, env) -> None:
        import omni.ui as ui
        from omni.kit.viewport.window import ViewportWindow

        if env.num_envs < 1:
            return

        default_window = ui.Workspace.get_window("Viewport")
        if default_window is None:
            return

        env_root = env.scene.env_prim_paths[0]
        env.sim.set_camera_view(eye=env.cfg.viewer.eye,
                                target=env.cfg.viewer.lookat)

        front_window = ViewportWindow(_FRONT_VIEWPORT_WINDOW)
        top_window = ViewportWindow(_TOP_VIEWPORT_WINDOW)
        wrist_window = ViewportWindow(_WRIST_VIEWPORT_WINDOW)
        self._windows = [front_window, top_window, wrist_window]

        for window in self._windows:
            window.dock_tab_bar_visible = False
            window.deferred_dock_in("Viewport")

        self._pump_ui(env, 6)
        front_window.dock_in(default_window, ui.DockPosition.RIGHT, 0.5)
        self._pump_ui(env, 6)
        top_window.dock_in(default_window, ui.DockPosition.BOTTOM, 0.5)
        self._pump_ui(env, 6)
        wrist_window.dock_in(front_window, ui.DockPosition.BOTTOM, 0.5)
        self._pump_ui(env, 8)

        self._bind_camera(_FRONT_VIEWPORT_WINDOW,
                          f"{env_root}/Robot/base/front_camera")
        self._bind_camera(_TOP_VIEWPORT_WINDOW, f"{env_root}/Scene/top_camera")
        self._bind_camera(_WRIST_VIEWPORT_WINDOW,
                          f"{env_root}/Robot/gripper/wrist_camera")
        self._pump_ui(env, 4)

    def close(self) -> None:
        for window in self._windows:
            try:
                window.destroy()
            except Exception:
                pass
        self._windows = []


def manual_terminate(env, success: bool):
    import torch
    from isaaclab.managers import TerminationTermCfg

    if hasattr(env, "termination_manager"):
        env.termination_manager.set_term_cfg(
            "success",
            TerminationTermCfg(
                func=lambda _env: torch.full(
                    (_env.num_envs,),
                    bool(success),
                    dtype=torch.bool,
                    device=_env.device,
                )
            ),
        )
        env.termination_manager.compute()


def _launch_app(args: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(vars(args))
    return app_launcher.app


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    simulation_app = _launch_app(args)
    from so101_hackathon.envs.pick_orange_env import PickOrangeEnvBuilder

    env_cfg = build_pick_orange_env_cfg(args)
    env = PickOrangeEnvBuilder().make_direct_env(
        env_cfg=env_cfg,
        render_mode="rgb_array" if getattr(args, "enable_cameras", False) else None,
    ).unwrapped
    original_success_cfg = None
    if hasattr(env, "termination_manager") and "success" in getattr(env.termination_manager, "active_terms", []):
        original_success_cfg = copy.deepcopy(
            env.termination_manager.get_term_cfg("success"))
    teleop = SO101LeaderTeleop(
        env, port=args.port, recalibrate=bool(args.recalibrate))
    disturbance_channel = FixedDisturbanceChannel(
        delay_steps=int(args.delay_steps),
        noise_std=float(args.noise_std),
        seed=int(env_cfg.seed),
    )
    viewport_layout = None
    teleop.display_controls()
    rate_limiter = RateLimiter(args.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop.reset()
    disturbance_channel.reset()
    if not getattr(args, "headless", False):
        viewport_layout = ViewportLayoutManager()
        viewport_layout.configure(env)

    interrupted = False

    def signal_handler(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    try:
        while simulation_app.is_running() and not interrupted:
            if teleop.pop_success_requested():
                print("Task Success!!!")
                manual_terminate(env, True)
                env.reset()
                disturbance_channel.reset()
                if original_success_cfg is not None:
                    env.termination_manager.set_term_cfg(
                        "success", original_success_cfg)
                continue
            if teleop.pop_reset_requested():
                manual_terminate(env, False)
                env.reset()
                disturbance_channel.reset()
                if original_success_cfg is not None:
                    env.termination_manager.set_term_cfg(
                        "success", original_success_cfg)
                continue

            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

            actions = teleop.advance()
            if actions is None:
                env.render()
            else:
                env.step(apply_action_disturbance(
                    actions, disturbance_channel))
            rate_limiter.sleep(env)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if viewport_layout is not None:
            viewport_layout.close()
        teleop.close()
        env.close()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
