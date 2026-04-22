"""Teleoperate the internal PickOrange task with an SO101 leader arm."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib
from pathlib import Path
import signal
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so101_hackathon.utils.eval_utils import add_app_launcher_args, load_yaml
from so101_hackathon.envs.env_runtime import dynamic_reset_gripper_effort_limit_sim
from so101_hackathon.sim.robots.so101_follower_spec import SO101_FOLLOWER_MOTOR_LIMITS
from so101_hackathon.deploy.runtime import (
    DEFAULT_DELAY_STEPS,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
    DEFAULT_NOISE_STD,
    FixedDisturbanceChannel,
    build_follower_action,
    hardware_obs_to_joint_positions,
)
from so101_hackathon.deploy.ultrazohm import UltraZohmDisturbanceChannel
from so101_hackathon.registry import create_controller, list_controller_names
from so101_hackathon.utils.rl_utils import (
    TELEOP_JOINT_NAMES,
    TELEOP_RESIDUAL_ACTION_SCALE,
    clamp_action,
    finite_difference_velocity,
)

_FRONT_VIEWPORT_WINDOW = "Front"
_TOP_VIEWPORT_WINDOW = "Top"
_WRIST_VIEWPORT_WINDOW = "Wrist"
DEFAULT_CALIBRATION_DIR = Path.home() / ".cache" / "huggingface" / \
    "lerobot" / "calibration"


def _discover_vendor_module_root() -> Path:
    """Discover vendor module root."""
    external_root = REPO_ROOT / "external"
    for candidate in external_root.glob("*/source/*/*"):
        if (candidate / "devices" / "lerobot").is_dir():
            return candidate.parent
    raise ModuleNotFoundError(
        "Could not locate the vendored device package root under external/.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
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
    parser.add_argument(
        "--controller",
        choices=list_controller_names(),
        default="raw",
        help="Registered controller to run between the leader target and PickOrange action.",
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
    parser.add_argument(
        "--controller-coeff",
        type=float,
        default=1.0,
        help="Blend between direct leader teleop and controller output.",
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
        help="Gaussian joint-space noise standard deviation in radians applied to joints 1-4 only.",
    )
    parser.add_argument(
        "--disturbance-channel",
        choices=["fixed", "ultrazohm"],
        default="fixed",
        help="Disturbance channel used after the controller command.",
    )
    parser.add_argument(
        "--uzohm-can-iface",
        default="can0",
        help="SocketCAN interface used when --disturbance-channel=ultrazohm.",
    )
    parser.add_argument(
        "--uzohm-timeout-s",
        type=float,
        default=1.0,
        help="UltraZohm manipulated-output timeout in seconds.",
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Delete the cached leader calibration file first.")
    if not has_app_launcher_args:
        parser.add_argument("--device", type=str, default="cpu",
                            help="Torch/Isaac device string.")
        parser.add_argument("--enable_cameras", action="store_true",
                            default=False, help="Enable camera rendering.")
    return parser


def validate_disturbance_args(args: argparse.Namespace) -> None:
    """Validate disturbance options."""
    if args.disturbance_channel == "ultrazohm" and int(args.num_envs) != 1:
        raise ValueError("UltraZohm disturbance supports --num_envs 1 only.")


def _remove_cached_leader_calibration(leader_id: str) -> None:
    """Remove cached leader calibration."""
    calibration_path = DEFAULT_CALIBRATION_DIR / f"{leader_id}.json"
    if calibration_path.exists():
        calibration_path.unlink()


def build_pick_orange_env_cfg(args: argparse.Namespace):
    """Build pick orange env cfg."""
    from so101_hackathon.envs.pick_orange_env import build_pick_orange_env_cfg as _build_pick_orange_env_cfg

    return _build_pick_orange_env_cfg(
        teleop_device=args.teleop_device,
        seed=args.seed,
        num_envs=args.num_envs,
        device=args.device,
    )


def build_controller_config(args: argparse.Namespace, *, seed: int | None = None) -> dict[str, Any]:
    """Build controller config."""
    controller_config = load_yaml(args.controller_config)
    controller_config.setdefault("device", args.device)
    controller_config.setdefault(
        "seed", int(seed if seed is not None else (
            args.seed if args.seed is not None else 42))
    )
    if args.checkpoint_path is not None:
        controller_config["checkpoint_path"] = args.checkpoint_path
    return controller_config


@dataclass
class RateLimiter:
    hz: int

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        self.last_time = time.time()
        self.sleep_duration = 1.0 / max(int(self.hz), 1)
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        """Sleep until the next control tick."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


class KeyboardTeleopState:
    def __init__(self) -> None:
        """Initialize the object."""
        import carb
        import omni

        self._carb = carb
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._appwindow = omni.appwindow.get_default_app_window()
        self.started = False
        self._reset_requested = False
        self._success_requested = False
        if self._appwindow is None:
            self.started = True
            print(
                "[WARN] Isaac app window is unavailable; starting teleoperation immediately "
                "without keyboard reset/success hotkeys."
            )
            return
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event)

    def close(self) -> None:
        """Close owned resources."""
        if getattr(self, "_keyboard_sub", None) is not None and self._input is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """Handle on keyboard event."""
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
        """Run pop reset requested."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def pop_success_requested(self) -> bool:
        """Run pop success requested."""
        requested = self._success_requested
        self._success_requested = False
        return requested


class SO101LeaderTeleop:
    def __init__(self, env, *, port: str, recalibrate: bool) -> None:
        """Initialize the object."""
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
            from so101_hackathon.deploy.hardware import load_leader_follower_hardware_dependencies

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
        """Handle request reset."""
        self._reset_requested = True
        self._success_requested = False

    def _request_success(self) -> None:
        """Handle request success."""
        self._reset_requested = True
        self._success_requested = True

    def close(self) -> None:
        """Close owned resources."""
        if self._keyboard is not None:
            self._keyboard.close()
        if hasattr(self._leader, "disconnect"):
            self._leader.disconnect()

    def display_controls(self) -> None:
        """Display controls."""
        if self._leader_api == "vendor" and hasattr(self._leader, "display_controls"):
            self._leader.display_controls()
            return
        print("Teleop controls:")
        print("  B: start teleoperation")
        print("  R: reset task")
        print("  N: mark success and reset task")
        print("  Ctrl+C: quit")

    def reset(self) -> None:
        """Reset internal state."""
        self._reset_requested = False
        self._success_requested = False
        if hasattr(self._leader, "reset"):
            self._leader.reset()

    def pop_reset_requested(self) -> bool:
        """Run pop reset requested."""
        if self._leader_api == "vendor":
            requested = self._reset_requested
            self._reset_requested = False
            return requested
        return self._keyboard.pop_reset_requested()

    def pop_success_requested(self) -> bool:
        """Run pop success requested."""
        if self._leader_api == "vendor":
            requested = self._success_requested
            self._success_requested = False
            return requested
        return self._keyboard.pop_success_requested()

    def advance(self):
        """Advance the teleoperation device."""
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


def _is_tensor(values: Any) -> bool:
    """Return whether tensor."""
    return hasattr(values, "shape") and hasattr(values, "device") and hasattr(values, "dtype")


def _clone_action(values: Any) -> Any:
    """Handle clone action."""
    if _is_tensor(values):
        return values.clone()
    if isinstance(values, list):
        return [float(value) for value in values]
    return [float(value) for value in list(values)]


def _zeros_like_action(values: Any) -> Any:
    """Handle zeros like action."""
    if _is_tensor(values):
        return values.new_zeros(values.shape)
    return [0.0 for _ in values]


def _as_action_like(values: Any, reference: Any) -> Any:
    """Handle as action like."""
    if _is_tensor(reference):
        if _is_tensor(values):
            action = values.to(device=reference.device, dtype=reference.dtype)
        else:
            action = reference.new_tensor(values)
        if action.ndim == 1 and reference.ndim == 2:
            action = action.unsqueeze(0).expand_as(reference)
        return action
    if _is_tensor(values):
        values = values.detach().cpu().tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError(
                f"Expected a single action vector, received batch shape {len(values)}")
        values = values[0]
    return [float(value) for value in values]


def adapt_controller_action(
    *,
    leader_action: Any,
    controller_action: Any,
    controller: Any,
    controller_coeff: float,
) -> Any:
    """Run adapt controller action."""
    coeff = float(controller_coeff)
    if coeff < 0.0 or coeff > 1.0:
        raise ValueError(
            f"controller_coeff must be within [0, 1], received {coeff}")

    if getattr(controller, "action_mode", "absolute") == "residual":
        residual_action = _as_action_like(
            clamp_action(controller_action, limit=1.0), leader_action)
        if not _is_tensor(leader_action):
            return [
                float(leader_value) + coeff * TELEOP_RESIDUAL_ACTION_SCALE * float(residual_value)
                for leader_value, residual_value in zip(leader_action, residual_action)
            ]
        return leader_action + coeff * TELEOP_RESIDUAL_ACTION_SCALE * residual_action

    absolute_action = _as_action_like(controller_action, leader_action)
    if not _is_tensor(leader_action):
        return [
            float(leader_value) + coeff *
            (float(controller_value) - float(leader_value))
            for leader_value, controller_value in zip(leader_action, absolute_action)
        ]
    return leader_action + coeff * (absolute_action - leader_action)


def read_follower_joint_positions(env):
    """Read follower joint positions."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(
        list(TELEOP_JOINT_NAMES), preserve_order=True)
    return robot.data.joint_pos[:, joint_ids]


def clamp_sim_joint_positions(actions, env):
    """Clamp sim joint positions."""
    try:
        robot = env.scene["robot"]
        joint_ids, _ = robot.find_joints(
            list(TELEOP_JOINT_NAMES), preserve_order=True)
        lower_limits = robot.data.soft_joint_pos_limits[:, joint_ids, 0]
        upper_limits = robot.data.soft_joint_pos_limits[:, joint_ids, 1]
    except Exception:
        return actions
    return actions.clamp(min=lower_limits, max=upper_limits)


class SimTeleopObservationBuilder:
    def __init__(self) -> None:
        """Initialize the object."""
        self._previous_leader_joint_pos = None
        self._previous_joint_error = None
        self._previous_action = None

    def reset(self) -> None:
        """Reset internal state."""
        self._previous_leader_joint_pos = None
        self._previous_joint_error = None
        self._previous_action = None

    def set_previous_action(self, action) -> None:
        """Set previous action."""
        self._previous_action = _clone_action(action)

    def build(self, *, leader_joint_pos, follower_joint_pos, dt: float):
        """Build the observation."""
        dt = max(float(dt), 1.0e-6)
        if _is_tensor(leader_joint_pos):
            joint_error = leader_joint_pos - follower_joint_pos
        else:
            follower_joint_pos = _as_action_like(
                follower_joint_pos, leader_joint_pos)
            joint_error = [
                float(leader_value) - float(follower_value)
                for leader_value, follower_value in zip(leader_joint_pos, follower_joint_pos)
            ]
        if self._previous_leader_joint_pos is None:
            leader_joint_vel = _zeros_like_action(leader_joint_pos)
            joint_error_vel = _zeros_like_action(joint_error)
        elif _is_tensor(leader_joint_pos):
            previous_leader = _as_action_like(
                self._previous_leader_joint_pos, leader_joint_pos)
            previous_error = _as_action_like(
                self._previous_joint_error, joint_error)
            leader_joint_vel = finite_difference_velocity(
                leader_joint_pos, previous_leader, dt)
            joint_error_vel = finite_difference_velocity(
                joint_error, previous_error, dt)
        else:
            previous_leader = _as_action_like(
                self._previous_leader_joint_pos, leader_joint_pos)
            previous_error = _as_action_like(
                self._previous_joint_error, joint_error)
            leader_joint_vel = finite_difference_velocity(
                leader_joint_pos, previous_leader, dt)
            joint_error_vel = finite_difference_velocity(
                joint_error, previous_error, dt)

        previous_action = (
            _zeros_like_action(leader_joint_pos)
            if self._previous_action is None
            else _as_action_like(self._previous_action, leader_joint_pos)
        )

        self._previous_leader_joint_pos = _clone_action(leader_joint_pos)
        self._previous_joint_error = _clone_action(joint_error)

        if _is_tensor(leader_joint_pos):
            import torch

            return torch.cat(
                [
                    leader_joint_pos,
                    leader_joint_vel,
                    joint_error,
                    joint_error_vel,
                    previous_action,
                ],
                dim=-1,
            )
        return (
            list(leader_joint_pos)
            + list(leader_joint_vel)
            + list(joint_error)
            + list(joint_error_vel)
            + list(previous_action)
        )


def _single_action_values(actions) -> list[float]:
    """Return one six-joint action from a vector or single-env batch."""
    if _is_tensor(actions):
        if actions.ndim == 1:
            return actions.detach().cpu().tolist()
        if actions.ndim == 2 and actions.shape[0] == 1:
            return actions[0].detach().cpu().tolist()
        raise ValueError(
            f"Expected one action vector, received tensor shape {tuple(actions.shape)}")
    if isinstance(actions, list) and actions and isinstance(actions[0], list):
        if len(actions) != 1:
            raise ValueError(
                f"Expected one action vector, received batch length {len(actions)}")
        return [float(value) for value in actions[0]]
    return [float(value) for value in actions]


def _with_single_action_values(actions, command: list[float]):
    """Return actions with every slot replaced by one six-joint command."""
    if _is_tensor(actions):
        disturbed_actions = actions.clone()
        command_tensor = actions.new_tensor(command)
        if actions.ndim == 1:
            disturbed_actions[:] = command_tensor
        else:
            disturbed_actions[:, :] = command_tensor
        return disturbed_actions
    if isinstance(actions, list) and actions and isinstance(actions[0], list):
        return [list(command) for _ in actions]
    return list(command)


def apply_action_disturbance(actions, channel: FixedDisturbanceChannel):
    """Apply action disturbance."""
    disturbed_command = channel.apply(_single_action_values(actions))
    return _with_single_action_values(actions, disturbed_command)


def apply_ultrazohm_action_disturbance(actions, channel: UltraZohmDisturbanceChannel):
    """Apply UltraZohm action disturbance through LeRobot action values."""
    action_dict = build_follower_action(_single_action_values(actions))
    manipulated_action = channel.apply(action_dict)
    disturbed_command = hardware_obs_to_joint_positions(manipulated_action)
    return _with_single_action_values(actions, disturbed_command)


class ViewportLayoutManager:
    def __init__(self) -> None:
        """Initialize the object."""
        self._windows = []

    def _pump_ui(self, env, num_frames: int = 4) -> None:
        """Handle pump ui."""
        for _ in range(num_frames):
            env.render()
            env.sim.render()
            time.sleep(0.01)

    def _bind_camera(self, window_name: str, camera_prim_path: str) -> None:
        """Handle bind camera."""
        from omni.kit.viewport.utility import get_viewport_from_window_name

        viewport_api = get_viewport_from_window_name(window_name)
        if viewport_api is not None:
            viewport_api.camera_path = camera_prim_path

    def configure(self, env) -> None:
        """Run configure."""
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
        """Close owned resources."""
        for window in self._windows:
            try:
                window.destroy()
            except Exception:
                pass
        self._windows = []


def manual_terminate(env, success: bool):
    """Run manual terminate."""
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
    """Launch app."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(vars(args))
    return app_launcher.app


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    validate_disturbance_args(args)

    simulation_app = _launch_app(args)
    from so101_hackathon.envs.pick_orange_env import PickOrangeEnvBuilder

    env_cfg = build_pick_orange_env_cfg(args)
    env = PickOrangeEnvBuilder().make_direct_env(
        env_cfg=env_cfg,
        render_mode="rgb_array" if getattr(
            args, "enable_cameras", False) else None,
    ).unwrapped
    controller_config = build_controller_config(args, seed=env_cfg.seed)
    controller = create_controller(
        args.controller, env=None, config=controller_config)
    observation_builder = SimTeleopObservationBuilder()
    original_success_cfg = None
    if hasattr(env, "termination_manager") and "success" in getattr(env.termination_manager, "active_terms", []):
        original_success_cfg = copy.deepcopy(
            env.termination_manager.get_term_cfg("success"))
    teleop = SO101LeaderTeleop(
        env, port=args.port, recalibrate=bool(args.recalibrate))
    if args.disturbance_channel == "ultrazohm":
        disturbance_channel = UltraZohmDisturbanceChannel(
            can_iface=args.uzohm_can_iface,
            timeout_s=float(args.uzohm_timeout_s),
        )
        disturbance_channel.connect()
    else:
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
    controller.reset()
    observation_builder.reset()
    disturbance_channel.reset()
    if not getattr(args, "headless", False):
        viewport_layout = ViewportLayoutManager()
        viewport_layout.configure(env)

    interrupted = False
    last_uzohm_warning_s = 0.0

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
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
                controller.reset()
                observation_builder.reset()
                disturbance_channel.reset()
                if original_success_cfg is not None:
                    env.termination_manager.set_term_cfg(
                        "success", original_success_cfg)
                continue
            if teleop.pop_reset_requested():
                manual_terminate(env, False)
                env.reset()
                controller.reset()
                observation_builder.reset()
                disturbance_channel.reset()
                if original_success_cfg is not None:
                    env.termination_manager.set_term_cfg(
                        "success", original_success_cfg)
                continue

            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

            leader_actions = teleop.advance()
            if leader_actions is None:
                env.render()
            else:
                follower_joint_pos = read_follower_joint_positions(env)
                controller_obs = observation_builder.build(
                    leader_joint_pos=leader_actions,
                    follower_joint_pos=follower_joint_pos,
                    dt=1.0 / max(int(args.step_hz), 1),
                )
                controller_actions = controller.act(controller_obs)
                controller_decided_actions = adapt_controller_action(
                    leader_action=leader_actions,
                    controller_action=controller_actions,
                    controller=controller,
                    controller_coeff=float(args.controller_coeff),
                )
                if args.disturbance_channel == "ultrazohm":
                    raw_actions = clamp_sim_joint_positions(
                        controller_decided_actions, env)
                    try:
                        actions = apply_ultrazohm_action_disturbance(
                            raw_actions, disturbance_channel)
                    except Exception as exc:
                        now = time.time()
                        if (now - last_uzohm_warning_s) >= 2.0:
                            print(f"[WARN] UltraZohm disturbance failed; using raw command this step: {exc}")
                            last_uzohm_warning_s = now
                        actions = raw_actions
                    actions = clamp_sim_joint_positions(actions, env)
                else:
                    actions = clamp_sim_joint_positions(
                        apply_action_disturbance(controller_decided_actions, disturbance_channel), env)
                env.step(actions)
                observation_builder.set_previous_action(controller_decided_actions)
            rate_limiter.sleep(env)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        if viewport_layout is not None:
            viewport_layout.close()
        close_method = getattr(disturbance_channel, "close", None)
        if callable(close_method):
            close_method()
        teleop.close()
        env.close()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
