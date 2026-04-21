"""Real-hardware adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import time

from so101_hackathon.deploy.runtime import (
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_PORT,
)

DEFAULT_CALIBRATION_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_vendor_module_root() -> Path:
    external_root = _REPO_ROOT / "external"
    for candidate in external_root.glob("*/source/*/*"):
        if (candidate / "devices" / "lerobot").is_dir():
            return candidate.parent
    raise ModuleNotFoundError("Could not locate the vendored device package root under external/.")


_VENDOR_MODULE_ROOT = _discover_vendor_module_root()
_VENDOR_DEVICE_CACHE_DIR = _VENDOR_MODULE_ROOT / "".join(["lei", "saac"]) / "devices" / "lerobot" / ".cache"


@dataclass
class _RepoSOLeaderConfig:
    port: str
    id: str = DEFAULT_LEADER_ID
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR


@dataclass
class _RepoSOFollowerConfig:
    port: str
    id: str = DEFAULT_FOLLOWER_ID
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR
    disable_gripper: bool = False


def _ensure_repo_lerobot_path() -> None:
    if str(_VENDOR_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_VENDOR_MODULE_ROOT))


def _load_repo_motor_dependencies():
    _ensure_repo_lerobot_path()
    package_name = "".join(["lei", "saac"])
    errors_module = importlib.import_module(f"{package_name}.devices.lerobot.common.errors")
    motors_module = importlib.import_module(f"{package_name}.devices.lerobot.common.motors")

    return (
        getattr(errors_module, "DeviceAlreadyConnectedError"),
        getattr(errors_module, "DeviceNotConnectedError"),
        getattr(motors_module, "FeetechMotorsBus"),
        getattr(motors_module, "Motor"),
        getattr(motors_module, "MotorCalibration"),
        getattr(motors_module, "MotorNormMode"),
        getattr(motors_module, "OperatingMode"),
    )


def _default_motor_map(Motor, MotorNormMode):
    return {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def _maybe_without_gripper(values: dict[str, object], *, disable_gripper: bool) -> dict[str, object]:
    if not disable_gripper:
        return dict(values)
    filtered = dict(values)
    filtered.pop("gripper", None)
    return filtered


def _with_joint_field_suffix(values: dict[str, float]) -> dict[str, float]:
    return {f"{joint_name}.pos": float(value) for joint_name, value in values.items()}


def _resolve_calibration_path(calibration_dir: Path, device_id: str, role: str) -> Path:
    calibration_dir = Path(calibration_dir)
    candidate_names = [f"{device_id}.json"]
    if role == "leader":
        candidate_names.extend(["leader.json", "so101_leader.json"])
    elif role == "follower":
        candidate_names.extend(["follower.json", "so101_follower.json"])

    for candidate_name in candidate_names:
        candidate_path = calibration_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    if role == "leader":
        repo_cache_candidate = _VENDOR_DEVICE_CACHE_DIR / "so101_leader.json"
        if repo_cache_candidate.exists():
            return repo_cache_candidate
    elif role == "follower":
        repo_cache_candidate = _VENDOR_DEVICE_CACHE_DIR / "so101_follower.json"
        if repo_cache_candidate.exists():
            return repo_cache_candidate

    available_files = sorted(path.name for path in calibration_dir.glob("*.json")) if calibration_dir.exists() else []
    raise FileNotFoundError(
        f"Could not find calibration for {role} `{device_id}` in `{calibration_dir}`. "
        f"Tried: {', '.join(candidate_names)}. "
        f"Available calibration files: {available_files or 'none'}"
    )


def _load_motor_calibration(calibration_dir: Path, device_id: str, role: str, MotorCalibration):
    calibration_path = _resolve_calibration_path(calibration_dir, device_id, role)
    with calibration_path.open("r", encoding="utf-8") as handle:
        json_data = json.load(handle)
    return {
        motor_name: MotorCalibration(
            id=int(motor_data["id"]),
            drive_mode=int(motor_data["drive_mode"]),
            homing_offset=int(motor_data["homing_offset"]),
            range_min=int(motor_data["range_min"]),
            range_max=int(motor_data["range_max"]),
        )
        for motor_name, motor_data in json_data.items()
    }


def _save_motor_calibration(calibration_path: Path, calibration: dict[str, object]) -> None:
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        motor_name: {
            "id": motor_cfg.id,
            "drive_mode": motor_cfg.drive_mode,
            "homing_offset": motor_cfg.homing_offset,
            "range_min": motor_cfg.range_min,
            "range_max": motor_cfg.range_max,
        }
        for motor_name, motor_cfg in calibration.items()
    }
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


class _RepoSOLeader:
    def __init__(self, cfg: _RepoSOLeaderConfig):
        (
            self._device_already_connected_error,
            self._device_not_connected_error,
            FeetechMotorsBus,
            Motor,
            MotorCalibration,
            MotorNormMode,
            OperatingMode,
        ) = _load_repo_motor_dependencies()
        self._OperatingMode = OperatingMode
        self.cfg = cfg
        self.port = cfg.port
        self.calibration_dir = Path(cfg.calibration_dir)
        self.calibration = _load_motor_calibration(self.calibration_dir, cfg.id, "leader", MotorCalibration)
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=_default_motor_map(Motor, MotorNormMode),
            calibration=self.calibration,
        )

    def connect(self):
        if self.bus.is_connected:
            raise self._device_already_connected_error("SO101-Leader is already connected.")
        self.bus.connect()
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, self._OperatingMode.POSITION.value)

    def disconnect(self):
        if not self.bus.is_connected:
            raise self._device_not_connected_error("SO101-Leader is not connected.")
        self.bus.disconnect()

    def get_action(self):
        return _with_joint_field_suffix(self.bus.sync_read("Present_Position"))


class _RepoSOFollower:
    def __init__(self, cfg: _RepoSOFollowerConfig):
        (
            self._device_already_connected_error,
            self._device_not_connected_error,
            FeetechMotorsBus,
            Motor,
            MotorCalibration,
            MotorNormMode,
            OperatingMode,
        ) = _load_repo_motor_dependencies()
        self._OperatingMode = OperatingMode
        self.cfg = cfg
        self.disable_gripper = bool(getattr(cfg, "disable_gripper", False))
        self.port = cfg.port
        self.calibration_dir = Path(cfg.calibration_dir)
        self.calibration = _maybe_without_gripper(
            _load_motor_calibration(self.calibration_dir, cfg.id, "follower", MotorCalibration),
            disable_gripper=self.disable_gripper,
        )
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=_maybe_without_gripper(
                _default_motor_map(Motor, MotorNormMode),
                disable_gripper=self.disable_gripper,
            ),
            calibration=self.calibration,
        )

    def connect(self):
        if self.bus.is_connected:
            raise self._device_already_connected_error("SO101-Follower is already connected.")
        self.bus.connect()
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, self._OperatingMode.POSITION.value)
        self.bus.enable_torque()

    def disconnect(self):
        if not self.bus.is_connected:
            raise self._device_not_connected_error("SO101-Follower is not connected.")
        self.bus.disconnect()

    def get_observation(self):
        return _with_joint_field_suffix(self.bus.sync_read("Present_Position"))

    def send_action(self, action):
        for field_name, value in action.items():
            joint_name = field_name[:-4] if field_name.endswith(".pos") else field_name
            if joint_name not in self.bus.motors:
                continue
            self.bus.write("Goal_Position", joint_name, float(value))
        return dict(action)


def load_leader_follower_hardware_dependencies() -> tuple[object, object, object, object, object]:
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig
        from lerobot.utils.robot_utils import precise_sleep
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot":
            raise
        return _RepoSOLeader, _RepoSOLeaderConfig, _RepoSOFollower, _RepoSOFollowerConfig, time.sleep

    return SOLeader, SOLeaderConfig, SOFollower, SOFollowerConfig, precise_sleep


def calibrate_so101_arm(
    *,
    role: str,
    port: str,
    device_id: str,
    calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
    disable_gripper: bool = False,
) -> Path:
    """Run interactive calibration for one SO101 arm and save the result."""

    if role not in {"leader", "follower"}:
        raise ValueError(f"Unsupported calibration role `{role}`")

    (
        _device_already_connected_error,
        _device_not_connected_error,
        FeetechMotorsBus,
        Motor,
        MotorCalibration,
        MotorNormMode,
        OperatingMode,
    ) = _load_repo_motor_dependencies()
    del _device_already_connected_error, _device_not_connected_error, OperatingMode

    motors = _maybe_without_gripper(
        _default_motor_map(Motor, MotorNormMode),
        disable_gripper=disable_gripper,
    )
    motor_names = list(motors.keys())
    bus = FeetechMotorsBus(port=port, motors=motors)
    calibration_dir = Path(calibration_dir)
    calibration_path = calibration_dir / f"{device_id}.json"

    print(f"\nRunning calibration for {role} `{device_id}` on {port}")
    if disable_gripper:
        print("[INFO] Gripper motor excluded from calibration.")

    bus.connect()
    try:
        bus.disable_torque()

        input("Move the arm to the middle of its range of motion and press ENTER...")
        homing_offsets = bus.set_half_turn_homings()

        print("Move all joints through their full range of motion.")
        print("Press ENTER when done...")
        range_mins, range_maxes = bus.record_ranges_of_motion()

        calibration = {
            name: MotorCalibration(
                id=motors[name].id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
            for name in motor_names
        }
        _save_motor_calibration(calibration_path, calibration)
        print(f"Calibration saved to {calibration_path}")
    finally:
        bus.disconnect(disable_torque=False)

    return calibration_path


def create_leader_follower_pair(
    *,
    follower_port: str = DEFAULT_FOLLOWER_PORT,
    follower_id: str = DEFAULT_FOLLOWER_ID,
    leader_port: str = DEFAULT_LEADER_PORT,
    leader_id: str = DEFAULT_LEADER_ID,
    disable_follower_gripper: bool = False,
    SOLeader,
    SOLeaderConfig,
    SOFollower,
    SOFollowerConfig,
):
    leader_cfg = SOLeaderConfig(port=leader_port)
    leader_cfg.id = leader_id
    leader_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR

    follower_cfg = SOFollowerConfig(port=follower_port)
    follower_cfg.id = follower_id
    follower_cfg.calibration_dir = DEFAULT_CALIBRATION_DIR
    setattr(follower_cfg, "disable_gripper", bool(disable_follower_gripper))

    leader = SOLeader(leader_cfg)
    follower = SOFollower(follower_cfg)
    if disable_follower_gripper and not isinstance(follower_cfg, _RepoSOFollowerConfig):
        follower.bus.motors.pop("gripper", None)
        follower.bus.calibration.pop("gripper", None)
        if hasattr(follower, "calibration") and isinstance(follower.calibration, dict):
            follower.calibration.pop("gripper", None)

    return leader, follower
