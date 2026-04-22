"""Shared environment helpers for beginner-friendly teleop envs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass

from so101_hackathon.paths import ASSETS_ROOT
from so101_hackathon.sim.robots import SO101_FOLLOWER_CFG
from so101_hackathon.sim.robots.so101_follower_spec import (
    SO101_ARM_JOINT_NAMES,
    SO101_BASE_BODY_NAME,
    SO101_EE_BODY_NAME,
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_REST_POSE_RANGE_DEG,
    SO101_JAW_BODY_NAME,
    SO101_JOINT_NAMES,
    convert_motor_observation_to_joint_positions,
)


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


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device: str) -> None:
    """Run dynamic reset gripper effort limit sim."""
    if teleop_device != "so101leader":
        return
    write_gripper_effort_limit_sim(env, env.scene["robot"])


def write_gripper_effort_limit_sim(env, env_arm) -> None:
    """Write gripper effort limit sim."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "Runtime gripper effort updates require `torch`.") from exc

    gripper_pos = env_arm.data.body_link_pos_w[:, -1]

    object_positions = []
    object_masses = []
    for obj in env.scene._rigid_objects.values():
        object_positions.append(obj.data.body_link_pos_w[:, 0])
        object_masses.append(obj.data.default_mass)
    if not object_positions:
        return

    object_positions = torch.stack(object_positions)
    object_masses = torch.stack(object_masses)
    distances = torch.sqrt(
        torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))
    _, min_indices = torch.min(distances, dim=0)
    target_masses = object_masses[min_indices.cpu(), 0, 0]
    target_effort_limits = (
        target_masses / 0.15).to(env_arm._data.joint_effort_limits.device)

    current_effort_limit_sim = env_arm._data.joint_effort_limits[:, -1]
    need_update = torch.abs(target_effort_limits -
                            current_effort_limit_sim) > 0.1
    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]
        env_arm.write_joint_effort_limit_to_sim(limits=new_limits, joint_ids=[
                                                5 for _ in range(env.num_envs)])


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Run ee frame state."""
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w[:,
                                                             0, :], ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)


def joint_pos_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Run joint pos target."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]


def init_single_arm_action_cfg(action_cfg, teleop_device: str):
    """Initialize single arm action cfg."""
    if teleop_device != "so101leader":
        raise ValueError(
            f"Unsupported teleop device `{teleop_device}`. Only `so101leader` is supported.")

    action_cfg.arm_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SO101_ARM_JOINT_NAMES),
        scale=1.0,
    )
    action_cfg.gripper_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0,
    )

    for field in fields(action_cfg):
        value = getattr(action_cfg, field.name, None)
        if value is None or value is MISSING:
            raise ValueError(
                f"Action configuration `{field.name}` was not set for `{teleop_device}`.")
    return action_cfg


def preprocess_single_arm_device_action(action: dict[str, Any], *, num_envs: int, device: torch.device) -> torch.Tensor:
    """Preprocess single arm device action."""
    if not action.get("so101_leader"):
        raise NotImplementedError(
            "Only SO101 leader teleop actions are supported.")
    joint_state = normalize_joint_state(action["joint_state"])
    motor_limits = action.get("motor_limits") or SO101_FOLLOWER_MOTOR_LIMITS
    joint_positions = convert_motor_observation_to_joint_positions(
        joint_state, motor_limits=motor_limits)
    processed_action = torch.zeros(
        (num_envs, len(SO101_JOINT_NAMES)), device=device, dtype=torch.float32)
    processed_action[:, :] = torch.tensor(
        joint_positions, device=device, dtype=torch.float32)
    return processed_action


def normalize_joint_state(joint_state: dict[str, Any]) -> dict[str, float]:
    """Normalize joint state."""
    normalized: dict[str, float] = {}
    for joint_name in SO101_JOINT_NAMES:
        if joint_name in joint_state:
            normalized[joint_name] = float(joint_state[joint_name])
            continue
        pos_field_name = f"{joint_name}.pos"
        if pos_field_name in joint_state:
            normalized[joint_name] = float(joint_state[pos_field_name])
            continue
        raise KeyError(
            f"Missing joint `{joint_name}` in leader state. "
            f"Available fields: {sorted(joint_state.keys())}"
        )
    return normalized


def is_so101_at_rest_pose(joint_pos: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Return whether so101 at rest pose."""
    is_reset = torch.ones(
        joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
    joint_pos_deg = joint_pos / torch.pi * 180.0
    for joint_name, (min_pos, max_pos) in SO101_FOLLOWER_REST_POSE_RANGE_DEG.items():
        joint_idx = joint_names.index(joint_name)
        is_reset = torch.logical_and(
            is_reset,
            torch.logical_and(
                joint_pos_deg[:, joint_idx] > min_pos, joint_pos_deg[:, joint_idx] < max_pos),
        )
    return is_reset


def _require_pxr():
    """Handle require pxr."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ModuleNotFoundError as exc:  # pragma: no cover - requires Isaac runtime
        raise RuntimeError(
            "USD scene parsing requires the `pxr` modules from the Isaac runtime.") from exc
    return Usd, UsdGeom, UsdPhysics


def get_stage(usd_path):
    """Return stage."""
    Usd, _, _ = _require_pxr()
    return Usd.Stage.Open(usd_path)


def get_all_prims(stage, prim=None, prims_list=None):
    """Return all prims."""
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def get_prim_pos_rot(prim):
    """Return prim pos rot."""
    Usd, UsdGeom, _ = _require_pxr()
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None, None
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if matrix.Orthonormalize(issueWarning=True):
        rot = matrix.ExtractRotationQuat()
        rot_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[
            1], rot.GetImaginary()[2]]
    else:
        rot_list = [1.0, 0.0, 0.0, 0.0]
    pos = matrix.ExtractTranslation()
    return list(pos), rot_list


def _is_articulation_root(prim):
    """Return whether articulation root."""
    _, _, UsdPhysics = _require_pxr()
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def _is_rigidbody(prim):
    """Return whether rigidbody."""
    _, _, UsdPhysics = _require_pxr()
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def _get_articulation_joints(articulation_prim):
    """Return articulation joints."""
    _, _, UsdPhysics = _require_pxr()
    joints = []

    def recurse(prim):
        """Run recurse."""
        if UsdPhysics.Joint(prim):
            joints.append(prim)
        for child in prim.GetChildren():
            recurse(child)

    recurse(articulation_prim)
    return joints


def _is_fixed_joint(prim):
    """Return whether fixed joint."""
    return prim.GetTypeName() == "PhysicsFixedJoint"


def _get_all_joints_without_fixed(articulation_prim):
    """Return all joints without fixed."""
    joints = _get_articulation_joints(articulation_prim)
    return [joint for joint in joints if not _is_fixed_joint(joint)]


def _match_specific_name(prim_path, specific_name_list, exclude_name_list):
    """Handle match specific name."""
    match_specific = True if specific_name_list is None else any(
        name in prim_path for name in specific_name_list)
    match_exclude = False if exclude_name_list is None else any(
        name in prim_path for name in exclude_name_list)
    return match_specific and not match_exclude


def parse_usd_and_create_subassets(usd_path, env_cfg, specific_name_list=None, exclude_name_list=None):
    """Parse usd and create subassets."""
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.assets.rigid_object import RigidObjectCfg
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone

    @clone
    def spawn_from_prim_path(prim_path, spawn, translation, orientation):
        """Run spawn from prim path."""
        return prim_utils.get_prim_at_path(prim_path)

    stage = get_stage(usd_path)
    prims = get_all_prims(stage)
    articulation_sub_prims = []
    created_names: dict[str, int] = {}

    for prim in prims:
        if _is_articulation_root(prim) and _match_specific_name(
            prim.GetPath().pathString, specific_name_list, exclude_name_list
        ):
            pos, rot = get_prim_pos_rot(prim)
            joints = _get_all_joints_without_fixed(prim)
            if not joints:
                continue
            original_prim_path = prim.GetPath().pathString
            name = original_prim_path.split("/")[-1]
            if name in created_names:
                created_names[name] += 1
                name = f"{name}_{created_names[name]}"
            else:
                created_names[name] = 0
            sub_prim_path = original_prim_path[original_prim_path.find(
                "/", 1) + 1:]
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
            setattr(
                env_cfg.scene,
                name,
                ArticulationCfg(
                    prim_path=prim_path,
                    spawn=None,
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=pos, rot=rot),
                    actuators={},
                ),
            )
            articulation_sub_prims.extend(get_all_prims(stage, prim))

    for prim in prims:
        if _is_rigidbody(prim) and _match_specific_name(prim.GetPath().pathString, specific_name_list, exclude_name_list):
            if prim in articulation_sub_prims:
                continue
            pos, rot = get_prim_pos_rot(prim)
            original_prim_path = prim.GetPath().pathString
            name = original_prim_path.split("/")[-1]
            if name in created_names:
                created_names[name] += 1
                name = f"{name}_{created_names[name]}"
            else:
                created_names[name] = 0
            sub_prim_path = original_prim_path[original_prim_path.find(
                "/", 1) + 1:]
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
            setattr(
                env_cfg.scene,
                name,
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=RigidObjectSpawnerCfg(func=spawn_from_prim_path),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=pos, rot=rot),
                ),
            )


KITCHEN_WITH_ORANGE_USD_PATH = str(
    (ASSETS_ROOT / "scenes" / "kitchen_with_orange" / "scene.usd").resolve())

KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_ORANGE_USD_PATH,
    )
)


@configclass
class SingleArmTaskSceneCfg(InteractiveSceneCfg):
    scene: AssetBaseCfg = MISSING

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_BASE_BODY_NAME}",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_EE_BODY_NAME}",
                name=SO101_EE_BODY_NAME,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_JAW_BODY_NAME}",
                name=SO101_JAW_BODY_NAME,
                offset=OffsetCfg(pos=(-0.021, -0.070, 0.02)),
            ),
        ],
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_EE_BODY_NAME}/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=False,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Scene/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.25, -0.45, 1.7),
            rot=(0.0, 0.7071068, 0.7071068, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=False,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_BASE_BODY_NAME}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=False,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class SingleArmActionsCfg:
    arm_action: Any = MISSING
    gripper_action: Any = MISSING


@configclass
class SingleArmEventCfg:
    reset_all: EventTerm = EventTerm(
        func=isaac_mdp.reset_scene_to_default, mode="reset")


@configclass
class SingleArmObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)
        wrist = ObsTerm(
            func=isaac_mdp.image,
            params={"sensor_cfg": SceneEntityCfg(
                "wrist"), "data_type": "rgb", "normalize": False},
        )
        front = ObsTerm(
            func=isaac_mdp.image,
            params={"sensor_cfg": SceneEntityCfg(
                "front"), "data_type": "rgb", "normalize": False},
        )
        ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg(
                "ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=joint_pos_target, params={
                                   "asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            """Finalize dataclass initialization."""
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SingleArmRewardsCfg:
    pass


@configclass
class SingleArmTerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class SingleArmTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: SingleArmTaskSceneCfg = MISSING
    observations: SingleArmObservationsCfg = MISSING
    actions: SingleArmActionsCfg = SingleArmActionsCfg()
    events: SingleArmEventCfg = SingleArmEventCfg()
    rewards: SingleArmRewardsCfg = SingleArmRewardsCfg()
    terminations: SingleArmTerminationsCfg = MISSING

    dynamic_reset_gripper_effort_limit: bool = True
    task_description: str = MISSING

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (2.0, -0.5, 1.0)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (
            0.05, 0.05, 0.05)

    def use_teleop_device(self, teleop_device: str) -> None:
        """Run use teleop device."""
        self.task_type = teleop_device
        self.actions = init_single_arm_action_cfg(self.actions, teleop_device)

    def preprocess_device_action(self, action: dict[str, Any]) -> torch.Tensor:
        """Preprocess device action."""
        return preprocess_single_arm_device_action(action, num_envs=self.scene.num_envs, device=self.sim.device)
