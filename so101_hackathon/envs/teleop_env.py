"""Single public Isaac Lab environment for the hackathon repo."""

from __future__ import annotations

from typing import Any

from so101_hackathon.envs.base_env import BaseHackathonEnvBuilder, TeleopEnvLaunch
from so101_hackathon.sim.robots.so101_follower_spec import (
    SO101_BASE_BODY_NAME,
    SO101_CONTACT_SENSOR_BODY_NAMES,
    SO101_EE_BODY_NAME,
    SO101_JAW_BODY_NAME,
    SO101_JOINT_NAMES,
)
from so101_hackathon.utils.rl_utils import TELEOP_RESIDUAL_ACTION_SCALE


def _require_isaac_stack() -> None:
    """Handle require isaac stack."""
    BaseHackathonEnvBuilder().require_isaac_stack()


def _build_cfg_classes():
    """Build cfg classes."""
    _require_isaac_stack()

    import isaaclab.sim as sim_utils
    import so101_hackathon.sim.mdp as so101_mdp
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import ActionTermCfg as ActionTerm
    from isaaclab.managers import CurriculumTermCfg as CurrTerm
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from so101_hackathon.sim.robots import SO101_FOLLOWER_CFG

    arm_joint_names = list(SO101_JOINT_NAMES)

    @configclass
    class TeleopSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )

        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
        )
        robot: ArticulationCfg | None = None
        leader_robot: ArticulationCfg | None = None
        arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll|gripper|jaw)",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
        )
        ee_frame = FrameTransformerCfg(
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
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75), intensity=2500.0),
        )

    @configclass
    class CommandsCfg:
        leader_joints = so101_mdp.TaskSpaceLeaderCommandCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            resampling_time_range=(3.5, 4.5),
            waypoint_limit_margin=0.05,
            debug_vis=False,
        )

    @configclass
    class ActionsCfg:
        arm_action: ActionTerm = so101_mdp.ResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            command_name="leader_joints",
            scale=TELEOP_RESIDUAL_ACTION_SCALE,
            offset=0.0,
            preserve_order=True,
            max_delay=8,
            delay_range=(0, 0),
            noise_std_range=(0.0, 0.0),
        )

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            leader_joint_pos = ObsTerm(func=so101_mdp.command_joint_positions, params={
                                       "command_name": "leader_joints"})
            leader_joint_vel = ObsTerm(func=so101_mdp.command_joint_velocities, params={
                                       "command_name": "leader_joints"})
            joint_error = ObsTerm(
                func=so101_mdp.joint_tracking_error,
                params={"command_name": "leader_joints", "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=arm_joint_names)},
            )
            joint_error_vel = ObsTerm(
                func=so101_mdp.joint_velocity_error,
                params={"command_name": "leader_joints", "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=arm_joint_names)},
            )
            previous_action = ObsTerm(func=so101_mdp.last_action)

            """Optional low-dimensional policy layout to be used when needed."""

            # joint_pos = ObsTerm(func=so101_mdp.joint_pos)
            # joint_vel = ObsTerm(func=so101_mdp.joint_vel)
            # joint_pos_rel = ObsTerm(func=so101_mdp.joint_pos_rel)
            # joint_vel_rel = ObsTerm(func=so101_mdp.joint_vel_rel)
            # ee_frame_state = ObsTerm(
            #     func=so101_mdp.ee_frame_state,
            #     params={
            #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            #         "robot_cfg": SceneEntityCfg("robot"),
            #     },
            # )
            # joint_pos_target = ObsTerm(
            #     func=so101_mdp.joint_pos_target,
            #     params={"asset_cfg": SceneEntityCfg("robot")},
            # )

            def __post_init__(self):
                """Finalize dataclass initialization."""
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 1
                self.flatten_history_dim = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class EventCfg:
        reset_robot_joints = EventTerm(
            func=so101_mdp.reset_scene_to_default, mode="reset")

    @configclass
    class RewardsCfg:
        joint_position_tracking = RewTerm(
            func=so101_mdp.joint_position_tracking_l2,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=arm_joint_names), "command_name": "leader_joints"},
        )
        joint_velocity_tracking = RewTerm(
            func=so101_mdp.joint_velocity_tracking_l2,
            weight=0.2,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=arm_joint_names), "command_name": "leader_joints"},
        )
        action_rate = RewTerm(func=so101_mdp.action_rate_l2, weight=-5.0e-2)
        joint_acceleration = RewTerm(
            func=so101_mdp.joint_acceleration_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=arm_joint_names)},
        )
        action_magnitude = RewTerm(
            func=so101_mdp.action_magnitude_l2,
            weight=-5.0e-3,
        )

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=so101_mdp.time_out, time_out=True)
        # collision = DoneTerm(
        #     func=so101_mdp.illegal_contact,
        #     params={
        #         "threshold": 5.0,
        #         "sensor_cfg": SceneEntityCfg(
        #             "arm_contact",
        #             body_names=list(SO101_CONTACT_SENSOR_BODY_NAMES),
        #         ),
        #     },
        # )
        # excessive_joint_error = DoneTerm(
        #     func=so101_mdp.joint_error_too_large,
        #     params={
        #         "command_name": "leader_joints",
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joint_names),
        #         "max_abs_error": 0.75,
        #     },
        # )
        # joint_limit_violation = DoneTerm(
        #     func=so101_mdp.joint_limit_violation,
        #     params={"asset_cfg": SceneEntityCfg(
        #         "robot", joint_names=arm_joint_names), "position_tolerance": 0.02},
        # )
        # unstable_joint_velocity = DoneTerm(
        #     func=so101_mdp.unstable_joint_velocity,
        #     params={"asset_cfg": SceneEntityCfg(
        #         "robot", joint_names=arm_joint_names), "max_velocity": 2.0},
        # )

    @configclass
    class CurriculumCfg:
        disturbance = CurrTerm(
            func=so101_mdp.disturbance_curriculum,
            params={
                "action_name": "arm_action",
                "schedule": [
                    {"max_step": 5000, "stage": 1, "delay_range": (
                        0, 0), "noise_range": (0.0, 0.0)},
                    {"max_step": 15000, "stage": 2, "delay_range": (
                        0, 0), "noise_range": (0.0, 0.005)},
                    {"max_step": 30000, "stage": 3, "delay_range": (
                        2, 2), "noise_range": (0.0, 0.005)},
                    {"max_step": 50000, "stage": 4, "delay_range": (
                        0, 3), "noise_range": (0.0, 0.01)},
                    {"max_step": None, "stage": 5, "delay_range": (
                        0, 8), "noise_range": (0.0, 0.03)},
                ],
            },
        )

    @configclass
    class TeleopEnvCfg(ManagerBasedRLEnvCfg):
        scene: TeleopSceneCfg = TeleopSceneCfg(num_envs=2048, env_spacing=2.5)
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        commands: CommandsCfg = CommandsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        def __post_init__(self):
            """Finalize dataclass initialization."""
            legacy_joint_pos = {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 1.57,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            }
            self.scene.robot = SO101_FOLLOWER_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=SO101_FOLLOWER_CFG.spawn.replace(
                    activate_contact_sensors=True),
                init_state=SO101_FOLLOWER_CFG.init_state.replace(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos=legacy_joint_pos,
                    joint_vel={".*": 0.0},
                ),
            )
            self.decimation = 1
            self.sim.dt = 1.0 / 60.0
            self.sim.render_interval = self.decimation
            self.episode_length_s = 12.0
            self.viewer.eye = (2.5, 2.5, 1.5)

    return TeleopEnvCfg


def _enable_eval_leader_robot(
    env_cfg: Any,
    *,
    x_offset: float = 0.60,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    opacity: float = 1.0,
    color: tuple[float, float, float] = (0.85, 0.10, 0.10),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Attach a second visible leader robot in front of the follower for eval visuals."""

    import isaaclab.sim as sim_utils

    robot_init_state = env_cfg.scene.robot.init_state
    base_pos = tuple(getattr(robot_init_state, "pos", (0.0, 0.0, 0.0)))
    leader_pos = (
        base_pos[0] + x_offset,
        base_pos[1] + y_offset,
        base_pos[2] + z_offset,
    )
    leader_spawn = env_cfg.scene.robot.spawn.replace(
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            opacity=opacity,
            metallic=0.0,
            roughness=0.2,
        ),
        scale=scale,
    )
    env_cfg.scene.leader_robot = env_cfg.scene.robot.replace(
        prim_path="{ENV_REGEX_NS}/LeaderRobot",
        spawn=leader_spawn,
        init_state=robot_init_state.replace(pos=leader_pos),
    )


class TrainingTeleopEnvBuilder(BaseHackathonEnvBuilder):
    """Builder for the legacy training/evaluation teleop environment."""

    env_id = "so101-hackathon-teleop"

    def build_env_cfg(
        self,
        *,
        num_envs: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        delay_steps: int | None = None,
        noise_std: float | None = None,
        record_video: bool = False,
        show_leader_ghost: bool = False,
        eval_time_out_only: bool = False,
        **_: Any,
    ) -> Any:
        """Build env cfg."""
        teleop_env_cfg_cls = _build_cfg_classes()
        env_cfg = teleop_env_cfg_cls()
        env_cfg.seed = seed
        env_cfg.sim.device = device
        if record_video:
            env_cfg.viewer.resolution = (1280, 720)
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
        if show_leader_ghost:
            _enable_eval_leader_robot(env_cfg)
        if eval_time_out_only:
            env_cfg.terminations.collision = None
            env_cfg.terminations.excessive_joint_error = None
            env_cfg.terminations.joint_limit_violation = None
            env_cfg.terminations.unstable_joint_velocity = None

        arm_action = env_cfg.actions.arm_action
        if hasattr(arm_action, "fixed_delay_steps"):
            arm_action.fixed_delay_steps = delay_steps
        if hasattr(arm_action, "fixed_noise_std"):
            arm_action.fixed_noise_std = noise_std
        return env_cfg


def make_teleop_env(
    *,
    headless: bool = False,
    num_envs: int | None = None,
    seed: int = 42,
    device: str = "cpu",
    delay_steps: int | None = None,
    noise_std: float | None = None,
    enable_cameras: bool = False,
    wrap_for_rl: bool = False,
    record_video: bool = False,
    video_dir: str | None = None,
    video_length: int = 600,
    show_leader_ghost: bool = False,
    eval_time_out_only: bool = False,
) -> Any:
    """Create the single teleop environment used by all controllers."""
    builder = TrainingTeleopEnvBuilder()
    env_cfg = builder.build_env_cfg(
        num_envs=num_envs,
        seed=seed,
        device=device,
        delay_steps=delay_steps,
        noise_std=noise_std,
        record_video=record_video,
        show_leader_ghost=show_leader_ghost,
        eval_time_out_only=eval_time_out_only,
    )
    return builder.make_env(
        env_id=builder.env_id,
        env_cfg=env_cfg,
        enable_cameras=enable_cameras,
        record_video=record_video,
        video_dir=video_dir,
        video_length=video_length,
        wrap_for_rl=wrap_for_rl,
    )


def launch_and_make_teleop_env(
    *,
    headless: bool = False,
    num_envs: int | None = None,
    seed: int = 42,
    device: str = "cpu",
    delay_steps: int | None = None,
    noise_std: float | None = None,
    enable_cameras: bool = False,
    wrap_for_rl: bool = False,
    record_video: bool = False,
    video_dir: str | None = None,
    video_length: int = 600,
    show_leader_ghost: bool = False,
    eval_time_out_only: bool = False,
    app_launcher_args: Any | None = None,
) -> TeleopEnvLaunch:
    """Launch Isaac Sim and return the wrapped teleop environment.

    The function hides the Isaac Lab application setup so students can focus on
    controller code instead of simulator bootstrapping.
    """
    builder = TrainingTeleopEnvBuilder()
    builder.require_isaac_stack()

    import argparse
    import gymnasium as gym

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

    _build_cfg_classes()
    if "so101-hackathon-teleop" not in gym.registry:
        gym.register(
            id="so101-hackathon-teleop",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
        )
    env = make_teleop_env(
        enable_cameras=enable_cameras,
        num_envs=num_envs,
        seed=seed,
        device=device,
        delay_steps=delay_steps,
        noise_std=noise_std,
        wrap_for_rl=wrap_for_rl,
        record_video=record_video,
        video_dir=video_dir,
        video_length=video_length,
        show_leader_ghost=show_leader_ghost,
        eval_time_out_only=eval_time_out_only,
    )
    return TeleopEnvLaunch(
        simulation_app=simulation_app,
        env=env,
    )
