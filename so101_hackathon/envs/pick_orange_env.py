"""PickOrange teleop environment assembly under the shared envs package."""

from __future__ import annotations

import time
from typing import Any

import torch
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass

from so101_hackathon.envs.common import (
    BaseHackathonEnvBuilder,
    KITCHEN_WITH_ORANGE_CFG,
    KITCHEN_WITH_ORANGE_USD_PATH,
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
    is_so101_at_rest_pose,
    parse_usd_and_create_subassets,
)


DEFAULT_PICK_ORANGE_ENV_ID = "so101-hackathon-pick-orange-v0"


def orange_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Run orange grasped."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    return torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)


def put_orange_to_plate(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Run put orange to plate."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    orange: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    plate_x, plate_y = plate.data.root_pos_w[:, 0], plate.data.root_pos_w[:, 1]
    orange_x, orange_y = orange.data.root_pos_w[:, 0], orange.data.root_pos_w[:, 1]
    orange_in_plate_x = torch.logical_and(orange_x < plate_x + x_range[1], orange_x > plate_x + x_range[0])
    orange_in_plate_y = torch.logical_and(orange_y < plate_y + y_range[1], orange_y > plate_y + y_range[0])
    orange_in_plate = torch.logical_and(orange_in_plate_x, orange_in_plate_y)

    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    orange_pos = orange.data.root_pos_w
    pos_diff = torch.linalg.vector_norm(orange_pos - end_effector_pos, dim=1)
    ee_near_orange = pos_diff < diff_threshold
    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold
    placed = torch.logical_and(orange_in_plate, ee_near_orange)
    return torch.logical_and(placed, gripper_open)


def task_done(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Run task done."""
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_height = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        orange_x = orange.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        orange_y = orange.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        orange_height = orange.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        done = torch.logical_and(done, orange_x < plate_x + x_range[1])
        done = torch.logical_and(done, orange_x > plate_x + x_range[0])
        done = torch.logical_and(done, orange_y < plate_y + y_range[1])
        done = torch.logical_and(done, orange_y > plate_y + y_range[0])
        done = torch.logical_and(done, orange_height < plate_height + height_range[1])
        done = torch.logical_and(done, orange_height > plate_height + height_range[0])

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    return torch.logical_and(done, is_so101_at_rest_pose(joint_pos, joint_names))


@configclass
class PickOrangeSceneCfg(SingleArmTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)
    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()
    terminations: SingleArmTerminationsCfg = SingleArmTerminationsCfg()
    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        super().__post_init__()
        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH,
            self,
            specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"],
        )


class PickOrangeEnvBuilder(BaseHackathonEnvBuilder):
    """Builder for the teleop kitchen env used by internal task scripts."""

    env_id = DEFAULT_PICK_ORANGE_ENV_ID

    def build_env_cfg(
        self,
        *,
        teleop_device: str = "so101leader",
        seed: int | None = None,
        num_envs: int = 1,
        device: str = "cpu",
        **_: Any,
    ) -> PickOrangeEnvCfg:
        """Build env cfg."""
        env_cfg = PickOrangeEnvCfg()
        env_cfg.use_teleop_device(teleop_device)
        env_cfg.seed = seed if seed is not None else int(time.time())
        env_cfg.scene.num_envs = int(num_envs)
        env_cfg.sim.device = device
        return env_cfg

    def make_direct_env(self, *, env_cfg: PickOrangeEnvCfg, render_mode: str | None = None):
        """Create direct env."""
        self.require_isaac_stack()
        return ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)


def build_pick_orange_env_cfg(**kwargs: Any) -> PickOrangeEnvCfg:
    """Return the configured PickOrange env cfg."""

    return PickOrangeEnvBuilder().build_env_cfg(**kwargs)
