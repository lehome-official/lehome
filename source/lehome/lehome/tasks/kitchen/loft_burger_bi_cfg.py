from __future__ import annotations


import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
from ..base.base_env_cfg import BaseEnvCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
import os


@configclass
class LoftBurgerEnvCfg(BaseEnvCfg):
    variant_name: str = "default"
    """Environment configuration inheriting from the base LeHome scene config."""

    light_intensity: float = 50000.0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg
        )

    # robot
    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-3.49, 6.82, 0.78), 
            rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "shoulder_pan": -1.2363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ),
    )
    right_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-3.49, 7.12, 0.78),  
            rot=(0.707, 0.0, 0.0, -0.707),
            joint_pos={
                "shoulder_pan": 1.2363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": 0.085,
                "gripper": -0.01176,
            },
        ),
    )
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.205, -0.4, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    burger_beef: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_beef",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_Beef_Patties001/Burger_Beef_Patties001_Def.usd"
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-3.76, 6.8, 0.828),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    burger_board: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_ChoppingBlock/Burger_ChoppingBlock.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.75, 6.7, 0.81),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    burger_plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_Plate/Burger_Plate.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.71, 7.1, 0.81),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    burger_bread2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_bread2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_Bread002/Burger_Bread002.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.71, 7.1, 0.823),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    burger_cheese: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_cheese",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Volumetric_Objects/burger/Assets/Burger_Cheese001/Burger_Cheese001_Def.usd"
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-3.71, 7.1, 0.833),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    scene_deactivation_enabled: bool = True
    scene_deactivation_root_path: str = "/World/Scene"
    scene_keep_prim_path_prefixes: tuple[str, ...] = (
        "/World/Scene/Ceiling",
        "/World/Scene/Wall",
        "/World/Scene/Kitchen/Kitchen_Cabinet002",
        "/World/Scene/Kitchen/Stovetop017",
        "/World/Scene/Kitchen/SeasoningBox001",
        "/World/Scene/Kitchen/Jar046",
        "/World/Scene/Kitchen/Shovel008",
        "/World/Scene/Kitchen/Shovel007",
        "/World/Scene/Kitchen/WoodenSpoon013",
    )
    scene_deactivation_log_limit: int = 8

    viewer: ViewerCfg = ViewerCfg(eye=(-3.15, 6.97, 1.6), lookat=(-3.65, 6.97, 1.0))
