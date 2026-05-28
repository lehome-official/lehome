from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from lehome.assets.robots.lekiwi import LEKIWI_CFG


@configclass
class LekiwiEnvCfg(DirectRLEnvCfg):
    # 环境配置
    decimation = 1
    episode_length_s = 1000
    action_scale = 3.0  # 动作缩放
    base_action_gain = 3.0  # 底盘增量控制增益
    arm_action_gain = 3.0  # 机械臂相对控制增益
    action_space = 9  # 3个轮子 + 5个机械臂关节 + 1个夹爪
    observation_space = 18  # 9个关节位置 + 9个关节速度
    state_space = 0
    robot_initial_position = (2.75, -1.5, 0.0)
    robot_initial_orientation = (0.0, 0.0, 0.0, 1.0)
    render_cfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="DLAA", dlss_mode=2
    )
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # 机器人配置
    robot: ArticulationCfg = LEKIWI_CFG.replace(
        prim_path="/World/Robot",
        init_state=LEKIWI_CFG.init_state.replace(
            pos=robot_initial_position
        ),  # 初始位置
    )

    # 摄像头配置 (使用与xlerobot相同的配置)
    front_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base_plate_layer1_v5/front_camera",
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
            lock_camera=True,
        ),
        width=640,
        height=480,
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )
