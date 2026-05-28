from __future__ import annotations
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from lehome.assets.robots.xlerobot import XLEROBOT_CFG

@configclass
class XlerobotEnvCfg(DirectRLEnvCfg):
    # 环境配置
    decimation = 1
    episode_length_s = 1000
    action_scale = 3.0  # 动作缩放
    action_space = 15  # 旋转关节
    observation_space = 34  # 关节位置和速度
    state_space = 0
    robot_initial_position = (2.75, -1.5, 0.0)
    robot_initial_orientation = (0.0, 0.0, 0.0, 1.0)
    render_cfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="DLAA", dlss_mode=2
    )
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, render=render_cfg)

    # 机器人配置
    robot: ArticulationCfg = XLEROBOT_CFG.replace(
        prim_path="/World/Robot",
        init_state=XLEROBOT_CFG.init_state.replace(pos=robot_initial_position),
    )
    # 摄像头配置 
    front_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/xlerobot/base_link/front_camera",
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
