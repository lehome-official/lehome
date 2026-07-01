from __future__ import annotations
import os
import torch
import warp as wp
from collections.abc import Sequence
from typing import Any

from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from lehome.assets.object.Garment import GarmentObject
from lehome.assets.object.fluid import FluidObject
from lehome.devices.action_process import preprocess_device_action
from omegaconf import OmegaConf

from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from .loft_wipe_cfg import LoftWipeEnvCfg


def _to_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


class LoftWipeEnv(BaseEnv):
    """Washroom wipe task built on top of the shared LeHome base environment."""

    cfg: BaseEnvCfg | LoftWipeEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | LoftWipeEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self._obs_initialized = False

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to load the shared LeHome scene and lighting first.
        super()._setup_scene()
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.towel = GarmentObject(
            prim_path="/World/Objects/Towel",
            usd_path=os.getcwd() + "/Assets/objects/Thin-Shells/Towel/towel.usd",
            visual_usd_path=os.getcwd() + "/Assets/Material/Garment/linen_Blue.usd",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/washroom/config_file/particle_towel_cfg.yaml"
            ),
        )
        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/objects/Fluids/water/water.usdc",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/washroom/config_file/fluid.yaml"
            ),
            use_container=False,
        )

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        robot_joint_pos = _to_torch(self.robot.data.joint_pos)
        joint_pos = torch.cat(
            [robot_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)

        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        depth_mm = self._depth_to_uint16_mm(top_camera_depth)
        wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.wrist_rgb": wrist_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": depth_mm,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        wrist_joint_pos = _to_torch(self.robot.data.default_joint_pos)[env_ids]
        self.robot.write_joint_position_to_sim(
            wrist_joint_pos, joint_ids=None, env_ids=env_ids
        )
        if not self._obs_initialized:
            self.initialize_obs()
            self._obs_initialized = True
        self.towel.reset()
        self.object.reset(soft=True)

    def _get_success(self) -> torch.Tensor:
        success = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return success

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()
        self.towel.initialize()

    def get_all_pose(self):
        return {
            "Towel": self.towel.get_pose_data(),  # GarmentObject
            "Water": self.object.get_pose_data(),  # FluidObject
        }

    def set_all_pose(self, pose_dict, env_ids: Sequence[int] | None = None):
        if "Towel" in pose_dict:
            self.towel.set_pose_from_data(pose_dict["Towel"])
        if "Water" in pose_dict:
            self.object.set_pose_from_data(pose_dict["Water"])
