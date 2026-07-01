from __future__ import annotations
import torch

# from dataclasses import MISSING
from typing import Any, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab_physx.assets import DeformableObject
from isaaclab.sensors import TiledCamera
from .loft_burger_bi_cfg import LoftBurgerEnvCfg
from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
import numpy as np
from lehome.utils.success_checker import success_checker_burger


class LoftBurgerEnv(BaseEnv):
    """Kitchen burger task built on top of the shared LeHome base environment."""

    cfg: BaseEnvCfg | LoftBurgerEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | LoftBurgerEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment
        self.action_scale = self.cfg.action_scale
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to load the shared LeHome scene and lighting first.
        super()._setup_scene()
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        self.burger_beef = DeformableObject(self.cfg.burger_beef)
        self.burger_board = RigidObject(self.cfg.burger_board)
        self.burger_plate = RigidObject(self.cfg.burger_plate)
        self.burger_bread2 = RigidObject(self.cfg.burger_bread2)
        self.burger_cheese = DeformableObject(self.cfg.burger_cheese)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.rigid_objects["burger_plate"] = self.burger_plate
        self.scene.rigid_objects["burger_bread2"] = self.burger_bread2
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.rigid_objects["burger_board"] = self.burger_board
        self.scene.deformable_objects["burger_beef"] = self.burger_beef
        self.scene.deformable_objects["burger_cheese"] = self.burger_cheese
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        depth_mm = self._depth_to_uint16_mm(top_camera_depth)
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
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

    def _get_success(self) -> torch.Tensor:
        beef_pos = self.burger_beef.data.root_pos_w 
        plate_pos = self.burger_plate.data.root_pos_w
        success = success_checker_burger(beef_pos=beef_pos, plate_pos=plate_pos)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        # defalut position
        burger_plate_pos = self.burger_plate.data.default_root_state[env_ids].clone()
        burger_bread2_pos = self.burger_bread2.data.default_root_state[env_ids].clone()
        burger_cheese_pos = self.burger_cheese.data.default_nodal_state_w[env_ids].clone()
        burger_beef_pos = self.burger_beef.data.default_nodal_state_w[env_ids].clone()
        burger_board_pose = self.burger_board.data.default_root_state[env_ids, :7].clone()

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]

        # random position noise (x,y)
        rand_vals = torch.empty(1, 1, 2, device="cuda").uniform_(-0.1, 0.1)
        rand_vals_1 = torch.empty(len(env_ids), 2, device="cuda").uniform_(-0.05, 0.05)

        # plate (root state)
        random_plate_pos = burger_plate_pos.clone()
        random_plate_pos[..., :2] += rand_vals_1

        # bread2 (root state)
        random_bread2_pos = burger_bread2_pos.clone()
        random_bread2_pos[..., :2] += rand_vals_1

        # cheese (nodal state)
        random_cheese_pos = burger_cheese_pos.clone()
        random_cheese_pos[..., :2] += rand_vals_1.unsqueeze(1)

        # beef (nodal state)
        random_beef_pos = burger_beef_pos.clone()
        random_beef_pos[..., :2] += rand_vals

        # sim
        self.burger_plate.write_root_state_to_sim(random_plate_pos, env_ids=env_ids)
        self.burger_board.write_root_pose_to_sim(burger_board_pose, env_ids=env_ids)
        self.burger_bread2.write_root_state_to_sim(random_bread2_pos, env_ids=env_ids)
        self.burger_cheese.write_nodal_state_to_sim(random_cheese_pos, env_ids=env_ids)
        self.burger_beef.write_nodal_state_to_sim(random_beef_pos, env_ids=env_ids)

        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )

        # Save states after reset for robust recording
        self.burger_beef_reset_state = self.burger_beef.data.nodal_state_w[env_ids].clone().cpu().numpy()
        self.burger_cheese_reset_state = self.burger_cheese.data.nodal_state_w[env_ids].clone().cpu().numpy()
        self.burger_plate_reset_state = self.burger_plate.data.root_state_w[env_ids].clone().cpu().numpy()
        self.burger_bread2_reset_state = self.burger_bread2.data.root_state_w[env_ids].clone().cpu().numpy()

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.burger_beef_reset_state = self.burger_beef.data.nodal_state_w.clone().cpu().numpy()
        self.burger_cheese_reset_state = self.burger_cheese.data.nodal_state_w.clone().cpu().numpy()
        self.burger_plate_reset_state = self.burger_plate.data.root_state_w.clone().cpu().numpy()
        self.burger_bread2_reset_state = self.burger_bread2.data.root_state_w.clone().cpu().numpy()

    def get_all_pose(self):
        return {
            "BurgerBeef": {"nodal_state": self.burger_beef_reset_state.tolist()},
            "BurgerCheese": {"nodal_state": self.burger_cheese_reset_state.tolist()},
            "BurgerPlate": {
                "trans": self.burger_plate_reset_state[0, :3].tolist(),
                "rot": self.burger_plate_reset_state[0, 3:7].tolist(),
            },
            "BurgerBread": {
                "trans": self.burger_bread2_reset_state[0, :3].tolist(),
                "rot": self.burger_bread2_reset_state[0, 3:7].tolist(),
            },
        }

    def set_all_pose(self, pose_dict, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        
        # Deformable Objects
        if "BurgerBeef" in pose_dict:
            beef_tensor = torch.tensor(pose_dict["BurgerBeef"]["nodal_state"], dtype=torch.float32, device=self.device)
            self.burger_beef.write_nodal_state_to_sim(beef_tensor, env_ids=env_ids)
        if "BurgerCheese" in pose_dict:
            cheese_tensor = torch.tensor(pose_dict["BurgerCheese"]["nodal_state"], dtype=torch.float32, device=self.device)
            self.burger_cheese.write_nodal_state_to_sim(cheese_tensor, env_ids=env_ids)
            
        # Rigid Objects
        if "BurgerPlate" in pose_dict:
            plate_data = pose_dict["BurgerPlate"]
            plate_state = self.burger_plate.data.default_root_state[env_ids].clone()
            plate_state[..., :3] = torch.tensor(plate_data["trans"], dtype=torch.float32, device=self.device)
            plate_state[..., 3:7] = torch.tensor(plate_data["rot"], dtype=torch.float32, device=self.device)
            plate_state[..., 7:] = 0.0
            self.burger_plate.write_root_state_to_sim(plate_state, env_ids=env_ids)
            
        if "BurgerBread" in pose_dict:
            bread_data = pose_dict["BurgerBread"]
            bread_state = self.burger_bread2.data.default_root_state[env_ids].clone()
            bread_state[..., :3] = torch.tensor(bread_data["trans"], dtype=torch.float32, device=self.device)
            bread_state[..., 3:7] = torch.tensor(bread_data["rot"], dtype=torch.float32, device=self.device)
            bread_state[..., 7:] = 0.0
            self.burger_bread2.write_root_state_to_sim(bread_state, env_ids=env_ids)
