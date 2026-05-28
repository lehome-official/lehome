import os

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
    UsdFileCfg,
)
from isaaclab.assets import Articulation
from .lekiwi_cfg import LekiwiEnvCfg

from lehome.assets.scenes.loft import LEKIWI_LOFT_USD_PATH
from lehome.assets.object.Garment import GarmentObject
from lehome.devices.lekiwi_action_process import clamp_lekiwi_joint_targets
from omegaconf import OmegaConf


class LekiwiEnv(DirectRLEnv):
    cfg: LekiwiEnvCfg

    def __init__(self, cfg: LekiwiEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.action_scale = self.cfg.action_scale
        self._base_action_gain = float(
            getattr(self.cfg, "base_action_gain", self.action_scale)
        )
        self._arm_action_gain = float(
            getattr(self.cfg, "arm_action_gain", self.action_scale)
        )
        self.joint_pos = self._robot.data.joint_pos
        self.actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )
        self._robot_initial_position = torch.tensor(
            self.cfg.robot_initial_position, device=self.device
        )
        self._robot_initial_orientation = torch.tensor(
            self.cfg.robot_initial_orientation, device=self.device
        )
        # Action mode: relative for keyboard, absolute for hybrid arm control.
        self._action_mode = "relative"
        self._debug_arm_limit = (
            os.getenv("LEHOME_DEBUG_LEKIWI_ARM_LIMIT", "0") == "1"
        )

    def _setup_scene(self):
        """Set up the scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        cfg = UsdFileCfg(usd_path=f"{LEKIWI_LOFT_USD_PATH}")
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )
        print(f"[LekiwiEnv] scene_usd={LEKIWI_LOFT_USD_PATH}")
        print(
            f"[LekiwiEnv] initial_pose=({tuple(self.cfg.robot_initial_position)}, "
            f"{tuple(self.cfg.robot_initial_orientation)})"
        )

        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.5)
        )

        if hasattr(self.cfg, 'front_camera'):
            self._front_camera = self.cfg.front_camera.class_type(
                self.cfg.front_camera
            )
            self.scene.sensors["front_camera"] = self._front_camera

        if os.getenv("LEHOME_DISABLE_GARMENT", "0") != "1":
            garment_usd = (
                "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/"
                "TCLC_002/TCLC_002_obj.usd"
            )
            garment_config = (
                "source/lehome/lehome/tasks/lekiwi_task/"
                "config_file/particle_garment_cfg.yaml"
            )
            self.garment = GarmentObject(
                prim_path="/World/Cloth",
                usd_path=garment_usd,
                visual_usd_path="Assets/Material/Garment/linen_Blue.usd",
                config=OmegaConf.load(garment_config),
            )

        # Garment initialization is delayed until after physics starts.
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _post_physics_step(self):
        """Handle post-physics-step updates."""
        if not hasattr(self, '_initial_pose_set'):
            self._set_robot_initial_pose()
            self._initial_pose_set = True

        if not hasattr(self, '_garment_initialized'):
            try:
                if hasattr(self, 'garment') and self.garment is not None:
                    self.garment.initialize()
                    print("[LekiwiEnv] Garment initialized.")
                    self._garment_initialized = True
            except Exception as e:
                print(f"[LekiwiEnv] Failed to initialize garment: {e}")
                import traceback
                traceback.print_exc()

        super()._post_physics_step()

    def _set_robot_initial_pose(self):
        """Set the initial robot pose."""
        try:
            initial_position = self._robot_initial_position  # X, Y, Z
            initial_orientation = self._robot_initial_orientation

            initial_pose = torch.cat([initial_position, initial_orientation])
            self._robot.write_root_pose_to_sim(initial_pose.unsqueeze(0))

            print(f"[LekiwiEnv] Initial position set to: {initial_position}")
            print(f"[LekiwiEnv] Initial orientation set to: {initial_orientation}")

        except Exception as e:
            print(f"[LekiwiEnv] Failed to set initial pose: {e}")
            import traceback
            traceback.print_exc()

    def set_action_mode(self, mode: str):
        """Set action mode: 'relative' or 'absolute'."""
        assert mode in ["relative", "absolute"]
        self._action_mode = mode
        print(f"[LekiwiEnv] action_mode={self._action_mode}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Preprocess actions before a physics step."""
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        self.actions = actions.clone()

    def _apply_action(self):
        """Apply actions to the robot."""
        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(0)

        current_joint_pos = self._robot.data.joint_pos
        joint_names = self._robot.data.joint_names

        # Base joints use relative control.
        base_joints = {
            "ST3215_Servo_Motor_v1_2_Revolute_60": 0,
            "ST3215_Servo_Motor_v1_1_Revolute_62": 1,
            "ST3215_Servo_Motor_v1_Revolute_64": 2,
        }

        # Arm and gripper joints may use relative or absolute control.
        arm_joints = {
            "STS3215_03a_v1_Revolute_45": 3,
            "STS3215_03a_v1_1_Revolute_49": 4,
            "STS3215_03a_v1_2_Revolute_51": 5,
            "STS3215_03a_v1_3_Revolute_53": 6,
            "STS3215_03a_Wrist_Roll_v1_Revolute_55": 7,
            "STS3215_03a_v1_4_Revolute_57": 8,
        }

        action_mode = getattr(self, "_action_mode", "relative")

        target_joint_pos = current_joint_pos.clone()

        if action_mode == "relative":
            for joint_name, action_idx in base_joints.items():
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    target_joint_pos[:, joint_idx] += (
                        self.actions[:, action_idx] * self._base_action_gain
                    )
            for joint_name, action_idx in arm_joints.items():
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    target_joint_pos[:, joint_idx] += (
                        self.actions[:, action_idx] * self._arm_action_gain
                    )
        else:
            for joint_name, action_idx in base_joints.items():
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    target_joint_pos[:, joint_idx] += (
                        self.actions[:, action_idx] * self._base_action_gain
                    )

            for joint_name, action_idx in arm_joints.items():
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    target_joint_pos[:, joint_idx] = (
                        self.actions[:, action_idx]
                    )

        target_joint_pos_before_clamp = target_joint_pos.clone()
        target_joint_pos = clamp_lekiwi_joint_targets(
            target_joint_pos, joint_names
        )
        if self._debug_arm_limit:
            changed = (
                torch.abs(target_joint_pos - target_joint_pos_before_clamp)
                > 1.0e-6
            )
            if torch.any(changed):
                hit_ids = torch.nonzero(changed[0], as_tuple=False).flatten()
                hit_names = [joint_names[i] for i in hit_ids.tolist()]
                print(f"[LekiwiEnv] arm_limit_hit joints={hit_names}")
        self._robot.set_joint_position_target(target_joint_pos)

    def _get_observations(self):
        """Return policy observations."""
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        observations = torch.cat([joint_pos, joint_vel], dim=-1)

        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        """Teleoperation does not use rewards."""
        return torch.zeros((self.num_envs, 1), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Teleoperation episodes do not terminate automatically."""
        return (
            torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            ),
            torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )
        )

    def _reset_idx(self, env_ids):
        """Reset selected environments."""
        print(f"[LekiwiEnv] Resetting env_ids={env_ids}")

        self._robot.write_joint_position_to_sim(
            self._robot.data.default_joint_pos[env_ids], env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim(
            self._robot.data.default_joint_vel[env_ids], env_ids=env_ids
        )

        initial_position = self._robot_initial_position
        initial_orientation = self._robot_initial_orientation
        initial_pose = torch.cat([initial_position, initial_orientation])
        self._robot.write_root_pose_to_sim(
            initial_pose.unsqueeze(0), env_ids=env_ids
        )
        print(f"[LekiwiEnv] Robot reset position: {initial_position}")

        self.actions[env_ids] = 0.0

        if hasattr(self, 'garment') and self.garment is not None:
            try:
                if hasattr(self.garment, 'initial_points_positions'):
                    print("[LekiwiEnv] Resetting garment.")
                    self.garment.reset()
                    print("[LekiwiEnv] Garment reset complete.")
                else:
                    print("[LekiwiEnv] Garment not initialized; initializing now.")
                    self.garment.initialize()
                    print("[LekiwiEnv] Garment initialized; resetting now.")
                    self.garment.reset()
                    print("[LekiwiEnv] Garment reset complete.")
            except Exception as e:
                print(f"[LekiwiEnv] Failed to reset garment: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[LekiwiEnv] No garment configured; skipping garment reset.")

        super()._reset_idx(env_ids)
        print("[LekiwiEnv] Reset complete.")
