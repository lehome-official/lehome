import os
from pathlib import Path
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.assets import Articulation

from .xlerobot_cfg import XlerobotEnvCfg
from lehome.assets.scenes.loft import XLEROBOT_LOFT_USD_PATH
from lehome.assets.robots.xlerobot import XLEROBOT_JOINT_LIMITS
from lehome.assets.object.Garment import GarmentObject
from lehome.utils.constant import ASSETS_ROOT
from omegaconf import OmegaConf


class XlerobotEnv(DirectRLEnv):
    cfg: XlerobotEnvCfg

    _BASE_JOINT_NAMES = (
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    )

    _ARM_ACTION_MAPPING = (
        ("Rotation", 1),
        ("Pitch", 2),
        ("Elbow", 3),
        ("Wrist_Pitch", 4),
        ("Wrist_Roll", 5),
        ("Jaw", 6),
        ("Rotation_2", 7),
        ("Pitch_2", 8),
        ("Elbow_2", 9),
        ("Wrist_Pitch_2", 10),
        ("Wrist_Roll_2", 11),
        ("Jaw_2", 12),
        ("head_pan_joint", 13),
        ("head_tilt_joint", 14),
    )

    def __init__(self, cfg: XlerobotEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.count_abc = 0
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self._robot.data.joint_pos
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self._robot_initial_position = torch.tensor(self.cfg.robot_initial_position, device=self.device)
        self._robot_initial_orientation = torch.tensor(self.cfg.robot_initial_orientation, device=self.device)

        # Action mode: relative for keyboard, absolute for hybrid arm control.
        self._action_mode = "relative"

        # Relative-control scales used by keyboard mode.
        self._rel_arm_scale = float(os.getenv("LEHOME_XLEROBOT_REL_ARM_SCALE", "3.0"))
        self._rel_gripper_scale = float(os.getenv("LEHOME_XLEROBOT_REL_GRIPPER_SCALE", "0.75"))
        self._rel_head_scale = float(os.getenv("LEHOME_XLEROBOT_REL_HEAD_SCALE", "0.5"))

        # Absolute arm target filtering used by leader/hybrid modes.
        self._arm_abs_filter_alpha = float(os.getenv("LEHOME_XLEROBOT_ARM_ABS_FILTER_ALPHA", "1.0"))
        self._arm_abs_filter_alpha = float(min(max(self._arm_abs_filter_alpha, 0.0), 1.0))
        self._arm_abs_deadband = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_ABS_DEADBAND", "0.0")))
        self._arm_abs_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_ABS_MAX_STEP", "0.0")))
        self._jaw_abs_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_JAW_ABS_MAX_STEP", "0.18")))
        self._debug_arm_limit = os.getenv("LEHOME_DEBUG_XLEROBOT_ARM_LIMIT", "0") == "1"
        self._debug_arm_filter = os.getenv("LEHOME_DEBUG_XLEROBOT_ARM_FILTER", "0") == "1"

        # Base-control parameters.
        self._base_vxy_max = 1.2  # m/s
        self._base_wz_max = 2.4  # rad/s
        self._base_axy_max = 8.0  # m/s^2
        self._base_awz_max = 16.0  # rad/s^2
        self._base_control_dt = max(float(self.cfg.sim.dt * self.cfg.decimation), 1e-6)

        # Base-command cache in body frame and velocity cache in world frame.
        self._base_command_body = torch.zeros((self.num_envs, 3), device=self.device)
        self._base_target_vel_world = torch.zeros((self.num_envs, 3), device=self.device)
        self._base_current_vel_world = torch.zeros((self.num_envs, 3), device=self.device)
        self._base_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Base-control mode:
        # - joint_velocity: drive root_x/y/z dummy joints with position/velocity targets.
        # - root_velocity: continuously write root velocity, the default compatibility path.
        # - auto: start with root_velocity and fall back to joint_velocity for Direct GPU API.
        requested_mode = os.getenv("LEHOME_XLEROBOT_BASE_CONTROL", "root_velocity").strip().lower()
        if requested_mode not in {"auto", "root_velocity", "joint_velocity"}:
            requested_mode = "root_velocity"
        self._base_control_mode_requested = requested_mode
        self._direct_gpu_api_enabled = self._is_direct_gpu_api_enabled()
        if requested_mode == "auto" and self._direct_gpu_api_enabled:
            self._base_control_mode = "joint_velocity"
            print(
                "[XlerobotEnv] Direct GPU API detected in auto mode; switching "
                "base control from root_velocity to joint_velocity."
            )
        elif requested_mode == "auto":
            self._base_control_mode = "root_velocity"
        else:
            self._base_control_mode = requested_mode

        # Resolve joint ids and fail fast on incompatible assets.
        self._resolve_joint_ids_or_fail()
        self._debug_base_env = os.getenv("LEHOME_DEBUG_XLEROBOT_ENV", "0") == "1"
        self._debug_step = 0
        self._base_target_pos[:] = self._robot.data.joint_pos[:, self._base_joint_ids]
        self._arm_abs_filtered_target = self._robot.data.joint_pos[:, self._arm_joint_ids].clone()
        print(
            f"[XlerobotEnv] base_control_mode={self._base_control_mode} "
            f"(requested={self._base_control_mode_requested})"
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._tune_gripper_collision_for_stability()

        cfg = UsdFileCfg(usd_path=f"{XLEROBOT_LOFT_USD_PATH}")
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )
        print(f"[XlerobotEnv] scene_usd={XLEROBOT_LOFT_USD_PATH}")
        print(
            f"[XlerobotEnv] initial_pose=({tuple(self.cfg.robot_initial_position)}, "
            f"{tuple(self.cfg.robot_initial_orientation)})"
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.5))

        if hasattr(self.cfg, "front_camera"):
            self._front_camera = self.cfg.front_camera.class_type(self.cfg.front_camera)
            self.scene.sensors["front_camera"] = self._front_camera

        if os.getenv("LEHOME_DISABLE_GARMENT", "0") != "1":
            task_dir = Path(__file__).resolve().parent
            garment_cfg = task_dir / "particle_garment_cfg.yaml"
            self.garment = GarmentObject(
                prim_path="/World/Cloth",
                usd_path=str(
                    Path(ASSETS_ROOT)
                    / "Garment"
                    / "Tops"
                    / "Collar_Lsleeve_FrontClose"
                    / "TCLC_002"
                    / "TCLC_002_obj.usd"
                ),
                visual_usd_path=str(
                    Path(ASSETS_ROOT) / "Material" / "Garment" / "linen_Blue.usd"
                ),
                config=OmegaConf.load(str(garment_cfg)),
            )

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _tune_gripper_collision_for_stability(self):
        """Reduce jaw collision 'air-wall' effect by refining collision approximation/offsets."""
        try:
            import omni.usd
            from pxr import PhysxSchema
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # Tunable knobs for quick on-machine iteration.
        approximation = os.getenv(
            "LEHOME_XLEROBOT_GRIPPER_COLLISION_APPROX", "convexDecomposition"
        )
        contact_offset = float(os.getenv("LEHOME_XLEROBOT_GRIPPER_CONTACT_OFFSET", "0.0015"))
        rest_offset = float(os.getenv("LEHOME_XLEROBOT_GRIPPER_REST_OFFSET", "0.0"))

        base_path = str(self.cfg.robot.prim_path).rstrip("/")
        base_candidates = [base_path, f"{base_path}/xlerobot"]
        rel_paths = [
            "/Fixed_Jaw/collisions",
            "/Moving_Jaw/collisions",
            "/Fixed_Jaw_2/collisions",
            "/Moving_Jaw_2/collisions",
        ]

        updated = []
        for base in base_candidates:
            for rel in rel_paths:
                prim_path = f"{base}{rel}"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                try:
                    px_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    if px_col and px_col.GetApproximationAttr():
                        px_col.GetApproximationAttr().Set(approximation)
                    if px_col and px_col.GetContactOffsetAttr():
                        px_col.GetContactOffsetAttr().Set(contact_offset)
                    if px_col and px_col.GetRestOffsetAttr():
                        px_col.GetRestOffsetAttr().Set(rest_offset)
                    updated.append(prim_path)
                except Exception:
                    # Best effort: keep teleop running even if schema ops fail on this prim.
                    continue

        if updated:
            print(
                "[XlerobotEnv] gripper_collision_tuned "
                f"approx={approximation} contact_offset={contact_offset} rest_offset={rest_offset} "
                f"count={len(updated)}"
            )

    def _is_direct_gpu_api_enabled(self) -> bool:
        """Best-effort detection of PhysX Direct GPU API mode."""
        # 1) Try scene schema attribute.
        try:
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                for prim in stage.Traverse():
                    if not prim.IsA(UsdPhysics.Scene):
                        continue
                    for attr_name in (
                        "physxScene:enableDirectGPUAPI",
                        "physxScene:enableDirectGpuApi",
                    ):
                        attr = prim.GetAttribute(attr_name)
                        if attr and attr.HasAuthoredValueOpinion():
                            value = attr.Get()
                            if value is not None:
                                return bool(value)
        except Exception:
            pass

        # 2) Try carb settings keys used in some Isaac Sim versions.
        try:
            import carb

            settings = carb.settings.get_settings()
            for key in (
                "/physics/enableDirectGPUAPI",
                "/physics/directGpuApi",
                "/persistent/physics/enableDirectGPUAPI",
            ):
                value = settings.get(key)
                if isinstance(value, bool):
                    return value
        except Exception:
            pass

        return False

    def _can_write_root_state(self) -> bool:
        """Whether root pose/velocity writes are allowed in current runtime."""
        if self._direct_gpu_api_enabled:
            return False
        return True

    def _resolve_joint_ids_or_fail(self):
        """Resolve joint ids and validate required base dummy joints."""
        joint_names = self._robot.data.joint_names

        missing_base = [name for name in self._BASE_JOINT_NAMES if name not in joint_names]
        if missing_base:
            raise RuntimeError(
                "Xlerobot base joints are missing; continuous base control cannot start."
                f" missing={missing_base}. "
                "Run `python scripts/tools/inspect_xlerobot_usd.py` to inspect the USD joint definitions."
            )

        self._base_joint_id_map = {name: joint_names.index(name) for name in self._BASE_JOINT_NAMES}
        self._base_joint_ids = [self._base_joint_id_map[name] for name in self._BASE_JOINT_NAMES]

        self._arm_joint_ids = []
        self._arm_action_indices = []
        self._arm_joint_names = []
        for joint_name, action_index in self._ARM_ACTION_MAPPING:
            if joint_name in joint_names:
                self._arm_joint_ids.append(joint_names.index(joint_name))
                self._arm_action_indices.append(action_index)
                self._arm_joint_names.append(joint_name)

        if not self._arm_joint_ids:
            raise RuntimeError("No Xlerobot arm/head joints were mapped; check the robot USD and action mapping.")

        # Build explicit arm/gripper/head limits, preferring runtime USD limits.
        arm_lower = []
        arm_upper = []
        runtime_limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        use_runtime_limits = runtime_limits is not None

        if use_runtime_limits:
            runtime_limits = runtime_limits[0] if runtime_limits.dim() == 3 else runtime_limits

        for joint_id, name in zip(self._arm_joint_ids, self._arm_joint_names):
            if use_runtime_limits:
                lo = float(runtime_limits[joint_id, 0].item())
                hi = float(runtime_limits[joint_id, 1].item())
            elif name in XLEROBOT_JOINT_LIMITS:
                lo, hi = XLEROBOT_JOINT_LIMITS[name]
            else:
                lo, hi = (-1.0e9, 1.0e9)
            arm_lower.append(float(lo))
            arm_upper.append(float(hi))

        self._arm_joint_lower_limits = torch.tensor(arm_lower, device=self.device)
        self._arm_joint_upper_limits = torch.tensor(arm_upper, device=self.device)

        # Per-joint relative scales keep gripper steps smaller than arm steps.
        rel_scales = []
        for name in self._arm_joint_names:
            if "Jaw" in name:
                rel_scales.append(self._rel_gripper_scale)
            elif name.startswith("head_"):
                rel_scales.append(self._rel_head_scale)
            else:
                rel_scales.append(self._rel_arm_scale)
        self._arm_relative_scales = torch.tensor(rel_scales, device=self.device, dtype=torch.float32)

        # Per-joint absolute target step limits, with smaller jaw steps.
        abs_steps = []
        for name in self._arm_joint_names:
            if "Jaw" in name:
                abs_steps.append(self._jaw_abs_max_step)
            else:
                abs_steps.append(self._arm_abs_max_step)
        self._arm_abs_max_step_per_joint = torch.tensor(abs_steps, device=self.device, dtype=torch.float32)

        if self._debug_arm_limit:
            limit_src = "runtime_soft_joint_pos_limits" if use_runtime_limits else "XLEROBOT_JOINT_LIMITS"
            print(f"[XlerobotEnv] arm_limit_source={limit_src}")
            for i, name in enumerate(self._arm_joint_names):
                print(
                    f"[XlerobotEnv] arm_limit {name}: "
                    f"[{float(self._arm_joint_lower_limits[i]):.4f}, {float(self._arm_joint_upper_limits[i]):.4f}]"
                )

        body_names = list(self._robot.data.body_names)
        self._base_body_id = body_names.index("base_link") if "base_link" in body_names else 0

    def set_base_command(self, command):
        """Set the base command in body frame.

        Accepts shape (3,) or (num_envs, 3), interpreted as normalized
        (vx_body, vy_body, wz) in [-1, 1].
        """
        command_tensor = torch.as_tensor(command, dtype=torch.float32, device=self.device)

        if command_tensor.ndim == 1:
            if command_tensor.numel() != 3:
                raise ValueError(f"Base command must contain 3 values, got shape: {tuple(command_tensor.shape)}")
            command_tensor = command_tensor.unsqueeze(0)

        if command_tensor.shape[-1] != 3:
            raise ValueError(f"Base command last dimension must be 3, got shape: {tuple(command_tensor.shape)}")

        if command_tensor.shape[0] == 1 and self.num_envs > 1:
            command_tensor = command_tensor.repeat(self.num_envs, 1)
        elif command_tensor.shape[0] != self.num_envs:
            raise ValueError(
                f"Base command batch size does not match num_envs: batch={command_tensor.shape[0]}, num_envs={self.num_envs}"
            )

        self._base_command_body[:] = torch.clamp(command_tensor, -1.0, 1.0)

    def _post_physics_step(self):
        """Handle post-physics-step updates."""
        if not hasattr(self, "_initial_pose_set"):
            self._set_robot_initial_pose()
            self._initial_pose_set = True

        if not hasattr(self, "_garment_initialized"):
            try:
                if hasattr(self, "garment") and self.garment is not None:
                    self.garment.initialize()
                    print("[XlerobotEnv] Garment initialized.")
                    self._garment_initialized = True
            except Exception as e:
                print(f"[XlerobotEnv] Failed to initialize garment: {e}")
                import traceback
                traceback.print_exc()

        super()._post_physics_step()

    def _set_robot_initial_pose(self):
        """Set the initial robot pose."""
        try:
            initial_pose = torch.cat([self._robot_initial_position, self._robot_initial_orientation])
            if self._can_write_root_state():
                self._robot.write_root_pose_to_sim(initial_pose.unsqueeze(0))

            # Clear base command and velocity targets.
            self._base_command_body.zero_()
            self._base_target_vel_world.zero_()
            self._base_current_vel_world.zero_()
            self._robot.set_joint_velocity_target(
                torch.zeros((self.num_envs, 3), device=self.device),
                joint_ids=self._base_joint_ids,
            )
            if self._can_write_root_state():
                self._robot.write_root_velocity_to_sim(
                    torch.zeros((self.num_envs, 6), device=self.device)
                )
            self._base_target_pos[:] = self._robot.data.joint_pos[:, self._base_joint_ids]
            self._robot.set_joint_position_target(
                self._base_target_pos,
                joint_ids=self._base_joint_ids,
            )
            self._arm_abs_filtered_target[:] = self._robot.data.joint_pos[:, self._arm_joint_ids]

            print(f"[XlerobotEnv] Initial position set to: {self._robot_initial_position}")
            print(f"[XlerobotEnv] Initial orientation set to: {self._robot_initial_orientation}")

        except Exception as e:
            print(f"[XlerobotEnv] Failed to set initial pose: {e}")
            import traceback
            traceback.print_exc()

    def set_action_mode(self, mode: str):
        """Set action mode: 'relative' or 'absolute'."""
        assert mode in ["relative", "absolute"]
        self._action_mode = mode
        print(f"[XlerobotEnv] action_mode={self._action_mode}")

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

        # Arms and head support relative and absolute target modes.
        arm_target_joint_pos = current_joint_pos[:, self._arm_joint_ids].clone()
        action_mode = getattr(self, "_action_mode", "relative")

        if action_mode == "relative":
            input_action = self.actions[:, self._arm_action_indices]
            scales = self._arm_relative_scales.unsqueeze(0)
            arm_target_joint_pos += input_action * scales
        else:
            for i, action_idx in enumerate(self._arm_action_indices):
                arm_target_joint_pos[:, i] = self.actions[:, action_idx]
            arm_target_joint_pos = self._filter_absolute_arm_target(arm_target_joint_pos)

        # Clamp to explicit Xlerobot joint limits.
        raw_target_before_clamp = arm_target_joint_pos.clone()
        arm_target_joint_pos = torch.max(
            torch.min(arm_target_joint_pos, self._arm_joint_upper_limits),
            self._arm_joint_lower_limits,
        )
        if self._debug_arm_limit:
            hit_lower = raw_target_before_clamp < (self._arm_joint_lower_limits + 1.0e-6)
            hit_upper = raw_target_before_clamp > (self._arm_joint_upper_limits - 1.0e-6)
            if torch.any(hit_lower | hit_upper):
                hit_idx = torch.nonzero(hit_lower[0] | hit_upper[0], as_tuple=False).flatten().tolist()
                if hit_idx:
                    hit_names = [self._arm_joint_names[i] for i in hit_idx]
                    print(f"[XlerobotEnv] arm_limit_hit joints={hit_names}")

        self._arm_abs_filtered_target[:] = arm_target_joint_pos
        self._robot.set_joint_position_target(arm_target_joint_pos, joint_ids=self._arm_joint_ids)

        self._apply_base_velocity_command()

    @staticmethod
    def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternions in wxyz order."""
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _get_control_yaw(self) -> tuple[torch.Tensor, str]:
        """Return yaw for base control, preferring visible base_link pose."""
        if hasattr(self._robot.data, "body_link_quat_w"):
            quat = self._robot.data.body_link_quat_w[:, self._base_body_id]
            return self._yaw_from_quat_wxyz(quat), "base_link_quat_w"

        # Compatibility fallback from root_link_pose_w, interpreted as wxyz.
        quat = self._robot.data.root_link_pose_w[:, 3:7]
        return self._yaw_from_quat_wxyz(quat), "root_link_pose_w"

    def _apply_base_velocity_command(self):
        """Convert body-frame base command to world-frame velocity targets."""
        # Normalized command to physical target velocity in body frame.
        desired_body_vel = self._base_command_body.clone()
        desired_body_vel[:, 0:2] *= self._base_vxy_max
        desired_body_vel[:, 2] *= self._base_wz_max

        # body -> world:
        # - joint_velocity: use base yaw joint.
        # - root_velocity: use robot root pose to avoid world-frame-only motion.
        if self._base_control_mode == "joint_velocity":
            yaw = self._robot.data.joint_pos[:, self._base_joint_id_map["root_z_rotation_joint"]]
            yaw_source = "joint"
        else:
            yaw, yaw_source = self._get_control_yaw()
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        self._base_target_vel_world[:, 0] = cos_yaw * desired_body_vel[:, 0] - sin_yaw * desired_body_vel[:, 1]
        self._base_target_vel_world[:, 1] = sin_yaw * desired_body_vel[:, 0] + cos_yaw * desired_body_vel[:, 1]
        self._base_target_vel_world[:, 2] = desired_body_vel[:, 2]

        # Velocity ramping avoids instantaneous jumps.
        max_delta_xy = self._base_axy_max * self._base_control_dt
        max_delta_wz = self._base_awz_max * self._base_control_dt

        delta = self._base_target_vel_world - self._base_current_vel_world
        self._base_current_vel_world[:, 0:2] += torch.clamp(delta[:, 0:2], -max_delta_xy, max_delta_xy)
        self._base_current_vel_world[:, 2] += torch.clamp(delta[:, 2], -max_delta_wz, max_delta_wz)

        if self._base_control_mode == "joint_velocity":
            self._base_target_pos += self._base_current_vel_world * self._base_control_dt
            self._robot.set_joint_position_target(self._base_target_pos, joint_ids=self._base_joint_ids)
            self._robot.set_joint_velocity_target(self._base_current_vel_world, joint_ids=self._base_joint_ids)
        else:
            root_vel = torch.zeros((self.num_envs, 6), device=self.device)
            root_vel[:, 0] = self._base_current_vel_world[:, 0]
            root_vel[:, 1] = self._base_current_vel_world[:, 1]
            root_vel[:, 5] = self._base_current_vel_world[:, 2]
            self._robot.write_root_velocity_to_sim(root_vel)

        if self._debug_base_env:
            self._debug_step += 1
            if self._debug_step % 30 == 0:
                q = self._robot.data.joint_pos[0, self._base_joint_ids]
                qd = self._robot.data.joint_vel[0, self._base_joint_ids]
                root_pos = self._robot.data.root_link_pose_w[0, :3]
                base_body_pos = self._robot.data.body_link_pos_w[0, self._base_body_id]
                print(
                    "[XlerobotEnv] "
                    f"mode={self._base_control_mode} "
                    f"yaw_source={yaw_source} "
                    f"yaw={float(yaw[0]):.4f} "
                    f"cmd_body={self._base_command_body[0].tolist()} "
                    f"target_vel_world={self._base_target_vel_world[0].tolist()} "
                    f"curr_vel_world={self._base_current_vel_world[0].tolist()} "
                    f"joint_pos={[float(v) for v in q]} "
                    f"joint_vel={[float(v) for v in qd]} "
                    f"root_pos={[float(v) for v in root_pos]} "
                    f"base_body_pos={[float(v) for v in base_body_pos]}"
                )

    def _get_observations(self):
        """Return policy observations."""
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        observations = torch.cat([joint_pos, joint_vel], dim=-1)
        return {"policy": observations}

    def _filter_absolute_arm_target(self, desired_target: torch.Tensor) -> torch.Tensor:
        """Filter absolute arm targets with deadband, low-pass, and step limits."""
        prev_target = self._arm_abs_filtered_target
        delta = desired_target - prev_target

        if self._arm_abs_deadband > 0.0:
            deadband_mask = torch.abs(delta) < self._arm_abs_deadband
            delta = torch.where(deadband_mask, torch.zeros_like(delta), delta)
        desired_after_deadband = prev_target + delta

        filtered = prev_target + self._arm_abs_filter_alpha * (desired_after_deadband - prev_target)
        if torch.any(self._arm_abs_max_step_per_joint > 0.0):
            max_step = self._arm_abs_max_step_per_joint.unsqueeze(0)
            raw_step = filtered - prev_target
            # max_step <= 0 means no per-step limit for that joint.
            limited_step = torch.clamp(raw_step, -max_step, max_step)
            step = torch.where(max_step > 0.0, limited_step, raw_step)
            filtered = prev_target + step

        if self._debug_arm_filter:
            delta_norm = torch.norm((desired_target - prev_target)[0]).item()
            filt_norm = torch.norm((filtered - prev_target)[0]).item()
            if delta_norm > 1.0e-6:
                print(
                    "[XlerobotEnv] arm_filter "
                    f"raw_step_norm={delta_norm:.5f} "
                    f"filtered_step_norm={filt_norm:.5f}"
                )
        return filtered

    def _get_rewards(self) -> torch.Tensor:
        """Teleoperation does not use rewards."""
        return torch.zeros((self.num_envs, 1), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Teleoperation episodes do not terminate automatically."""
        return (
            torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool),
            torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool),
        )

    def _reset_idx(self, env_ids):
        """Reset selected environments."""
        print(f"[XlerobotEnv] Resetting env_ids={env_ids}")

        self._robot.write_joint_position_to_sim(self._robot.data.default_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

        if self._can_write_root_state():
            initial_pose = torch.cat([self._robot_initial_position, self._robot_initial_orientation])
            self._robot.write_root_pose_to_sim(initial_pose.unsqueeze(0), env_ids=env_ids)
        print(f"[XlerobotEnv] Robot reset position: {self._robot_initial_position}")

        self.actions[env_ids] = 0.0

        # Clear base state to avoid residual velocity after reset.
        self._base_command_body[env_ids] = 0.0
        self._base_target_vel_world[env_ids] = 0.0
        self._base_current_vel_world[env_ids] = 0.0
        self._base_target_pos[env_ids] = self._robot.data.joint_pos[env_ids][:, self._base_joint_ids]
        self._robot.set_joint_position_target(
            self._base_target_pos[env_ids],
            joint_ids=self._base_joint_ids,
            env_ids=env_ids,
        )
        self._arm_abs_filtered_target[env_ids] = self._robot.data.joint_pos[env_ids][:, self._arm_joint_ids]
        env_count = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._robot.set_joint_velocity_target(
            torch.zeros((env_count, 3), device=self.device),
            joint_ids=self._base_joint_ids,
            env_ids=env_ids,
        )
        if self._can_write_root_state():
            self._robot.write_root_velocity_to_sim(
                torch.zeros((env_count, 6), device=self.device),
                env_ids=env_ids,
            )

        if hasattr(self, "garment") and self.garment is not None:
            try:
                if hasattr(self.garment, "initial_points_positions"):
                    print("[XlerobotEnv] Resetting garment.")
                    self.garment.reset()
                    print("[XlerobotEnv] Garment reset complete.")
                else:
                    print("[XlerobotEnv] Garment not initialized; initializing now.")
                    self.garment.initialize()
                    print("[XlerobotEnv] Garment initialized; resetting now.")
                    self.garment.reset()
                    print("[XlerobotEnv] Garment reset complete.")
            except Exception as e:
                print(f"[XlerobotEnv] Failed to reset garment: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[XlerobotEnv] No garment configured; skipping garment reset.")

        super()._reset_idx(env_ids)
        print("[XlerobotEnv] Reset complete.")
