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
# 衣服相关的导入
from lehome.assets.object.Garment import GarmentObject
from lehome.devices.lekiwi_action_process import clamp_lekiwi_joint_targets
from omegaconf import OmegaConf


class LekiwiEnv(DirectRLEnv):
    cfg: LekiwiEnvCfg

    def __init__(self, cfg: LekiwiEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # 初始化动作缓冲区
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
        # 动作模式：relative | absolute（默认relative，向后兼容）
        self._action_mode = "relative"
        self._debug_arm_limit = (
            os.getenv("LEHOME_DEBUG_LEKIWI_ARM_LIMIT", "0") == "1"
        )

    def _setup_scene(self):
        # 创建机器人
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        """设置场景"""
        # 加载 Lekiwi 场景
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

        # 添加地面平面
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.5)
        )

        # 添加摄像头到场景
        if hasattr(self.cfg, 'front_camera'):
            self._front_camera = self.cfg.front_camera.class_type(
                self.cfg.front_camera
            )
            self.scene.sensors["front_camera"] = self._front_camera

        if os.getenv("LEHOME_DISABLE_GARMENT", "0") != "1":
            # 添加衣服到场景
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

        # 不要在这里初始化衣服，延迟到物理仿真启动后

        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)

        # 添加光源
        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _post_physics_step(self):
        """物理步进后的处理"""
        # 在第一次物理步进后设置机器人位置
        if not hasattr(self, '_initial_pose_set'):
            self._set_robot_initial_pose()
            self._initial_pose_set = True

        # 在物理仿真启动后初始化衣服
        if not hasattr(self, '_garment_initialized'):
            try:
                if hasattr(self, 'garment') and self.garment is not None:
                    self.garment.initialize()
                    print("✅ 衣服对象初始化成功")
                    self._garment_initialized = True
            except Exception as e:
                print(f"❌ 衣服对象初始化失败: {e}")
                import traceback
                traceback.print_exc()

        # 调用父类方法
        super()._post_physics_step()

    def _set_robot_initial_pose(self):
        """设置机器人初始位姿"""
        try:
            # 设置机器人初始位置
            initial_position = self._robot_initial_position  # X, Y, Z
            initial_orientation = self._robot_initial_orientation  # 四元数

            # 组合位姿
            initial_pose = torch.cat([initial_position, initial_orientation])

            # 应用位姿
            self._robot.write_root_pose_to_sim(initial_pose.unsqueeze(0))

            print(f"Lekiwi初始位置设置为: {initial_position}")
            print(f"Lekiwi初始朝向设置为: {initial_orientation}")

        except Exception as e:
            print(f"设置机器人初始位姿时出错: {e}")
            import traceback
            traceback.print_exc()

    def set_action_mode(self, mode: str):
        """设置动作模式: 'relative' | 'absolute'"""
        assert mode in ["relative", "absolute"]
        self._action_mode = mode
        print(f"LekiwiEnv 动作模式: {self._action_mode}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """预处理动作"""
        # 确保actions是正确的形状
        if actions.dim() == 1:
            # 如果是1维，扩展为2维 [num_envs, action_dim]
            actions = actions.unsqueeze(0)

        # 存储动作
        self.actions = actions.clone()

    def _apply_action(self):
        """应用动作到机器人 - 支持混合模式（底盘相对+机械臂绝对）"""
        # 确保actions是2维的
        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(0)

        # 获取当前关节位置
        current_joint_pos = self._robot.data.joint_pos

        # 使用关节名称进行映射
        joint_names = self._robot.data.joint_names

        # 定义底盘关节（使用相对控制）
        base_joints = {
            "ST3215_Servo_Motor_v1_2_Revolute_60": 0,
            "ST3215_Servo_Motor_v1_1_Revolute_62": 1,
            "ST3215_Servo_Motor_v1_Revolute_64": 2,
        }

        # 定义机械臂和夹爪关节（使用绝对控制）
        arm_joints = {
            "STS3215_03a_v1_Revolute_45": 3,
            "STS3215_03a_v1_1_Revolute_49": 4,
            "STS3215_03a_v1_2_Revolute_51": 5,
            "STS3215_03a_v1_3_Revolute_53": 6,
            "STS3215_03a_Wrist_Roll_v1_Revolute_55": 7,
            "STS3215_03a_v1_4_Revolute_57": 8,
        }

        # 动作模式：relative（纯键盘）或 hybrid（混合）
        action_mode = getattr(self, "_action_mode", "relative")

        target_joint_pos = current_joint_pos.clone()

        if action_mode == "relative":
            # 纯相对控制模式（键盘控制）
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
            # 混合模式：底盘相对 + 机械臂绝对
            # 底盘使用相对控制（增量累积）
            for joint_name, action_idx in base_joints.items():
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                    target_joint_pos[:, joint_idx] += (
                        self.actions[:, action_idx] * self._base_action_gain
                    )

            # 机械臂和夹爪使用绝对控制（直接设置位置）
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
        """获取观察值"""
        # 获取机器人关节位置和速度
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # 组合观察值
        observations = torch.cat([joint_pos, joint_vel], dim=-1)

        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        """键盘控制场景 - 不需要奖励"""
        return torch.zeros((self.num_envs, 1), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """键盘控制场景 - 永不结束"""
        return (
            torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            ),
            torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )
        )

    def _reset_idx(self, env_ids):
        """重置指定环境"""
        print(f"开始重置Lekiwi环境 {env_ids}")

        # 重置机器人状态
        self._robot.write_joint_position_to_sim(
            self._robot.data.default_joint_pos[env_ids], env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim(
            self._robot.data.default_joint_vel[env_ids], env_ids=env_ids
        )

        # 重置机器人到初始位置
        initial_position = self._robot_initial_position
        initial_orientation = self._robot_initial_orientation
        initial_pose = torch.cat([initial_position, initial_orientation])
        self._robot.write_root_pose_to_sim(
            initial_pose.unsqueeze(0), env_ids=env_ids
        )
        print(f"Lekiwi重置到位置: {initial_position}")

        # 重置动作缓冲区
        self.actions[env_ids] = 0.0

        # 重置衣服（如果存在且已初始化）
        if hasattr(self, 'garment') and self.garment is not None:
            try:
                # 检查衣服对象是否已初始化
                if hasattr(self.garment, 'initial_points_positions'):
                    print("开始重置衣服...")
                    self.garment.reset()
                    print("✅ 衣服重置成功")
                else:
                    print("⚠️ 衣服对象未初始化，尝试初始化...")
                    self.garment.initialize()
                    print("✅ 衣服对象初始化成功，现在可以重置")
                    self.garment.reset()
                    print("✅ 衣服重置成功")
            except Exception as e:
                print(f"❌ 衣服重置失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ 衣服不存在，跳过重置")

        # 调用父类重置方法
        super()._reset_idx(env_ids)
        print("Lekiwi环境重置完成")
