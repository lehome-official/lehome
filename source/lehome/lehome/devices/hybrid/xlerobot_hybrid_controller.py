# source/lehome/lehome/devices/hybrid/xlerobot_hybrid_controller.py

import os
import numpy as np
import torch
from collections.abc import Callable

from ..device_base import Device
from ..lerobot import BiXlerobotLeader
from ..keyboard.xlerobot_keyboard import XlerobotKeyboard
from ...assets.robots.xlerobot import XLEROBOT_JOINT_LIMITS


class XlerobotHybridController(Device):
    """Xlerobot混合控制器：键盘控制底盘和头部，BiSO101Leader控制双臂"""
    
    def __init__(self, env, sensitivity: float = 1.0, 
                 left_arm_port: str = '/dev/ttyACM0', 
                 right_arm_port: str = '/dev/ttyACM1', 
                 recalibrate: bool = False):
        super().__init__(env)
        
        # 创建键盘控制器（仅用于底盘和头部控制）
        self.keyboard_controller = XlerobotKeyboard(env, sensitivity)
        
        # 创建双臂控制器
        self.bi_arm_controller = BiXlerobotLeader(
            env, 
            left_port=left_arm_port, 
            right_port=right_arm_port, 
            recalibrate=recalibrate
        )
        
        # 控制模式标志
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arms_only"
        
        # 状态标志
        self.started = False
        self._reset_state = False
        self._lerobot_compat = os.getenv("LEHOME_XLEROBOT_LEROBOT_COMPAT", "0") == "1"
        if self._lerobot_compat:
            print("XlerobotHybridController: lerobot-compatible arm mapping enabled.")

        # 机械臂抗抖参数（主要用于leader绝对控制）
        self._arm_smooth_alpha = float(os.getenv("LEHOME_XLEROBOT_ARM_SMOOTH_ALPHA", "0.25"))
        self._arm_smooth_alpha = float(np.clip(self._arm_smooth_alpha, 0.0, 1.0))
        self._arm_deadband = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_DEADBAND", "0.01")))
        self._arm_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_MAX_STEP", "0.05")))
        self._jaw_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_JAW_MAX_STEP", "0.18")))

        # 第二关节（shoulder_lift）轻微量程补偿，解决“收不回去”问题。
        self._shoulder_lift_gain = float(os.getenv("LEHOME_XLEROBOT_SHOULDER_LIFT_GAIN", "1.15"))
        self._shoulder_lift_gain = max(1.0, self._shoulder_lift_gain)

        self._left_arm_filtered = None
        self._right_arm_filtered = None
        self._latest_state_vector = None
        self._cached_arms_action = None

        # 默认启用 F6 模式切换快捷键
        self.add_callback("F6", lambda: None)
        
    def set_control_mode(self, mode: str):
        """设置控制模式"""
        assert mode in ["keyboard", "hybrid", "arms_only"], f"Invalid control mode: {mode}"
        if mode != self.control_mode:
            # 切模式时清空滤波缓存，避免旧模式残留目标带来突变。
            self._left_arm_filtered = None
            self._right_arm_filtered = None
            self._latest_state_vector = None
            self._cached_arms_action = None
            # 切到非键盘底盘模式时，强制清零底盘命令，避免残留速度命令。
            if mode == "arms_only":
                self._clear_base_command()
        self.control_mode = mode
        print(f"控制模式切换为: {mode}")

    def _clear_base_command(self):
        if hasattr(self.env, "set_base_command"):
            self.env.set_base_command(np.zeros(3, dtype=np.float32))

    def _build_arms_state_vector(self, arms_joint_state: dict, motor_limits: dict) -> np.ndarray:
        """Build a 15-dim action vector from bi-arm state only (no base/head)."""
        full_state = np.zeros(15, dtype=np.float32)

        if "left_arm" in arms_joint_state and arms_joint_state["left_arm"] is not None:
            left_arm_data = arms_joint_state["left_arm"]
            if isinstance(left_arm_data, dict):
                left_motor_limits = motor_limits.get("left_arm", {})
                left_processed = self._convert_arm_action(left_arm_data, left_motor_limits)
                left_processed = self._smooth_arm_action(left_processed, arm_side="left")
                full_state[1:7] = left_processed

        if "right_arm" in arms_joint_state and arms_joint_state["right_arm"] is not None:
            right_arm_data = arms_joint_state["right_arm"]
            if isinstance(right_arm_data, dict):
                right_motor_limits = motor_limits.get("right_arm", {})
                right_processed = self._convert_arm_action(right_arm_data, right_motor_limits)
                right_processed = self._smooth_arm_action(right_processed, arm_side="right")
                full_state[7:13] = right_processed

        return full_state

    def _build_hybrid_state_vector(
        self,
        keyboard_state: np.ndarray,
        arms_joint_state: dict,
        motor_limits: dict,
    ) -> np.ndarray:
        """Build a 15-dim action vector: keyboard for base/head + leaders for arms."""
        hybrid_state = self._build_arms_state_vector(arms_joint_state, motor_limits)
        hybrid_state[0] = keyboard_state[0]
        hybrid_state[13:15] = keyboard_state[13:15]
        return hybrid_state
        
    def get_device_state(self):
        """获取设备状态"""
        if self.control_mode == "keyboard":
            return self.keyboard_controller.get_device_state()
        elif self.control_mode == "arms_only":
            self._clear_base_command()
            arms_action = self._cached_arms_action or self.bi_arm_controller.input2action()
            arms_state = arms_action.get("joint_state", {})
            motor_limits = arms_action.get("motor_limits", {})
            return self._build_arms_state_vector(arms_state, motor_limits)
        else:  # hybrid mode
            keyboard_state = self.keyboard_controller.get_device_state()
            arms_action = self._cached_arms_action or self.bi_arm_controller.input2action()
            arms_state = arms_action.get("joint_state", {})
            motor_limits = arms_action.get("motor_limits", {})
            return self._build_hybrid_state_vector(keyboard_state, arms_state, motor_limits)

    def input2action(self):
        """将输入转换为动作"""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            self._latest_state_vector = action.get("joint_state")
            self._cached_arms_action = None
            return action
        elif self.control_mode == "arms_only":
            action = self.bi_arm_controller.input2action()
            self.started = action.get("started", False)
            # 更新缓存，并直接用本帧缓存构建动作，避免重复串口读取
            self._cached_arms_action = action
            self._clear_base_command()
            self._latest_state_vector = self._build_arms_state_vector(
                action.get("joint_state", {}),
                action.get("motor_limits", {}),
            )
            return action
        else:  # hybrid mode
            keyboard_action = self.keyboard_controller.input2action()
            arms_action = self.bi_arm_controller.input2action()
            
            # hybrid 模式允许键盘或双臂任一侧触发启动
            self.started = (
                keyboard_action.get("started", False)
                or arms_action.get("started", False)
            )
            
            # 更新缓存
            self._cached_arms_action = arms_action
            keyboard_state = keyboard_action.get("joint_state")
            if keyboard_state is None:
                keyboard_state = self.keyboard_controller.get_device_state()
            hybrid_joint_state = self._build_hybrid_state_vector(
                keyboard_state,
                arms_action.get("joint_state", {}),
                arms_action.get("motor_limits", {}),
            )
            
            hybrid_action = {
                "reset": keyboard_action.get("reset", False) or arms_action.get("reset", False),
                "started": self.started,
                "hybrid_controller": True,
                "keyboard": True,
                "bi_xlerobot_leader": True,
                "bi_so101_leader": True,  # backward-compatible alias
                "joint_state": hybrid_joint_state,
                "motor_limits": arms_action.get("motor_limits", {}),
            }
            self._latest_state_vector = hybrid_action.get("joint_state")
            
            return hybrid_action
    
    def advance(self):
        """获取当前动作"""
        if not self.started:
            return None

        # 优先使用input2action()阶段缓存，避免同一帧重复读串口导致抖动。
        action = self._latest_state_vector if self._latest_state_vector is not None else self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)
    
    def reset(self):
        """重置控制器状态"""
        self.keyboard_controller.reset()
        self.bi_arm_controller.reset()
        self.started = False
        self._reset_state = False
        self._left_arm_filtered = None
        self._right_arm_filtered = None
        self._latest_state_vector = None
        self._cached_arms_action = None
        self._clear_base_command()
    
    def add_callback(self, key: str, func: Callable):
        """添加回调函数"""
        # 键盘回调用于控制模式切换
        if key == "F6":  # F6键切换控制模式
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arms_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
        else:
            # 其他回调传递给键盘控制器
            self.keyboard_controller.add_callback(key, func)
            
            # 对于B键，也需要传递给BiSO101Leader以启动双臂控制
            if key == "B":
                # 直接传递B键回调给BiSO101Leader
                self.bi_arm_controller.add_callback(key, func)
            # 其他键不传递给BiSO101Leader，避免不必要的按键处理
    
    def __str__(self) -> str:
        """返回控制器信息"""
        msg = "Xlerobot混合控制器\n"
        msg += f"\t当前控制模式: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t键盘控制: 底盘移动(W/A/S/D) + 头部运动(Home/End/PageUp/PageDown)\n"
        msg += "\t双臂控制器: 机械臂和夹爪控制\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: 切换控制模式 (hybrid <-> keyboard <-> arms_only)\n"
        msg += "\tB: 启动控制\n"
        msg += "\tF5: 重置环境\n"
        msg += "\t退出: Ctrl+C\n"
        return msg

    def _convert_arm_action(self, joint_state: dict, motor_limits: dict) -> np.ndarray:
        """转换单臂动作，使用与原有BiSO101Leader相同的转换逻辑"""
        processed_action = np.zeros(6)
        
        # 如果没有motor_limits，返回零动作
        if not motor_limits:
            print("警告：没有找到motor_limits，返回零动作")
            return processed_action
        
        # 使用与原有代码相同的关节限制（角度）
        from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
        
        # 关节名称到索引的映射
        joint_mapping = {
            'shoulder_pan': 0,
            'shoulder_lift': 1,
            'elbow_flex': 2,
            'wrist_flex': 3,
            'wrist_roll': 4,
            'gripper': 5
        }
        
        # 关节方向校正配置（如果某个关节方向相反，设置为-1）
        joint_direction_correction = {
            'shoulder_pan': 1,    # 1表示正常方向，-1表示反向
            'shoulder_lift': -1,
            'elbow_flex': 1,
            'wrist_flex': 1,
            'wrist_roll': -1,
            'gripper': 1
        }
        
        # 关节零位偏移校正（弧度）
        # 正值表示仿真关节需要向正方向偏移，负值表示向负方向偏移
        joint_zero_offset = {
            'shoulder_pan': 0.0,      # 肩部旋转
            'shoulder_lift': 1.57,   # 肩部抬升：偏移-90度（-π/2）
            'elbow_flex': 1.57,      # 肘部弯曲：偏移-90度（-π/2）
            'wrist_flex': 0.0,        # 腕部弯曲
            'wrist_roll': 0.0,        # 腕部旋转
            'gripper': 0.0            # 夹爪
        }
        
        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
                
                # 将电机范围映射到关节范围（角度）
                processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                    * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                processed_radius = processed_degree / 180.0 * np.pi  # 转换为弧度

                if self._lerobot_compat:
                    # lerobot兼容 + xlerobot几何对齐：保留必要方向/零位修正，和历史控制行为对齐。
                    direction_correction = joint_direction_correction.get(joint_name, 1)
                    zero_offset = joint_zero_offset.get(joint_name, 0.0)
                    processed_action[index] = processed_radius * direction_correction + zero_offset
                else:
                    # 应用方向校正
                    direction_correction = joint_direction_correction.get(joint_name, 1)
                    processed_radius = processed_radius * direction_correction

                    # 应用零位偏移校正
                    zero_offset = joint_zero_offset.get(joint_name, 0.0)
                    processed_action[index] = processed_radius + zero_offset

                # 第二关节量程补偿：以零位为中心做小幅拉伸，便于收回。
                if joint_name == "shoulder_lift":
                    center = joint_zero_offset.get("shoulder_lift", 1.57)
                    processed_action[index] = (processed_action[index] - center) * self._shoulder_lift_gain + center

        # 与xlerobot真实关节限位对齐，防止映射越界。
        arm_joint_order = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        lower = np.array([XLEROBOT_JOINT_LIMITS[name][0] for name in arm_joint_order], dtype=np.float64)
        upper = np.array([XLEROBOT_JOINT_LIMITS[name][1] for name in arm_joint_order], dtype=np.float64)
        processed_action = np.clip(processed_action, lower, upper)
        return processed_action

    def _smooth_arm_action(self, target_action: np.ndarray, arm_side: str) -> np.ndarray:
        """对绝对目标做死区+低通+步长限制，降低leader抖动。"""
        arm_joint_order = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        lower = np.array([XLEROBOT_JOINT_LIMITS[name][0] for name in arm_joint_order], dtype=np.float64)
        upper = np.array([XLEROBOT_JOINT_LIMITS[name][1] for name in arm_joint_order], dtype=np.float64)

        filtered_prev = self._left_arm_filtered if arm_side == "left" else self._right_arm_filtered
        target = np.clip(np.asarray(target_action, dtype=np.float64), lower, upper)

        if filtered_prev is None:
            filtered = target.copy()
        else:
            delta = target - filtered_prev
            if self._arm_deadband > 0.0:
                delta[np.abs(delta) < self._arm_deadband] = 0.0
            target_after_deadband = filtered_prev + delta

            smoothed = filtered_prev + self._arm_smooth_alpha * (target_after_deadband - filtered_prev)
            if self._arm_max_step > 0.0:
                per_joint_max_step = np.full_like(smoothed, self._arm_max_step, dtype=np.float64)
                per_joint_max_step[5] = self._jaw_max_step
                step = np.clip(smoothed - filtered_prev, -per_joint_max_step, per_joint_max_step)
                filtered = filtered_prev + step
            else:
                filtered = smoothed

        filtered = np.clip(filtered, lower, upper)
        if arm_side == "left":
            self._left_arm_filtered = filtered
        else:
            self._right_arm_filtered = filtered
        return filtered.astype(np.float32)
