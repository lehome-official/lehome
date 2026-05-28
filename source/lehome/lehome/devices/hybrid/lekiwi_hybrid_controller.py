import numpy as np
import torch
from collections.abc import Callable

from ..device_base import Device
from ..lerobot import SO101Leader
from ..keyboard.lekiwi_keyboard import LekiwiKeyboard


class LekiwiHybridController(Device):
    """Lekiwi混合控制器：键盘控制底盘，SO101Leader控制机械臂"""

    def __init__(self, env, sensitivity: float = 1.0,
                 arm_port: str = '/dev/ttyACM0',
                 recalibrate: bool = False):
        super().__init__(env)

        # 创建键盘控制器（用于底盘控制）
        self.keyboard_controller = LekiwiKeyboard(env, sensitivity)

        # 创建机械臂控制器（使用独立的校准文件）
        self.arm_controller = SO101Leader(
            env,
            port=arm_port,
            recalibrate=recalibrate,
            calibration_file_name="lekiwi_so101_leader.json"
        )

        # 控制模式标志
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arm_only"

        # 状态标志
        self.started = False
        self._reset_state = False

        # 默认启用 F6 模式切换快捷键
        self.add_callback("F6", lambda: None)

    def set_control_mode(self, mode: str):
        """设置控制模式"""
        assert mode in ["keyboard", "hybrid", "arm_only"], (
            f"Invalid control mode: {mode}"
        )
        self.control_mode = mode
        print(f"控制模式切换为: {mode}")

    def get_device_state(self):
        """获取设备状态"""
        if self.control_mode == "keyboard":
            state = self.keyboard_controller.get_device_state()
            return state
        elif self.control_mode == "arm_only":
            arm_state = self.arm_controller.get_device_state()
            # 缓存arm_action，避免重复调用
            if not hasattr(self, '_cached_arm_action'):
                self._cached_arm_action = self.arm_controller.input2action()

            full_state = np.zeros(9)

            # 机械臂控制（索引3-8）
            if arm_state is not None:
                if isinstance(arm_state, dict):
                    motor_limits = self._cached_arm_action.get(
                        'motor_limits', {}
                    )
                    arm_processed = self._convert_arm_action(
                        arm_state, motor_limits
                    )
                    full_state[3:9] = arm_processed

            return full_state
        else:  # hybrid mode
            keyboard_state = self.keyboard_controller.get_device_state()
            arm_state = self.arm_controller.get_device_state()

            # 缓存arm_action，避免重复调用
            if not hasattr(self, '_cached_arm_action'):
                self._cached_arm_action = self.arm_controller.input2action()

            hybrid_state = np.zeros(9)

            # 底盘控制（索引0-2）：使用键盘控制
            hybrid_state[0:3] = keyboard_state[0:3]

            # 机械臂控制（索引3-8）：使用SO101Leader控制器
            if arm_state is not None:
                if isinstance(arm_state, dict):
                    motor_limits = self._cached_arm_action.get(
                        'motor_limits', {}
                    )
                    arm_processed = self._convert_arm_action(
                        arm_state, motor_limits
                    )
                    hybrid_state[3:9] = arm_processed

            return hybrid_state

    def input2action(self):
        """将输入转换为动作"""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            return action
        elif self.control_mode == "arm_only":
            action = self.arm_controller.input2action()
            self.started = action.get("started", False)
            # 更新缓存
            self._cached_arm_action = action
            return action
        else:  # hybrid mode
            keyboard_action = self.keyboard_controller.input2action()
            arm_action = self.arm_controller.input2action()

            # 设置started状态
            self.started = arm_action.get("started", False)

            # 更新缓存
            self._cached_arm_action = arm_action

            hybrid_action = {
                "reset": (keyboard_action.get("reset", False) or
                          arm_action.get("reset", False)),
                "started": arm_action.get("started", False),
                "lekiwi_hybrid": True,
                "keyboard": True,
                "so101_leader": True,
                "joint_state": self.get_device_state(),
                "motor_limits": arm_action.get("motor_limits", {}),
            }

            return hybrid_action

    def advance(self):
        """获取当前动作"""
        if not self.started:
            return None

        action = self.get_device_state()
        return torch.tensor(
            action, dtype=torch.float32, device=self.env.device
        )

    def reset(self):
        """重置控制器状态"""
        self.keyboard_controller.reset()
        self.arm_controller.reset()
        self.started = False
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """添加回调函数"""
        # 键盘回调用于控制模式切换
        if key == "F6":  # F6键切换控制模式
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arm_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
        elif key == "R":
            # Reset键需要传递给两个控制器
            self.keyboard_controller.add_callback(key, func)
            self.arm_controller.add_callback("R", func)
            # SO101Leader还期望这些键有回调，添加空回调避免KeyError
            self.arm_controller.add_callback("S", lambda: None)
            self.arm_controller.add_callback("N", func)
            self.arm_controller.add_callback("D", lambda: None)
        else:
            # 其他回调传递给键盘控制器
            self.keyboard_controller.add_callback(key, func)

            # 对于B键，也需要传递给SO101Leader以启动机械臂控制
            if key == "B":
                # 直接传递B键回调给SO101Leader
                self.arm_controller.add_callback(key, func)

    def __str__(self) -> str:
        """返回控制器信息"""
        msg = "Lekiwi混合控制器\n"
        msg += f"\t当前控制模式: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t键盘控制: 底盘移动(W/A/S/D/Q/E)\n"
        msg += "\tSO101Leader控制器: 机械臂和夹爪控制\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: 切换控制模式 (hybrid <-> keyboard <-> arm_only)\n"
        msg += "\tB: 启动控制\n"
        msg += "\tF5: 重置环境\n"
        msg += "\t退出: Ctrl+C\n"
        return msg

    def _convert_arm_action(
        self, joint_state: dict, motor_limits: dict
    ) -> np.ndarray:
        """转换SO101Leader动作到lekiwi关节空间"""
        processed_action = np.zeros(6)

        # 如果没有motor_limits，返回零动作
        if not motor_limits:
            print("警告：没有找到motor_limits，返回零动作")
            return processed_action

        # 使用SO101的关节限制（角度）
        from lehome.assets.robots.lerobot import (
            SO101_FOLLOWER_USD_JOINT_LIMLITS
        )

        # 关节名称到索引的映射
        joint_mapping = {
            'shoulder_pan': 0,      # 对应lekiwi的shoulder_pitch
            'shoulder_lift': 1,     # 对应lekiwi的shoulder_roll
            'elbow_flex': 2,        # 对应lekiwi的elbow
            'wrist_flex': 3,        # 对应lekiwi的wrist_pitch
            'wrist_roll': 4,        # 对应lekiwi的wrist_roll
            'gripper': 5            # 对应lekiwi的gripper
        }

        # 关节方向校正配置
        joint_direction_correction = {
            'shoulder_pan': 1,      # 1表示正常方向，-1表示反向
            'shoulder_lift': -1,
            'elbow_flex': -1,
            'wrist_flex': -1,
            'wrist_roll': -1,
            'gripper': -1
        }

        # 关节零位偏移校正（弧度）
        joint_zero_offset = {
            'shoulder_pan': 0.0,
            'shoulder_lift': -1.57,   # 偏移-90度（-π/2）
            'elbow_flex': 1.57,      # 偏移-90度（-π/2）
            'wrist_flex': 1.57,
            'wrist_roll': 1.57,
            'gripper': 0.0
        }

        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = (
                    SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
                )

                # 将电机范围映射到关节范围（角度）
                processed_degree = (
                    (joint_state[joint_name] - motor_limit_range[0]) /
                    (motor_limit_range[1] - motor_limit_range[0]) *
                    (joint_limit_range[1] - joint_limit_range[0]) +
                    joint_limit_range[0]
                )
                processed_radius = processed_degree / 180.0 * np.pi

                # 应用方向校正
                direction_correction = joint_direction_correction.get(
                    joint_name, 1
                )
                processed_radius = processed_radius * direction_correction

                # 应用零位偏移校正
                zero_offset = joint_zero_offset.get(joint_name, 0.0)
                processed_action[index] = processed_radius + zero_offset

        return processed_action
