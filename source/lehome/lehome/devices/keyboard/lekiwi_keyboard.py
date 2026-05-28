import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class LekiwiKeyboard(Device):
    """Lekiwi专用键盘控制器，支持全向轮移动底盘和单臂控制"""

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        self.sensitivity = sensitivity

        # 获取Omniverse接口
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # 键盘事件订阅
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self):
                obj._on_keyboard_event(event, *args),
        )

        # 创建键盘绑定
        self._create_key_bindings()

        # 命令缓冲区
        self._omnimove_command = None       # 全向移动指令
        self._arm_delta = np.zeros(5)       # 5个机械臂关节
        self._gripper_delta = 0.0           # 夹爪

        # 标志和回调
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # 按键状态跟踪
        self._pressed_keys = set()

    def _create_key_bindings(self):
        """创建键盘绑定"""
        self._key_bindings = {
            # 底盘全向轮控制（每个按键控制多个轮子的组合）
            "W": ("omnimove", "forward", 1.0),      # 前进：轮2+ 轮3+
            "S": ("omnimove", "backward", 1.0),     # 后退：轮2- 轮3-
            "A": ("omnimove", "left", 1.0),         # 左平移：轮1+ 轮3-
            "D": ("omnimove", "right", 1.0),        # 右平移：轮1- 轮2-
            "Q": ("omnimove", "rotate_left", 1.0),  # 原地左转：轮1- 轮2+ 轮3-
            "E": ("omnimove", "rotate_right", 1.0),  # 原地右转：轮1+ 轮2- 轮3+

            # 机械臂控制
            "R": ("arm", 0, 1.0),      # shoulder_pitch +
            "F": ("arm", 0, -1.0),     # shoulder_pitch -
            "T": ("arm", 1, 1.0),      # shoulder_roll +
            "G": ("arm", 1, -1.0),     # shoulder_roll -
            "Y": ("arm", 2, 1.0),      # elbow +
            "H": ("arm", 2, -1.0),     # elbow -
            "U": ("arm", 3, 1.0),      # wrist_pitch +
            "J": ("arm", 3, -1.0),     # wrist_pitch -
            "I": ("arm", 4, 1.0),      # wrist_roll +
            "K": ("arm", 4, -1.0),     # wrist_roll -

            # 夹爪控制
            "O": ("gripper", 0, 1.0),  # 夹爪开
            "P": ("gripper", 0, -1.0),  # 夹爪关

            # 控制键
            "B": ("control", "start", 1.0),   # 启动控制
            "N": ("control", "success", 1.0),  # 任务成功
            "F5": ("control", "reset", 1.0),  # 重置环境
            "F6": ("control", "mode_switch", 1.0),  # 切换混合控制模式
        }

    def get_device_state(self):
        """获取设备状态"""
        action = np.zeros(9)

        # 全向轮控制 - 根据移动指令设置3个轮子的速度
        if self._omnimove_command == "forward":
            # Q+A 组合：轮2正转 + 轮3正转
            action[1] = 1.0 * self.sensitivity
            action[2] = 1.0 * self.sensitivity
        elif self._omnimove_command == "backward":
            # E+D 组合：轮2反转 + 轮3反转
            action[1] = -1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "left":
            # W+E 组合：轮1正转 + 轮3反转
            action[0] = 1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "right":
            # S+D 组合：轮1反转 + 轮2反转
            action[0] = -1.0 * self.sensitivity
            action[1] = -1.0 * self.sensitivity
        elif self._omnimove_command == "rotate_left":
            # A+S+E 组合：轮1反转 + 轮2正转 + 轮3反转
            action[0] = -1.0 * self.sensitivity
            action[1] = 1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "rotate_right":
            # Q+W+D 组合：轮1正转 + 轮2反转 + 轮3正转
            action[0] = 1.0 * self.sensitivity
            action[1] = -1.0 * self.sensitivity
            action[2] = 1.0 * self.sensitivity

        # 机械臂控制 (3-7)
        action[3] = self._arm_delta[0] * self.sensitivity  # shoulder_pitch
        action[4] = self._arm_delta[1] * self.sensitivity  # shoulder_roll
        action[5] = self._arm_delta[2] * self.sensitivity  # elbow
        action[6] = self._arm_delta[3] * self.sensitivity  # wrist_pitch
        action[7] = self._arm_delta[4] * self.sensitivity  # wrist_roll

        # 夹爪控制 (8)
        action[8] = self._gripper_delta * self.sensitivity

        return action

    def input2action(self):
        """将输入转换为动作"""
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['lekiwi_keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def advance(self):
        """获取当前动作 - 兼容Device接口"""
        if not self.started:
            return None

        # 返回正确的设备上的张量
        action = self.get_device_state()
        return torch.tensor(
            action, dtype=torch.float32, device=self.env.device
        )

    def reset(self):
        """重置控制器状态"""
        self._omnimove_command = None
        self._arm_delta.fill(0)
        self._gripper_delta = 0.0
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """添加回调函数"""
        self._additional_callbacks[key] = func

    def __del__(self):
        """释放键盘接口"""
        if (hasattr(self, '_input') and hasattr(self, '_keyboard') and
                hasattr(self, '_keyboard_sub')):
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """处理键盘事件"""
        try:
            # 获取按键名称
            if hasattr(event, 'input') and hasattr(event.input, 'name'):
                key_name = event.input.name
            elif hasattr(event, 'name'):
                key_name = event.name
            else:
                return

            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]

                    # 处理控制键
                    if control_type == "control":
                        if index == "start":
                            self.started = True
                            self._reset_state = False
                            print("Lekiwi控制已启动！")
                        elif index == "success":
                            self.started = False
                            self._reset_state = True
                            if "N" in self._additional_callbacks:
                                self._additional_callbacks["N"]()
                        elif index == "reset":
                            self._reset_state = True
                            self.started = False
                            self.reset()
                            if "R" in self._additional_callbacks:
                                self._additional_callbacks["R"]()
                        elif index == "mode_switch":
                            if "F6" in self._additional_callbacks:
                                self._additional_callbacks["F6"]()
                        return

                    # 处理移动和机械臂控制键
                    if control_type == "omnimove":
                        self._omnimove_command = index  # "forward", "backward"
                    elif control_type == "arm":
                        self._arm_delta[index] = value
                    elif control_type == "gripper":
                        self._gripper_delta = value

                    self._pressed_keys.add(key_name)

            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]

                    # 只处理机器人控制键的释放，不处理控制键
                    if control_type == "omnimove":
                        if self._omnimove_command == index:
                            self._omnimove_command = None
                    elif control_type == "arm":
                        self._arm_delta[index] = 0.0
                    elif control_type == "gripper":
                        self._gripper_delta = 0.0

                    self._pressed_keys.discard(key_name)

        except Exception as e:
            print(f"键盘事件处理错误: {e}")

    def __str__(self) -> str:
        """返回控制器信息"""
        msg = "Lekiwi键盘控制器\n"
        msg += (
            f"\t键盘名称: {self._input.get_keyboard_name(self._keyboard)}\n"
        )
        msg += f"\t灵敏度: {self.sensitivity}\n"
        msg += f"\t控制状态: {'已启动' if self.started else '未启动'}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t底盘全向轮控制:\n"
        msg += "\t  前进: W (轮2+ 轮3+)\n"
        msg += "\t  后退: S (轮2- 轮3-)\n"
        msg += "\t  左平移: A (轮1+ 轮3-)\n"
        msg += "\t  右平移: D (轮1- 轮2-)\n"
        msg += "\t  原地左转: Q (轮1- 轮2+ 轮3-)\n"
        msg += "\t  原地右转: E (轮1+ 轮2- 轮3+)\n"
        msg += "\t机械臂控制:\n"
        msg += "\t  肩部俯仰: R/F\n"
        msg += "\t  肩部侧摆: T/G\n"
        msg += "\t  肘关节: Y/H\n"
        msg += "\t  腕部俯仰: U/J\n"
        msg += "\t  腕部旋转: I/K\n"
        msg += "\t夹爪控制:\n"
        msg += "\t  夹爪: O/P (开/关)\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t启动控制: B\n"
        msg += "\t任务成功: N\n"
        msg += "\t重置环境: F5\n"
        msg += "\t退出: Ctrl+C\n"
        msg += f"\t当前按下的键: {list(self._pressed_keys)}"
        return msg
