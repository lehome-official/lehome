import numpy as np
import torch
from collections.abc import Callable

from ..device_base import Device
from ..lerobot import SO101Leader
from ..keyboard.lekiwi_keyboard import LekiwiKeyboard


class LekiwiHybridController(Device):
    """Hybrid controller: keyboard base control plus SO101Leader arm control."""

    def __init__(self, env, sensitivity: float = 1.0,
                 arm_port: str = '/dev/ttyACM0',
                 recalibrate: bool = False):
        super().__init__(env)

        # Keyboard controller for the mobile base.
        self.keyboard_controller = LekiwiKeyboard(env, sensitivity)

        # Arm controller with a dedicated calibration file.
        self.arm_controller = SO101Leader(
            env,
            port=arm_port,
            recalibrate=recalibrate,
            calibration_file_name="lekiwi_so101_leader.json"
        )

        # Control mode state.
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arm_only"

        # Runtime state.
        self.started = False
        self._reset_state = False

        # Enable F6 mode switching by default.
        self.add_callback("F6", lambda: None)

    def set_control_mode(self, mode: str):
        """Set the active control mode."""
        assert mode in ["keyboard", "hybrid", "arm_only"], (
            f"Invalid control mode: {mode}"
        )
        self.control_mode = mode
        print(f"[LekiwiHybridController] Control mode set to: {mode}")

    def get_device_state(self):
        """Return the current device state."""
        if self.control_mode == "keyboard":
            state = self.keyboard_controller.get_device_state()
            return state
        elif self.control_mode == "arm_only":
            arm_state = self.arm_controller.get_device_state()
            # Cache arm action to avoid repeated serial reads in one frame.
            if not hasattr(self, '_cached_arm_action'):
                self._cached_arm_action = self.arm_controller.input2action()

            full_state = np.zeros(9)

            # Arm control (indices 3-8).
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

            # Cache arm action to avoid repeated serial reads in one frame.
            if not hasattr(self, '_cached_arm_action'):
                self._cached_arm_action = self.arm_controller.input2action()

            hybrid_state = np.zeros(9)

            # Base control (indices 0-2) from keyboard.
            hybrid_state[0:3] = keyboard_state[0:3]

            # Arm and gripper control (indices 3-8) from SO101Leader.
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
        """Convert device input into an action dictionary."""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            return action
        elif self.control_mode == "arm_only":
            action = self.arm_controller.input2action()
            self.started = action.get("started", False)
            # Update cached serial state.
            self._cached_arm_action = action
            return action
        else:  # hybrid mode
            keyboard_action = self.keyboard_controller.input2action()
            arm_action = self.arm_controller.input2action()

            # In hybrid mode, arm leader start state controls execution.
            self.started = arm_action.get("started", False)

            # Update cached serial state.
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
        """Return the current action tensor."""
        if not self.started:
            return None

        action = self.get_device_state()
        return torch.tensor(
            action, dtype=torch.float32, device=self.env.device
        )

    def reset(self):
        """Reset controller state."""
        self.keyboard_controller.reset()
        self.arm_controller.reset()
        self.started = False
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Register a controller callback."""
        # Keyboard callbacks handle mode switching.
        if key == "F6":
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arm_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
        elif key == "R":
            # Reset must be forwarded to both controllers.
            self.keyboard_controller.add_callback(key, func)
            self.arm_controller.add_callback("R", func)
            # SO101Leader expects these calibration callbacks to exist.
            self.arm_controller.add_callback("S", lambda: None)
            self.arm_controller.add_callback("N", func)
            self.arm_controller.add_callback("D", lambda: None)
        else:
            # Other callbacks go to the keyboard controller.
            self.keyboard_controller.add_callback(key, func)

            # Forward B so the SO101Leader can start arm control.
            if key == "B":
                self.arm_controller.add_callback(key, func)

    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Lekiwi Hybrid Controller\n"
        msg += f"\tControl mode: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeyboard: mobile base (W/A/S/D/Q/E)\n"
        msg += "\tSO101Leader: arm and gripper\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: switch mode (hybrid <-> keyboard <-> arm_only)\n"
        msg += "\tB: start control\n"
        msg += "\tF5: reset environment\n"
        msg += "\tExit: Ctrl+C\n"
        return msg

    def _convert_arm_action(
        self, joint_state: dict, motor_limits: dict
    ) -> np.ndarray:
        """Convert SO101Leader actions into the Lekiwi joint layout."""
        processed_action = np.zeros(6)

        if not motor_limits:
            print("[LekiwiHybridController] Missing motor limits; using zero arm action.")
            return processed_action

        # SO101 follower joint limits are expressed in degrees.
        from lehome.assets.robots.lerobot import (
            SO101_FOLLOWER_USD_JOINT_LIMLITS
        )

        # SO101 joint name to Lekiwi action index.
        joint_mapping = {
            'shoulder_pan': 0,
            'shoulder_lift': 1,
            'elbow_flex': 2,
            'wrist_flex': 3,
            'wrist_roll': 4,
            'gripper': 5
        }

        # Direction correction for SO101-to-Lekiwi geometry.
        joint_direction_correction = {
            'shoulder_pan': 1,
            'shoulder_lift': -1,
            'elbow_flex': -1,
            'wrist_flex': -1,
            'wrist_roll': -1,
            'gripper': -1
        }

        # Zero offsets in radians.
        joint_zero_offset = {
            'shoulder_pan': 0.0,
            'shoulder_lift': -1.57,
            'elbow_flex': 1.57,
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

                # Map motor range to SO101 joint range in degrees.
                processed_degree = (
                    (joint_state[joint_name] - motor_limit_range[0]) /
                    (motor_limit_range[1] - motor_limit_range[0]) *
                    (joint_limit_range[1] - joint_limit_range[0]) +
                    joint_limit_range[0]
                )
                processed_radius = processed_degree / 180.0 * np.pi

                # Apply direction correction.
                direction_correction = joint_direction_correction.get(
                    joint_name, 1
                )
                processed_radius = processed_radius * direction_correction

                # Apply zero offset correction.
                zero_offset = joint_zero_offset.get(joint_name, 0.0)
                processed_action[index] = processed_radius + zero_offset

        return processed_action
