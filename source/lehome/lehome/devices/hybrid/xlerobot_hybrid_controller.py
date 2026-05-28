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
    """Hybrid controller: keyboard base/head control plus dual SO101 leaders."""
    
    def __init__(self, env, sensitivity: float = 1.0, 
                 left_arm_port: str = '/dev/ttyACM0', 
                 right_arm_port: str = '/dev/ttyACM1', 
                 recalibrate: bool = False):
        super().__init__(env)
        
        # Keyboard controller for base and head control.
        self.keyboard_controller = XlerobotKeyboard(env, sensitivity)
        
        # Dual-arm leader controller.
        self.bi_arm_controller = BiXlerobotLeader(
            env, 
            left_port=left_arm_port, 
            right_port=right_arm_port, 
            recalibrate=recalibrate
        )
        
        # Control mode state.
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arms_only"
        
        # Runtime state.
        self.started = False
        self._reset_state = False
        self._lerobot_compat = os.getenv("LEHOME_XLEROBOT_LEROBOT_COMPAT", "0") == "1"
        if self._lerobot_compat:
            print("XlerobotHybridController: lerobot-compatible arm mapping enabled.")

        # Arm smoothing parameters for absolute leader control.
        self._arm_smooth_alpha = float(os.getenv("LEHOME_XLEROBOT_ARM_SMOOTH_ALPHA", "0.25"))
        self._arm_smooth_alpha = float(np.clip(self._arm_smooth_alpha, 0.0, 1.0))
        self._arm_deadband = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_DEADBAND", "0.01")))
        self._arm_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_ARM_MAX_STEP", "0.05")))
        self._jaw_max_step = max(0.0, float(os.getenv("LEHOME_XLEROBOT_JAW_MAX_STEP", "0.18")))

        # Mild shoulder-lift gain to improve reachable return motion.
        self._shoulder_lift_gain = float(os.getenv("LEHOME_XLEROBOT_SHOULDER_LIFT_GAIN", "1.15"))
        self._shoulder_lift_gain = max(1.0, self._shoulder_lift_gain)

        self._left_arm_filtered = None
        self._right_arm_filtered = None
        self._latest_state_vector = None
        self._cached_arms_action = None

        # Enable F6 mode switching by default.
        self.add_callback("F6", lambda: None)
        
    def set_control_mode(self, mode: str):
        """Set the active control mode."""
        assert mode in ["keyboard", "hybrid", "arms_only"], f"Invalid control mode: {mode}"
        if mode != self.control_mode:
            # Clear filters when switching modes to avoid stale targets.
            self._left_arm_filtered = None
            self._right_arm_filtered = None
            self._latest_state_vector = None
            self._cached_arms_action = None
            # Clear base commands when entering arms-only mode.
            if mode == "arms_only":
                self._clear_base_command()
        self.control_mode = mode
        print(f"[XlerobotHybridController] Control mode set to: {mode}")

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
        """Return the current device state."""
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
        """Convert device input into an action dictionary."""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            self._latest_state_vector = action.get("joint_state")
            self._cached_arms_action = None
            return action
        elif self.control_mode == "arms_only":
            action = self.bi_arm_controller.input2action()
            self.started = action.get("started", False)
            # Use one serial read per frame and reuse the cached result.
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
            
            # Hybrid mode may be started from either keyboard or arm leaders.
            self.started = (
                keyboard_action.get("started", False)
                or arms_action.get("started", False)
            )
            
            # Update cached serial state.
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
        """Return the current action tensor."""
        if not self.started:
            return None

        # Prefer input2action() cache to avoid duplicate serial reads.
        action = self._latest_state_vector if self._latest_state_vector is not None else self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)
    
    def reset(self):
        """Reset controller state."""
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
        """Register a controller callback."""
        # Keyboard callbacks handle mode switching.
        if key == "F6":
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arms_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
        else:
            # Other callbacks go to the keyboard controller.
            self.keyboard_controller.add_callback(key, func)
            
            # Forward B so BiSO101Leader can start arm control.
            if key == "B":
                self.bi_arm_controller.add_callback(key, func)
    
    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Xlerobot Hybrid Controller\n"
        msg += f"\tControl mode: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeyboard: base (W/A/S/D/Q/E) and head (Home/End/PageUp/PageDown)\n"
        msg += "\tDual leaders: arms and grippers\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: switch mode (hybrid <-> keyboard <-> arms_only)\n"
        msg += "\tB: start control\n"
        msg += "\tF5: reset environment\n"
        msg += "\tExit: Ctrl+C\n"
        return msg

    def _convert_arm_action(self, joint_state: dict, motor_limits: dict) -> np.ndarray:
        """Convert one SO101 leader arm into the Xlerobot arm layout."""
        processed_action = np.zeros(6)
        
        if not motor_limits:
            print("[XlerobotHybridController] Missing motor limits; using zero arm action.")
            return processed_action
        
        # SO101 follower joint limits are expressed in degrees.
        from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
        
        # SO101 joint name to Xlerobot arm action index.
        joint_mapping = {
            'shoulder_pan': 0,
            'shoulder_lift': 1,
            'elbow_flex': 2,
            'wrist_flex': 3,
            'wrist_roll': 4,
            'gripper': 5
        }
        
        # Direction correction for SO101-to-Xlerobot geometry.
        joint_direction_correction = {
            'shoulder_pan': 1,
            'shoulder_lift': -1,
            'elbow_flex': 1,
            'wrist_flex': 1,
            'wrist_roll': -1,
            'gripper': 1
        }
        
        # Zero offsets in radians.
        joint_zero_offset = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 1.57,
            'elbow_flex': 1.57,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        
        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
                
                # Map motor range to SO101 joint range in degrees.
                processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                    * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                processed_radius = processed_degree / 180.0 * np.pi

                if self._lerobot_compat:
                    # Preserve the historical lerobot-compatible alignment.
                    direction_correction = joint_direction_correction.get(joint_name, 1)
                    zero_offset = joint_zero_offset.get(joint_name, 0.0)
                    processed_action[index] = processed_radius * direction_correction + zero_offset
                else:
                    # Apply direction correction.
                    direction_correction = joint_direction_correction.get(joint_name, 1)
                    processed_radius = processed_radius * direction_correction

                    # Apply zero offset correction.
                    zero_offset = joint_zero_offset.get(joint_name, 0.0)
                    processed_action[index] = processed_radius + zero_offset

                if joint_name == "shoulder_lift":
                    center = joint_zero_offset.get("shoulder_lift", 1.57)
                    processed_action[index] = (processed_action[index] - center) * self._shoulder_lift_gain + center

        # Clamp to the Xlerobot arm/gripper limits.
        arm_joint_order = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        lower = np.array([XLEROBOT_JOINT_LIMITS[name][0] for name in arm_joint_order], dtype=np.float64)
        upper = np.array([XLEROBOT_JOINT_LIMITS[name][1] for name in arm_joint_order], dtype=np.float64)
        processed_action = np.clip(processed_action, lower, upper)
        return processed_action

    def _smooth_arm_action(self, target_action: np.ndarray, arm_side: str) -> np.ndarray:
        """Apply deadband, low-pass filtering, and per-step limits to arm targets."""
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
