import weakref
import os
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class XlerobotKeyboard(Device):
    """Keyboard controller for Xlerobot base, arms, grippers, and head."""

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        self.sensitivity = sensitivity

        # Omniverse input handles.
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Keyboard event subscription.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Key bindings.
        self._create_key_bindings()

        # Command buffers.
        self._position_delta = np.zeros(2, dtype=np.float32)  # x, y (body frame)
        self._rotation_delta = 0.0  # yaw (body frame)
        self._left_arm_delta = np.zeros(5, dtype=np.float32)
        self._right_arm_delta = np.zeros(5, dtype=np.float32)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta = np.zeros(2, dtype=np.float32)

        # State flags and callbacks.
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # Pressed-key tracking.
        self._pressed_keys = set()
        self._last_polled_reset_state = False
        self._last_polled_mode_switch_state = False
        self._debug_base = os.getenv("LEHOME_DEBUG_XLEROBOT_BASE", "0") == "1"

    def _create_key_bindings(self):
        """Create keyboard bindings."""
        self._key_bindings = {
            # Base commands in body frame.
            "W": ("position", 0, 1.0),
            "S": ("position", 0, -1.0),
            "A": ("position", 1, 1.0),
            "D": ("position", 1, -1.0),
            "Q": ("rotation_joint", 0, 1.0),
            "E": ("rotation_joint", 0, -1.0),

            # Left arm control.
            "R": ("left_arm", 0, 1.0),
            "F": ("left_arm", 0, -1.0),
            "T": ("left_arm", 1, 1.0),
            "G": ("left_arm", 1, -1.0),
            "Y": ("left_arm", 2, 1.0),
            "H": ("left_arm", 2, -1.0),
            "U": ("left_arm", 3, 1.0),
            "J": ("left_arm", 3, -1.0),
            "I": ("left_arm", 4, 1.0),
            "K": ("left_arm", 4, -1.0),

            # Right arm control.
            "NUMPAD_8": ("right_arm", 0, 1.0),
            "NUMPAD_2": ("right_arm", 0, -1.0),
            "NUMPAD_4": ("right_arm", 1, 1.0),
            "NUMPAD_6": ("right_arm", 1, -1.0),
            "NUMPAD_7": ("right_arm", 2, 1.0),
            "NUMPAD_9": ("right_arm", 2, -1.0),
            "NUMPAD_1": ("right_arm", 3, 1.0),
            "NUMPAD_3": ("right_arm", 3, -1.0),
            "NUMPAD_0": ("right_arm", 4, 1.0),
            "NUMPAD_PERIOD": ("right_arm", 4, -1.0),

            # Gripper control.
            "F1": ("left_gripper", 0, 1.0),
            "F2": ("left_gripper", 0, -1.0),
            "F3": ("right_gripper", 0, 1.0),
            "F4": ("right_gripper", 0, -1.0),

            # Head control.
            "HOME": ("head", 0, 1.0),
            "END": ("head", 0, -1.0),
            "PAGE_UP": ("head", 1, 1.0),
            "PAGE_DOWN": ("head", 1, -1.0),

            # Control keys.
            "B": ("control", "start", 1.0),
            "N": ("control", "success", 1.0),
            "F5": ("control", "reset", 1.0),
            "F6": ("control", "mode_switch", 1.0),
        }

    def _publish_base_command(self):
        """Publish base command to the env as (vx_body, vy_body, wz) in [-1, 1]."""
        if not hasattr(self.env, "set_base_command"):
            return
        base_cmd = np.array(
            [
                self._position_delta[0],
                self._position_delta[1],
                self._rotation_delta,
            ],
            dtype=np.float32,
        )
        base_cmd = np.clip(base_cmd, -1.0, 1.0)
        if self._debug_base and np.any(np.abs(base_cmd) > 1e-6):
            print(f"[XlerobotKeyboard] base_cmd={base_cmd.tolist()}")
        self.env.set_base_command(base_cmd)

    def _is_key_down(self, key_name: str) -> bool:
        """Query key state via polling as fallback when event callbacks are unreliable."""
        try:
            key_enum = getattr(carb.input.KeyboardInput, key_name)
            flags = self._input.get_keyboard_button_flags(self._keyboard, key_enum)
            return (flags & carb.input.BUTTON_FLAG_DOWN) != 0
        except Exception:
            return False

    def _poll_keyboard_state(self):
        """Poll key states each frame to keep controls responsive even if events are missed."""
        # Base controls
        w_down = self._is_key_down("W")
        s_down = self._is_key_down("S")
        a_down = self._is_key_down("A")
        d_down = self._is_key_down("D")
        q_down = self._is_key_down("Q")
        e_down = self._is_key_down("E")
        self._position_delta[0] = float(w_down) - float(s_down)
        self._position_delta[1] = float(a_down) - float(d_down)
        self._rotation_delta = float(q_down) - float(e_down)

        # Left arm
        self._left_arm_delta[0] = float(self._is_key_down("R")) - float(self._is_key_down("F"))
        self._left_arm_delta[1] = float(self._is_key_down("T")) - float(self._is_key_down("G"))
        self._left_arm_delta[2] = float(self._is_key_down("Y")) - float(self._is_key_down("H"))
        self._left_arm_delta[3] = float(self._is_key_down("U")) - float(self._is_key_down("J"))
        self._left_arm_delta[4] = float(self._is_key_down("I")) - float(self._is_key_down("K"))

        # Right arm
        self._right_arm_delta[0] = float(self._is_key_down("NUMPAD_8")) - float(self._is_key_down("NUMPAD_2"))
        self._right_arm_delta[1] = float(self._is_key_down("NUMPAD_4")) - float(self._is_key_down("NUMPAD_6"))
        self._right_arm_delta[2] = float(self._is_key_down("NUMPAD_7")) - float(self._is_key_down("NUMPAD_9"))
        self._right_arm_delta[3] = float(self._is_key_down("NUMPAD_1")) - float(self._is_key_down("NUMPAD_3"))
        self._right_arm_delta[4] = float(self._is_key_down("NUMPAD_0")) - float(self._is_key_down("NUMPAD_PERIOD"))

        # Grippers
        self._left_gripper_delta = float(self._is_key_down("F1")) - float(self._is_key_down("F2"))
        self._right_gripper_delta = float(self._is_key_down("F3")) - float(self._is_key_down("F4"))

        # Head
        self._head_delta[0] = float(self._is_key_down("HOME")) - float(self._is_key_down("END"))
        self._head_delta[1] = float(self._is_key_down("PAGE_UP")) - float(self._is_key_down("PAGE_DOWN"))

        # Start control (level-triggered)
        if self._is_key_down("B"):
            self.started = True
            self._reset_state = False

        # Reset / mode switch (edge-triggered to avoid repeated firing)
        reset_down = self._is_key_down("F5")
        if reset_down and not self._last_polled_reset_state:
            self._reset_state = True
            self.started = False
            self.reset()
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
        self._last_polled_reset_state = reset_down

        mode_down = self._is_key_down("F6")
        if mode_down and not self._last_polled_mode_switch_state:
            if "F6" in self._additional_callbacks:
                self._additional_callbacks["F6"]()
        self._last_polled_mode_switch_state = mode_down

    def get_device_state(self):
        """Return the current device state."""
        # Poll every frame so continuous controls remain responsive.
        self._poll_keyboard_state()
        action = np.zeros(15, dtype=np.float32)

        # Base commands are sent through the environment velocity interface.
        self._publish_base_command()

        # Left arm joints.
        action[1] = self._left_arm_delta[0] * self.sensitivity
        action[2] = self._left_arm_delta[1] * self.sensitivity
        action[3] = self._left_arm_delta[2] * self.sensitivity
        action[4] = self._left_arm_delta[3] * self.sensitivity
        action[5] = self._left_arm_delta[4] * self.sensitivity

        # Left gripper.
        action[6] = self._left_gripper_delta * self.sensitivity

        # Right arm joints.
        action[7] = self._right_arm_delta[0] * self.sensitivity
        action[8] = self._right_arm_delta[1] * self.sensitivity
        action[9] = self._right_arm_delta[2] * self.sensitivity
        action[10] = self._right_arm_delta[3] * self.sensitivity
        action[11] = self._right_arm_delta[4] * self.sensitivity

        # Right gripper.
        action[12] = self._right_gripper_delta * self.sensitivity

        # Head joints.
        action[13] = self._head_delta[0] * self.sensitivity
        action[14] = self._head_delta[1] * self.sensitivity

        return action

    def input2action(self):
        """Convert keyboard input into an action dictionary."""
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self.started
        if reset:
            self._reset_state = False
            return state
        state["joint_state"] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self.started
        ac_dict["keyboard"] = True
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        return ac_dict

    def advance(self):
        """Return the current action tensor."""
        # Poll before start so B can start control even if events are missed.
        self._poll_keyboard_state()
        if not self.started:
            return None

        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """Reset controller state."""
        self._position_delta.fill(0.0)
        self._rotation_delta = 0.0
        self._left_arm_delta.fill(0.0)
        self._right_arm_delta.fill(0.0)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta.fill(0.0)
        self._pressed_keys.clear()

        # Clear base velocity commands after reset or mode switches.
        if hasattr(self.env, "set_base_command"):
            self.env.set_base_command(np.zeros(3, dtype=np.float32))

    def add_callback(self, key: str, func: Callable):
        """Register a keyboard callback."""
        self._additional_callbacks[key] = func

    def __del__(self):
        """Release the keyboard subscription."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """Handle keyboard events."""
        try:
            if hasattr(event, "input") and hasattr(event.input, "name"):
                key_name = event.input.name
            elif hasattr(event, "name"):
                key_name = event.name
            else:
                return

            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]

                    # Control keys.
                    if control_type == "control":
                        if index == "start":
                            self.started = True
                            self._reset_state = False
                            print("[XlerobotKeyboard] Control started.")
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
                            # F6 is handled by polling to avoid double toggles.
                            pass
                        return

                    # Robot control keys.
                    if control_type == "position":
                        self._position_delta[index] = value
                    elif control_type == "rotation_joint":
                        self._rotation_delta = value
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = value
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = value
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = value
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = value
                    elif control_type == "head":
                        self._head_delta[index] = value

                    self._pressed_keys.add(key_name)

            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]

                    if control_type == "position":
                        self._position_delta[index] = 0.0
                    elif control_type == "rotation_joint":
                        self._rotation_delta = 0.0
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = 0.0
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = 0.0
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = 0.0
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = 0.0
                    elif control_type == "head":
                        self._head_delta[index] = 0.0

                    self._pressed_keys.discard(key_name)

        except Exception as e:
            print(f"[XlerobotKeyboard] Keyboard event error: {e}")

    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Xlerobot Keyboard Controller (continuous base control)\n"
        msg += f"\tKeyboard: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"\tSensitivity: {self.sensitivity}\n"
        msg += f"\tControl state: {'started' if self.started else 'stopped'}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBase (W/A/S/D + Q/E):\n"
        msg += "\t  Forward/backward: W/S\n"
        msg += "\t  Strafe left/right: A/D\n"
        msg += "\t  Rotate left/right: Q/E\n"
        msg += "\tLeft arm:\n"
        msg += "\t  Joint 1: R/F\n"
        msg += "\t  Joint 2: T/G\n"
        msg += "\t  Joint 3: Y/H\n"
        msg += "\t  Joint 4: U/J\n"
        msg += "\t  Joint 5: I/K\n"
        msg += "\tRight arm (numpad):\n"
        msg += "\t  Joint 1: 8/2\n"
        msg += "\t  Joint 2: 4/6\n"
        msg += "\t  Joint 3: 7/9\n"
        msg += "\t  Joint 4: 1/3\n"
        msg += "\t  Joint 5: 0/.\n"
        msg += "\tGrippers:\n"
        msg += "\t  Left open/close: F1/F2\n"
        msg += "\t  Right open/close: F3/F4\n"
        msg += "\tHead:\n"
        msg += "\t  Pan left/right: Home/End\n"
        msg += "\t  Tilt up/down: PageUp/PageDown\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart control: B\n"
        msg += "\tMark success: N\n"
        msg += "\tReset environment: F5\n"
        msg += "\tExit: Ctrl+C\n"
        msg += f"\tPressed keys: {list(self._pressed_keys)}"
        return msg
