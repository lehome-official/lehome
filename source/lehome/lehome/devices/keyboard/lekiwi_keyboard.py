import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class LekiwiKeyboard(Device):
    """Keyboard controller for the Lekiwi mobile base and arm."""

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
            lambda event, *args, obj=weakref.proxy(self):
                obj._on_keyboard_event(event, *args),
        )

        # Key bindings.
        self._create_key_bindings()

        # Command buffers.
        self._omnimove_command = None
        self._arm_delta = np.zeros(5)
        self._gripper_delta = 0.0

        # State flags and callbacks.
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # Pressed-key tracking.
        self._pressed_keys = set()

    def _create_key_bindings(self):
        """Create keyboard bindings."""
        self._key_bindings = {
            # Omni-wheel base control.
            "W": ("omnimove", "forward", 1.0),
            "S": ("omnimove", "backward", 1.0),
            "A": ("omnimove", "left", 1.0),
            "D": ("omnimove", "right", 1.0),
            "Q": ("omnimove", "rotate_left", 1.0),
            "E": ("omnimove", "rotate_right", 1.0),

            # Arm control.
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

            # Gripper control.
            "O": ("gripper", 0, 1.0),
            "P": ("gripper", 0, -1.0),

            # Control keys.
            "B": ("control", "start", 1.0),
            "N": ("control", "success", 1.0),
            "F5": ("control", "reset", 1.0),
            "F6": ("control", "mode_switch", 1.0),
        }

    def get_device_state(self):
        """Return the current device state."""
        action = np.zeros(9)

        # Omni-wheel base control.
        if self._omnimove_command == "forward":
            action[1] = 1.0 * self.sensitivity
            action[2] = 1.0 * self.sensitivity
        elif self._omnimove_command == "backward":
            action[1] = -1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "left":
            action[0] = 1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "right":
            action[0] = -1.0 * self.sensitivity
            action[1] = -1.0 * self.sensitivity
        elif self._omnimove_command == "rotate_left":
            action[0] = -1.0 * self.sensitivity
            action[1] = 1.0 * self.sensitivity
            action[2] = -1.0 * self.sensitivity
        elif self._omnimove_command == "rotate_right":
            action[0] = 1.0 * self.sensitivity
            action[1] = -1.0 * self.sensitivity
            action[2] = 1.0 * self.sensitivity

        # Arm control (3-7).
        action[3] = self._arm_delta[0] * self.sensitivity  # shoulder_pitch
        action[4] = self._arm_delta[1] * self.sensitivity  # shoulder_roll
        action[5] = self._arm_delta[2] * self.sensitivity  # elbow
        action[6] = self._arm_delta[3] * self.sensitivity  # wrist_pitch
        action[7] = self._arm_delta[4] * self.sensitivity  # wrist_roll

        # Gripper control (8).
        action[8] = self._gripper_delta * self.sensitivity

        return action

    def input2action(self):
        """Convert keyboard input into an action dictionary."""
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
        """Return the current action tensor."""
        if not self.started:
            return None

        action = self.get_device_state()
        return torch.tensor(
            action, dtype=torch.float32, device=self.env.device
        )

    def reset(self):
        """Reset controller state."""
        self._omnimove_command = None
        self._arm_delta.fill(0)
        self._gripper_delta = 0.0
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """Register a keyboard callback."""
        self._additional_callbacks[key] = func

    def __del__(self):
        """Release the keyboard subscription."""
        if (hasattr(self, '_input') and hasattr(self, '_keyboard') and
                hasattr(self, '_keyboard_sub')):
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """Handle keyboard events."""
        try:
            if hasattr(event, 'input') and hasattr(event.input, 'name'):
                key_name = event.input.name
            elif hasattr(event, 'name'):
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
                            print("[LekiwiKeyboard] Control started.")
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

                    # Robot control keys.
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

                    # Only robot-control keys need release handling.
                    if control_type == "omnimove":
                        if self._omnimove_command == index:
                            self._omnimove_command = None
                    elif control_type == "arm":
                        self._arm_delta[index] = 0.0
                    elif control_type == "gripper":
                        self._gripper_delta = 0.0

                    self._pressed_keys.discard(key_name)

        except Exception as e:
            print(f"[LekiwiKeyboard] Keyboard event error: {e}")

    def __str__(self) -> str:
        """Return a human-readable controller summary."""
        msg = "Lekiwi Keyboard Controller\n"
        msg += (
            f"\tKeyboard: {self._input.get_keyboard_name(self._keyboard)}\n"
        )
        msg += f"\tSensitivity: {self.sensitivity}\n"
        msg += f"\tControl state: {'started' if self.started else 'stopped'}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMobile base:\n"
        msg += "\t  Forward/backward: W/S\n"
        msg += "\t  Strafe left/right: A/D\n"
        msg += "\t  Rotate left/right: Q/E\n"
        msg += "\tArm:\n"
        msg += "\t  Shoulder pitch: R/F\n"
        msg += "\t  Shoulder roll: T/G\n"
        msg += "\t  Elbow: Y/H\n"
        msg += "\t  Wrist pitch: U/J\n"
        msg += "\t  Wrist roll: I/K\n"
        msg += "\tGripper:\n"
        msg += "\t  Open/close: O/P\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart control: B\n"
        msg += "\tMark success: N\n"
        msg += "\tReset environment: F5\n"
        msg += "\tExit: Ctrl+C\n"
        msg += f"\tPressed keys: {list(self._pressed_keys)}"
        return msg
