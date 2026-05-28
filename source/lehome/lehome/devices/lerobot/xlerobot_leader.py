import os
import json
import time
from collections.abc import Callable
from typing import Dict, Tuple
from pynput.keyboard import Listener

from .common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorNormMode,
    MotorCalibration,
    OperatingMode,
)
from .common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..device_base import Device

from lehome.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS


class XlerobotLeader(Device):
    """A Xlerobot Leader device for hybrid control."""

    def __init__(
        self,
        env,
        port: str = "/dev/ttyACM0",
        recalibrate: bool = False,
        calibration_file_name: str = "xlerobot_leader.json",
    ):
        super().__init__(env)
        self.port = port

        # calibration
        self.calibration_path = os.path.join(
            os.path.dirname(__file__), ".cache", calibration_file_name
        )
        if not os.path.exists(self.calibration_path) or recalibrate:
            self.calibrate()
        calibration = self._load_calibration()

        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        # connect
        self.connect()
        self._joint_names = tuple(self._bus.motors.keys())

        # Robustness knobs for transient serial read failures.
        self._sync_read_retry = max(0, int(os.getenv("LEHOME_XLEROBOT_SYNC_READ_RETRY", "2")))
        self._read_warn_every = max(1, int(os.getenv("LEHOME_XLEROBOT_READ_WARN_EVERY", "20")))
        self._read_backoff_sleep_s = max(0.0, float(os.getenv("LEHOME_XLEROBOT_READ_BACKOFF_S", "0.002")))
        self._consecutive_read_failures = 0
        self._last_read_fail_msg_time = 0.0
        self._last_joint_state = {name: 0.0 for name in self._joint_names}

        # Read initial state once; fall back to zeros without blocking startup.
        try:
            self._last_joint_state = self._bus.sync_read(
                "Present_Position", num_retry=self._sync_read_retry
            )
        except Exception as exc:
            print(f"[XlerobotLeader] initial sync_read failed, fallback to zeros: {exc}")

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()
        self.b_disable = False
        self.other_key_enable = False

    def __str__(self) -> str:
        """Returns: A string containing the information of xlerobot leader."""
        msg = "Xlerobot-Leader device for hybrid control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Xlerobot-Leader to control Xlerobot-Follower\n"
        msg += "\tThis version is specifically designed for hybrid control.\n"
        return msg

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start simulation")
        print_command("move leader", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    def on_press(self, key):
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            if key.char == "b":
                if self.b_disable == False:
                    self._started = True
                    self._reset_state = False
                    self.other_key_enable = True
                    print("[XlerobotLeader] Control started.")
        except AttributeError:
            pass
        except Exception as e:
            print(f"Error in keyboard callback: {e}")
            pass

    def get_device_state(self):
        try:
            joint_state = self._bus.sync_read(
                "Present_Position", num_retry=self._sync_read_retry
            )
            self._last_joint_state = joint_state
            self._consecutive_read_failures = 0
            return joint_state
        except Exception as exc:
            self._consecutive_read_failures += 1
            # Fall back to sequential reads so one dropped ID does not lose the frame.
            recovered_state = {}
            recovered_count = 0
            for joint_name in self._joint_names:
                try:
                    recovered_state[joint_name] = self._bus.read(
                        "Present_Position",
                        joint_name,
                        num_retry=self._sync_read_retry,
                    )
                    recovered_count += 1
                except Exception:
                    recovered_state[joint_name] = self._last_joint_state.get(joint_name, 0.0)

            if recovered_count > 0:
                self._last_joint_state = dict(recovered_state)
                # Throttle warnings while still surfacing serial instability.
                if (
                    self._consecutive_read_failures == 1
                    or self._consecutive_read_failures % self._read_warn_every == 0
                ):
                    now = time.monotonic()
                    if now - self._last_read_fail_msg_time > 0.2:
                        print(
                            "[XlerobotLeader] sync_read failed, "
                            f"fallback sequential read recovered {recovered_count}/{len(self._joint_names)} joints."
                        )
                        print(f"[XlerobotLeader] detail: {exc}")
                        self._last_read_fail_msg_time = now
                return dict(self._last_joint_state)

            if (
                self._consecutive_read_failures == 1
                or self._consecutive_read_failures % self._read_warn_every == 0
            ):
                now = time.monotonic()
                if now - self._last_read_fail_msg_time > 0.2:
                    print(
                        "[XlerobotLeader] sync_read dropped packet "
                        f"(failures={self._consecutive_read_failures}), "
                        "using last joint state fallback."
                    )
                    print(f"[XlerobotLeader] detail: {exc}")
                    self._last_read_fail_msg_time = now
            if self._read_backoff_sleep_s > 0:
                time.sleep(self._read_backoff_sleep_s)
            return dict(self._last_joint_state)

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self._started
        if reset:
            self._reset_state = False
            return state
        state["joint_state"] = self.get_device_state()
        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self._started
        ac_dict["xlerobot_leader"] = True
        ac_dict["so101_leader"] = True  # backward-compatible alias
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._motor_limits

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("Xlerobot-Leader is not connected.")
        self._bus.disconnect()
        print("Xlerobot-Leader disconnected.")

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Xlerobot-Leader is already connected.")
        self._bus.connect()
        self.configure()
        print("Xlerobot-Leader connected.")

    def configure(self) -> None:
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def calibrate(self):
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        self.connect()

        print("\n Running calibration of Xlerobot-Leader")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            "Move Xlerobot-Leader to the middle of its range of motion and press ENTER..."
        )
        homing_offset = self._bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")

        self.disconnect()

    def _load_calibration(self) -> Dict[str, MotorCalibration]:
        with open(self.calibration_path, "r") as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def _save_calibration(self, calibration: Dict[str, MotorCalibration]):
        save_calibration = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }
        if not os.path.exists(os.path.dirname(self.calibration_path)):
            os.makedirs(os.path.dirname(self.calibration_path))
        with open(self.calibration_path, "w") as f:
            json.dump(save_calibration, f, indent=4)
