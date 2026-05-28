from collections.abc import Callable

from .xlerobot_leader import XlerobotLeader
from ..device_base import Device


class BiXlerobotLeader(Device):
    def __init__(
        self,
        env,
        left_port: str = '/dev/ttyACM0',
        right_port: str = '/dev/ttyACM1',
        recalibrate: bool = False,
        mirror_right_from_left: bool = False,
    ):
        super().__init__(env)
        self._mirror_right_from_left = mirror_right_from_left or (left_port == right_port)

        # use left so101 leader as the main device to store state
        print("Connecting to left_xlerobot_leader...")
        self.left_xlerobot_leader = XlerobotLeader(env, left_port, recalibrate, "left_xlerobot_leader.json")
        if self._mirror_right_from_left:
            print("BiXlerobotLeader: single-arm mirror mode enabled (right arm follows left arm).")
            self.right_xlerobot_leader = self.left_xlerobot_leader
        else:
            print("Connecting to right_xlerobot_leader...")
            self.right_xlerobot_leader = XlerobotLeader(env, right_port, recalibrate, "right_xlerobot_leader.json")

        if not self._mirror_right_from_left:
            self.right_xlerobot_leader.listener.stop()

    def __str__(self) -> str:
        """Returns: A string containing the information of bi-xlerobot leader."""
        msg = "Bi-Xlerobot-Leader device for hybrid control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Bi-Xlerobot-Leader to control Bi-Xlerobot-Follower\n"
        msg += "\tThis version is specifically designed for hybrid control.\n"
        return msg

    def add_callback(self, key: str, func: Callable):
        self.left_xlerobot_leader.add_callback(key, func)
        if not self._mirror_right_from_left:
            self.right_xlerobot_leader.add_callback(key, lambda: None)

    def reset(self):
        self.left_xlerobot_leader.reset()
        if not self._mirror_right_from_left:
            self.right_xlerobot_leader.reset()

    def get_device_state(self):
        if self._mirror_right_from_left:
            left = self.left_xlerobot_leader.get_device_state()
            return {
                "left_arm": left,
                "right_arm": left,
            }
        return {
            "left_arm": self.left_xlerobot_leader.get_device_state(),
            "right_arm": self.right_xlerobot_leader.get_device_state()
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self.left_xlerobot_leader.reset_state
        state['started'] = self.left_xlerobot_leader.started
        if reset:
            self.left_xlerobot_leader.reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.left_xlerobot_leader.started
        ac_dict['bi_xlerobot_leader'] = True
        ac_dict['bi_so101_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = {
            'left_arm': self.left_xlerobot_leader.motor_limits,
            'right_arm': self.left_xlerobot_leader.motor_limits if self._mirror_right_from_left else self.right_xlerobot_leader.motor_limits
        }
        return ac_dict
