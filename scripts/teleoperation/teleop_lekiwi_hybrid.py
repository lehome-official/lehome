# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run lehome teleoperation with lekiwi hybrid control."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing  # noqa: E402

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="lehome teleoperation with lekiwi hybrid control."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="LeIsaac-Lekiwi-Direct-Task-v0",
    help="Name of the task."
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for the environment."
)
parser.add_argument(
    "--sensitivity",
    type=float,
    default=1.0,
    help="Sensitivity factor for keyboard control."
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=30,
    help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for SO101Leader arm controller."
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    help="Recalibrate the SO101Leader device."
)
parser.add_argument(
    "--control_mode",
    type=str,
    default="hybrid",
    choices=["keyboard", "hybrid", "arm_only"],
    help="Initial control mode."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import other modules after Isaac Sim is launched
import time  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab.envs import DirectRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from lehome.devices import LekiwiHybridController  # noqa: E402
from lehome.utils.env_utils import (  # noqa: E402
    dynamic_reset_gripper_effort_limit_sim
)
from lehome.devices.lekiwi_action_process import (  # noqa: E402
    init_lekiwi_action_cfg
)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Running lehome teleoperation with lekiwi hybrid control."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    # 初始化lekiwi动作配置为hybrid模式
    env_cfg = init_lekiwi_action_cfg(env_cfg, "lekiwi_hybrid")

    task_name = args_cli.task

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # create lekiwi hybrid controller
    teleop_interface = LekiwiHybridController(
        env,
        sensitivity=0.25 * args_cli.sensitivity,
        arm_port=args_cli.arm_port,
        recalibrate=args_cli.recalibrate
    )

    # 设置初始控制模式
    teleop_interface.set_control_mode(args_cli.control_mode)

    def sync_env_action_mode():
        """Sync env action mode with current hybrid controller mode.

        keyboard -> relative (align with teleop_lekiwi.py)
        hybrid/arm_only -> absolute
        """
        if not hasattr(env, "set_action_mode"):
            return
        desired_mode = (
            "relative" if teleop_interface.control_mode == "keyboard"
            else "absolute"
        )
        if getattr(env, "_action_mode", None) != desired_mode:
            env.set_action_mode(desired_mode)

    # 初始同步一次：保证 --control_mode=keyboard 时与纯键盘脚本一致
    sync_env_action_mode()

    # 输出控制说明
    print("\n" + "="*60)
    print("🎮 Lekiwi 混合控制说明")
    print("="*60)
    print("底盘控制（键盘）：")
    print(teleop_interface.keyboard_controller)
    print("\n机械臂控制（SO101Leader）：")
    print("  - 物理leader设备控制机械臂和夹爪")
    print("  - 按 F6 切换控制模式")
    print("  - 按 R/S/N/D 进行机械臂校准")
    print("="*60)
    print("💡 提示：按 B 键启动控制，按 F5 重置环境")
    print("="*60 + "\n")

    # add teleoperation key for env reset
    reset_detected = False

    def reset_env():
        nonlocal reset_detected
        reset_detected = True

    teleop_interface.add_callback("R", reset_env)  # Reset environment

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # 动态重置夹爪力矩限制
            dynamic_reset_gripper_effort_limit_sim(env, "lekiwi_hybrid")

            # 先更新状态，再获取动作
            teleop_interface.input2action()  # 更新状态
            sync_env_action_mode()  # F6切换后实时同步环境动作模式
            actions = teleop_interface.advance()  # 获取动作

            if actions is None:
                env.render()
            else:
                env.step(actions)

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

            # Handle environment reset
            if reset_detected:
                env.reset()
                reset_detected = False

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
