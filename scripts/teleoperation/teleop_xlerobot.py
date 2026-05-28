# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run lehome teleoperation with xlerobot control."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="lehome teleoperation with xlerobot control."
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
    default="LeIsaac-Xlerobot-Direct-Task-v0",
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import other modules after Isaac Sim is launched
import time
import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.devices import XlerobotKeyboard
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.devices.xlerobot_action_process import (
    init_xlerobot_action_cfg,
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
    """Running lehome teleoperation with xlerobot control."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    
    # 初始化xlerobot动作配置
    env_cfg = init_xlerobot_action_cfg(env_cfg, "keyboard")
    
    task_name = args_cli.task

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # create xlerobot keyboard controller
    teleop_interface = XlerobotKeyboard(
        env, 
        sensitivity=0.25 * args_cli.sensitivity
    )

    # add teleoperation key for env reset
    reset_detected = False

    def reset_env():
        nonlocal reset_detected
        reset_detected = True

    teleop_interface.add_callback("R", reset_env)  # Reset environment

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    teleop_interface.reset()
    
    count_render = 0
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # 动态重置夹爪力矩限制
            dynamic_reset_gripper_effort_limit_sim(env, "xlerobot")  # 修复：传递正确的设备类型
            
            # Get actions from keyboard
            actions = teleop_interface.advance()
            
            if actions is None:
                env.render()
            else:
                env.step(actions)
                
            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)
                
            # Initialize object on first render
            if count_render == 0:
                if (hasattr(env, 'object') and
                        hasattr(env.object, 'initialize')):
                    env.object.initialize()
                count_render += 1
                
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
