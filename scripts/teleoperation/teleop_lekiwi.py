# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run lehome teleoperation with lekiwi control."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing  # noqa: E402

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="lehome teleoperation with lekiwi control."
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

from lehome.devices import LekiwiKeyboard  # noqa: E402
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
    """Running lehome teleoperation with lekiwi control."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    # Initialize Lekiwi action configuration.
    env_cfg = init_lekiwi_action_cfg(env_cfg, "lekiwi_keyboard")

    task_name = args_cli.task

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # create lekiwi keyboard controller
    teleop_interface = LekiwiKeyboard(
        env,
        sensitivity=0.25 * args_cli.sensitivity
    )

    # Print control instructions.
    print("\n" + "="*60)
    print("Lekiwi Keyboard Teleoperation")
    print("="*60)
    print(teleop_interface)
    print("="*60)
    print("Press B to start control. Press F5 to reset the environment.")
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
            # Refresh gripper effort limits when dynamic gripper logic is enabled.
            dynamic_reset_gripper_effort_limit_sim(env, "lekiwi_keyboard")

            actions = teleop_interface.advance()

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
