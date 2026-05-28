# scripts/teleoperation/teleop_xlerobot_hybrid.py

"""Script to run lehome teleoperation with xlerobot hybrid control."""

import logging
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher


class _QuietIsaacLabWarnings(logging.Filter):
    """Filter known noisy IsaacLab warnings during teleoperation startup."""

    _IGNORED_PREFIXES = (
        "The `enable_external_forces_every_iteration` parameter",
        "Not all actuators are configured!",
    )

    def filter(self, record):
        return not str(record.getMessage()).startswith(self._IGNORED_PREFIXES)


def _suppress_noisy_startup_warnings():
    warning_filter = _QuietIsaacLabWarnings()
    for logger_name in (
        "isaaclab.sim.simulation_context",
        "isaaclab.assets.articulation.articulation",
    ):
        logging.getLogger(logger_name).addFilter(warning_filter)

# add argparse arguments
parser = argparse.ArgumentParser(
    description="lehome teleoperation with xlerobot hybrid control."
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
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left arm controller",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right arm controller",
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    default=False,
    help="recalibrate BiSO101Leader",
)
parser.add_argument(
    "--control_mode",
    type=str,
    default="hybrid",
    choices=["keyboard", "hybrid", "arms_only"],
    help="Control mode: keyboard, hybrid, or arms_only",
)
parser.add_argument(
    "--quiet_kit_logs",
    action="store_true",
    help="Suppress Omniverse/PhysX warning spam in the terminal. Diagnostics still go to Kit log files.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if args_cli.quiet_kit_logs:
    _suppress_noisy_startup_warnings()
    quiet_kit_args = (
        '--/log/outputStreamLevel="Fatal" '
        '--/log/debugConsoleLevel="Fatal" '
        '--/log/fileLogLevel="Warning"'
    )
    app_launcher_args["kit_args"] = f"{app_launcher_args.get('kit_args', '')} {quiet_kit_args}".strip()

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import other modules after Isaac Sim is launched
import time
import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.devices.hybrid import XlerobotHybridController
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
    """Running lehome teleoperation with xlerobot hybrid control."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    
    # Initialize Xlerobot action configuration.
    env_cfg = init_xlerobot_action_cfg(env_cfg, "hybrid")
    
    task_name = args_cli.task

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # create xlerobot hybrid controller
    teleop_interface = XlerobotHybridController(
        env, 
        sensitivity=0.25 * args_cli.sensitivity,
        left_arm_port=args_cli.left_arm_port,
        right_arm_port=args_cli.right_arm_port,
        recalibrate=args_cli.recalibrate
    )

    # Set initial control mode.
    teleop_interface.set_control_mode(args_cli.control_mode)

    def sync_env_action_mode():
        """Sync env action mode with current controller mode.

        keyboard -> relative (align with teleop_xlerobot.py)
        hybrid/arms_only -> absolute
        """
        if not hasattr(env, "set_action_mode"):
            return
        desired_mode = (
            "relative" if teleop_interface.control_mode == "keyboard"
            else "absolute"
        )
        if getattr(env, "_action_mode", None) != desired_mode:
            env.set_action_mode(desired_mode)

    # Initial sync keeps --control_mode=keyboard aligned with teleop_xlerobot.py.
    sync_env_action_mode()

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
            # Refresh gripper effort limits when dynamic gripper logic is enabled.
            dynamic_reset_gripper_effort_limit_sim(env, "xlerobot")
            
            teleop_interface.input2action()
            sync_env_action_mode()
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
