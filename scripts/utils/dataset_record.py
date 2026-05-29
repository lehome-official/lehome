"""Dataset recording utility functions for teleoperation data collection."""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import gymnasium as gym
import numpy as np
import torch

from isaacsim.simulation_app import SimulationApp
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import (
    Se3Keyboard,
    SO101Leader,
    BiSO101Leader,
    BiKeyboard,
)
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger

from .common import stabilize_robot

logger = get_logger(__name__)


def _compute_camera_intrinsics(
    width: int, height: int, focal_length: float, horizontal_aperture: float
) -> tuple[float, float, float, float]:
    """Compute pixel intrinsics from Isaac camera parameters."""
    vertical_aperture = horizontal_aperture * (height / width)
    fx = focal_length * width / horizontal_aperture
    fy = focal_length * height / vertical_aperture
    cx = width / 2.0
    cy = height / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def _resolve_top_camera_base_cfg(env: DirectRLEnv, top_camera_cfg: Any) -> Any:
    """Resolve which robot-base config the top camera is mounted on."""
    prim_path = str(getattr(top_camera_cfg, "prim_path", ""))

    if "Right_Robot" in prim_path:
        return getattr(env.cfg, "right_robot", None)
    if "Left_Robot" in prim_path:
        return getattr(env.cfg, "left_robot", None)

    for attr in ("robot", "right_robot", "left_robot"):
        robot_cfg = getattr(env.cfg, attr, None)
        if robot_cfg is not None:
            return robot_cfg
    return None


def _build_top_depth_camera_info(env: DirectRLEnv) -> Dict[str, Any]:
    """Extract task-specific top-camera and robot-base metadata for depth reprojection."""
    top_camera_cfg = getattr(env.cfg, "top_camera", None)
    robot_cfg = _resolve_top_camera_base_cfg(env, top_camera_cfg) if top_camera_cfg is not None else None
    if top_camera_cfg is None or robot_cfg is None:
        return {}

    width = int(top_camera_cfg.width)
    height = int(top_camera_cfg.height)
    focal_length = float(top_camera_cfg.spawn.focal_length)
    horizontal_aperture = float(top_camera_cfg.spawn.horizontal_aperture)
    fx, fy, cx, cy = _compute_camera_intrinsics(
        width, height, focal_length, horizontal_aperture
    )

    return {
        "image_size": [height, width],
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "focal_length": focal_length,
            "horizontal_aperture": horizontal_aperture,
        },
        "camera_to_base": {
            "translation": [float(v) for v in top_camera_cfg.offset.pos],
            "rotation_wxyz": [float(v) for v in top_camera_cfg.offset.rot],
            "convention": str(top_camera_cfg.offset.convention),
        },
        "base_to_world": {
            "translation": [float(v) for v in robot_cfg.init_state.pos],
            "rotation_wxyz": [float(v) for v in robot_cfg.init_state.rot],
        },
    }


def validate_task_and_device(args: argparse.Namespace, env: Optional[DirectRLEnv] = None) -> None:
    """Validate that task name matches the teleoperation device configuration.

    Checks the environment's actual action dimension and ensures it matches the
    specified teleop device (single-arm vs dual-arm).

    Args:
        args: Command-line arguments containing teleop_device.
        env: The environment instance to check the action dimension from.

    Raises:
        ValueError: If task is not specified.
        AssertionError: If the device type does not match the arm configuration.
    """
    if args.task is None:
        raise ValueError("Please specify --task.")

    if env is not None:
        action_dim = env.cfg.action_space
        is_bi = (action_dim == 12)
        if is_bi:
            assert (
                args.teleop_device == "bi-so101leader"
                or args.teleop_device == "bi-keyboard"
            ), f"Task has {action_dim} actions (bi-arm), but device is {args.teleop_device}"
        else:
            assert (
                args.teleop_device == "so101leader" or args.teleop_device == "keyboard"
            ), f"Task has {action_dim} actions (single-arm), but device is {args.teleop_device}"


def create_teleop_interface(
    env: DirectRLEnv, args: argparse.Namespace
) -> Union[Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard]:
    """Create teleoperation interface based on device type.

    Args:
        env: The environment instance.
        args: Command-line arguments containing teleop_device, port, sensitivity, etc.

    Returns:
        An instance of the selected teleoperation interface.

    Raises:
        ValueError: If the specified teleop_device is not supported.
    """
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(
    teleop_interface: Any, recording_enabled: bool = False
) -> Dict[str, bool]:
    """Register callback functions for teleoperation control keys.

    Key bindings:
      - S: Start recording / Enable control
      - N: Mark current episode as successful and save (Only during recording)
      - D: Discard current episode and re-record / Reset environment
      - ESC: Abort recording and exit

    Args:
        teleop_interface: The teleoperation interface instance.
        recording_enabled: Whether dataset recording is enabled.

    Returns:
        A dictionary containing boolean flags for control state.
    """
    flags = {
        "start": False,   # S: Recording/Control started
        "success": False, # N: Success marker (triggers save in record mode)
        "remove": False,  # D: Removal/Reset marker
        "abort": False,   # ESC: Exit marker
    }

    def on_start():
        flags["start"] = True
        logger.info("[S] Recording/Control started!")

    def on_success():
        if not recording_enabled or not flags["start"]:
            logger.info("[N] Success marker received (Recording NOT enabled, so no data saved).")
            return
        flags["success"] = True

    def on_remove():
        if not flags["start"]:
            logger.info("[D] Ignored (Press S to start first).")
            return

        flags["remove"] = True
        if recording_enabled:
            pass
        else:
            logger.info("[D] Reset requested.")

    def on_abort():
        flags["abort"] = True
        logger.warning("[ESC] Abort recording/control.")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)

    return flags


def create_dataset_if_needed(
    args: argparse.Namespace,
    env: DirectRLEnv,
) -> Tuple[Optional[LeRobotDataset], Optional[Path], Optional[Any], bool]:
    """Create a LeRobotDataset for recording if enabled.

    Automatically handles feature definitions including joint states, actions,
    camera views, and optional end-effector poses.

    Args:
        args: Command-line arguments containing dataset configuration.
        env: The environment instance to determine action/state dimensions.

    Returns:
        A tuple of (dataset, json_path, ee_solver, is_bi_arm).
        - dataset: The LeRobotDataset instance or None.
        - json_path: Path to the metadata JSON file.
        - ee_solver: RobotKinematics solver if needed for EE recording.
        - is_bi_arm: True if the environment is dual-arm.

    Raises:
        ValueError: If record_ee_pose is requested but URDF path is missing.
    """
    if not args.enable_record:
        return None, None, None, False

    # Joint names for SO101/Lerobot arms
    action_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    action_dim = env.cfg.action_space
    is_bi_arm = (action_dim == 12)

    if is_bi_arm:
        left_names = [f"left_{n}" for n in action_names]
        right_names = [f"right_{n}" for n in action_names]
        joint_names = left_names + right_names
    else:
        joint_names = action_names

    features: Dict[str, Dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": joint_names,
        },
    }

    if not getattr(args, "disable_depth", False):
        depth_info: Dict[str, Any] = {
            "unit": "millimeters",
            "range_mm": [0, 65535],
            "range_m": [0.0, 65.535],
            "precision_mm": 1,
            "conversion": "depth_meters = uint16_value / 1000.0",
        }
        camera_info = _build_top_depth_camera_info(env)
        if camera_info:
            depth_info["camera"] = camera_info

        features["observation.top_depth"] = {
            "dtype": "uint16",
            "shape": (480, 640),
            "names": ["height", "width"],
            "info": depth_info,
        }

    if is_bi_arm:
        image_keys = ["top_rgb", "left_rgb", "right_rgb"]
    else:
        # Check if environment actually has wrist camera
        image_keys = ["top_rgb"]
        obs_dict = env._get_observations()
        if "observation.images.wrist_rgb" in obs_dict:
            image_keys.append("wrist_rgb")

    for key in image_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    if getattr(args, "record_ee_pose", False):
        if is_bi_arm:
            ee_pose_dim = 16
            ee_pose_names = [
                "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_gripper",
                "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_gripper",
            ]
        else:
            ee_pose_dim = 8
            ee_pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }
        features["action.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }

    root_path = Path(getattr(args, "dataset_root", "Datasets/record"))

    dataset = LeRobotDataset.create(
        repo_id="abc",
        fps=30,
        root=get_next_experiment_path_with_gap(root_path),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )
    # Generic metadata file name
    json_path = dataset.root / "meta" / "episode_metadata.json"
    logger.info(f"Dataset recording enabled. Output directory: {dataset.root}")

    solver = None
    if getattr(args, "record_ee_pose", False):
        if not args.ee_urdf_path:
            raise ValueError("--record_ee_pose requires --ee_urdf_path")

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        from lehome.utils import RobotKinematics
        solver_joint_names = [n.replace("left_", "").replace("right_", "") for n in joint_names[:5]] # Simplified

        solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=solver_joint_names,
        )
        arm_type = "dual-arm" if is_bi_arm else "single-arm"
        logger.info(f"End-effector pose solver loaded ({arm_type})")

    return dataset, json_path, solver, is_bi_arm


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run the idle phase before recording starts.

    In this phase, the robot follows teleop commands but data is not saved.
    It handles environment initialization, robot stabilization, and capture of
    the initial object pose.

    Args:
        env: The environment instance.
        teleop_interface: The teleoperation interface.
        args: Command-line arguments.
        count_render: Counter tracking the number of render steps.

    Returns:
        A tuple of (object_initial_pose, updated_count_render).
    """
    # Continuously reset gripper effort limit to ensure responsive control
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

    # Get actions from the teleop device (Keyboard/Leader-Arm)
    actions = teleop_interface.advance()
    object_initial_pose = None

    # First-frame initialization
    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        count_render += 1

        logger.info("[Idle Phase] Stabilizing environment... (Please wait for CONTROL INSTRUCTIONS to appear.)")
        stabilize_robot(env)

    if actions is None:
        # If no teleop input, maintain current joint positions to prevent falling
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
            else:
                maintain_action = torch.zeros(1, len(current_state), dtype=torch.float32, device=env.device)
        else:
            maintain_action = torch.zeros(1, env.cfg.action_space, dtype=torch.float32, device=env.device)
        env.step(maintain_action)
        env.render()
    else:
        # Step the environment with teleop actions
        env.step(actions)
        if hasattr(env, "get_all_pose"):
            object_initial_pose = env.get_all_pose()

    # Ensure we capture the pose for the first episode
    if object_initial_pose is None and hasattr(env, "get_all_pose"):
        object_initial_pose = env.get_all_pose()

    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
    dataset: LeRobotDataset,
    json_path: Path,
    initial_object_pose: Optional[Dict[str, Any]],
    ee_solver: Optional[Any] = None,
    is_bi_arm: bool = False,
) -> Dict[str, Any]:
    """Run the recording phase to collect expert demonstrations.

    Handles the episode loop, frame-by-frame data collection, success marking,
    episode discarding, and saving to disk.

    Args:
        env: The environment instance.
        teleop_interface: The teleoperation interface.
        args: Command-line arguments.
        flags: Control flags (start, success, remove, abort).
        dataset: The LeRobotDataset instance for saving data.
        json_path: Path to saving additional episode metadata.
        initial_object_pose: Initial state of objects for the current episode.
        ee_solver: Optional RobotKinematics solver for Cartesian pose recording.
        is_bi_arm: Whether the task uses two arms.

    Returns:
        The object initial pose for the next episode.
    """
    episode_index = 0
    object_initial_pose = initial_object_pose

    # Safety check for initial pose
    if object_initial_pose is None and hasattr(env, "get_all_pose"):
        object_initial_pose = env.get_all_pose()

    # Main episode recording loop
    while episode_index < args.num_episode:
        # Check for global abort (ESC)
        if flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        success_announced = False

        # Single episode collection loop
        while not flags["success"]:
            # Check for abort inside frame loop
            if flags["abort"]:
                dataset.clear_episode_buffer()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            try:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
                actions = teleop_interface.advance()
            except Exception as e:
                logger.error(f"[Recording] Error in teleop interface: {e}", exc_info=True)
                actions = None

            if actions is None:
                env.render()
            else:
                env.step(actions)

            # Collect observations
            observations = env._get_observations()
            if getattr(args, "disable_depth", False) and "observation.top_depth" in observations:
                observations.pop("observation.top_depth")

            # Keep pointcloud conversion offline so teleoperation stays responsive
            # across the current multi-task release.
            if getattr(args, "enable_pointcloud", False):
                # Converting pointcloud online is extremely time-consuming (due to FPS sampling),
                # which would break teleoperation fluidity. Please convert offline instead.
                # pointcloud = env._get_workspace_pointcloud(num_points=4096, use_fps=True)
                logger.warning("Online pointcloud conversion is disabled to ensure teleop fluidity. Use offline scripts.")

            # Prepare data frame for LeRobotDataset
            frame = {**observations, "task": args.task_description}

            # Optional: Online End-Effector pose computation
            if (ee_solver is not None and "observation.state" in observations and "action" in observations):
                from lehome.utils import compute_ee_pose_single_arm
                obs_state = np.array(observations["observation.state"], dtype=np.float32)
                action_state = np.array(observations["action"], dtype=np.float32)

                if is_bi_arm:
                    # Bimanual IK computation
                    obs_left = compute_ee_pose_single_arm(ee_solver, obs_state[:6], args.ee_state_unit)
                    obs_right = compute_ee_pose_single_arm(ee_solver, obs_state[6:12], args.ee_state_unit)
                    frame["observation.ee_pose"] = np.concatenate([obs_left, obs_right], axis=0)
                    
                    act_left = compute_ee_pose_single_arm(ee_solver, action_state[:6], args.ee_state_unit)
                    act_right = compute_ee_pose_single_arm(ee_solver, action_state[6:12], args.ee_state_unit)
                    frame["action.ee_pose"] = np.concatenate([act_left, act_right], axis=0)
                else:
                    # Single-arm IK computation
                    frame["observation.ee_pose"] = compute_ee_pose_single_arm(ee_solver, obs_state, args.ee_state_unit)
                    frame["action.ee_pose"] = compute_ee_pose_single_arm(ee_solver, action_state, args.ee_state_unit)

            # Add frame to buffer
            dataset.add_frame(frame)

            # Check automatic success condition for terminal feedback.
            # This does not auto-save the episode; the operator still confirms
            # the trajectory with the success hotkey.
            if not success_announced:
                try:
                    env_success = env._get_success()
                    if isinstance(env_success, torch.Tensor):
                        reached_success = bool(env_success.any().item())
                    else:
                        reached_success = bool(env_success)

                    if reached_success:
                        success_announced = True
                        logger.info(
                            f"Episode {episode_index + 1} reached the success condition. Press N to save this trajectory."
                        )
                except (AttributeError, NotImplementedError):
                    pass

            # Check for episode truncation from environment
            _, truncated = env._get_dones()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.any().item()

            # Check if user wants to discard this episode (D key) or environment timed out
            if flags["remove"] or truncated:
                if truncated:
                    logger.warning(f"Episode {episode_index} truncated by environment (timeout). Discarding...")
                else:
                    logger.info(f"Episode {episode_index} discarded by user.")
                
                dataset.clear_episode_buffer()
                logger.info(f"Re-recording episode {episode_index}")
                env.reset()
                stabilize_robot(env)
                if hasattr(env, "get_all_pose"):
                    object_initial_pose = env.get_all_pose()
                flags["remove"] = False
                continue

        # Episode successfully completed (N key pressed)
        save_start_time = time.time()
        logger.info(f"[Recording] Saving episode {episode_index}...")
        dataset.save_episode()
        logger.info(f"[Recording] Episode {episode_index} saved (took {time.time() - save_start_time:.1f}s)")
        logger.info(
            f"Episode {episode_index + 1}/{args.num_episode} recorded successfully."
        )

        # Extract metadata (variant name) for future replay
        variant_name = getattr(env.cfg, "variant_name", "default")

        # Record the exact transformation of the object after stabilization
        if hasattr(env, "get_all_pose"):
            append_episode_initial_pose(
                json_path, episode_index, object_initial_pose,
                variant_name=variant_name
            )

        episode_index += 1
        logger.info(f"Progress: {episode_index}/{args.num_episode}")

        # Reset for next episode
        env.reset()
        stabilize_robot(env)
        if hasattr(env, "get_all_pose"):
            object_initial_pose = env.get_all_pose()

    # Finalize dataset writing
    dataset.clear_episode_buffer()
    dataset.finalize()
    logger.info(f"All {args.num_episode} episodes completed!")
    logger.info(f"Dataset collection completed successfully. Total episodes: {args.num_episode}")
    return object_initial_pose


def run_live_control_without_record(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
) -> None:
    """Run live teleoperation control without recording to a dataset.

    Useful for testing teleop devices or just interacting with the environment.
    Supports reset (D key) and success markers (N key).

    Args:
        env: The environment instance.
        teleop_interface: The teleoperation interface.
        args: Command-line arguments.
        flags: Control flags.
    """
    # Handle manual reset request
    if flags["remove"]:
        logger.info("[Live Control] Resetting environment...")
        env.reset()
        flags["remove"] = False
        return

    # Handle success marker (log only)
    if flags["success"]:
        logger.info("[Live Control] Success marker.")
        flags["success"] = False

    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()

    if actions is None:
        # Prevent arm from falling when no input is received
        obs = env._get_observations()
        if "observation.state" in obs:
            state = obs["observation.state"]
            action = torch.from_numpy(state).float().unsqueeze(0).to(env.device) if isinstance(state, np.ndarray) else torch.zeros(1, len(state), device=env.device)
        else:
            action = torch.zeros(1, env.cfg.action_space, device=env.device)
        env.step(action)
        env.render()
    else:
        env.step(actions)

    if args.log_success:
        _ = env._get_success()


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Main entry point for starting the dataset recording/control process.

    This function sets up the environment, teleop interface, and dataset,
    then enters the main application loop, dispatching to idle or recording phases.

    Args:
        args: Parsed command-line arguments.
        simulation_app: The running Isaac Sim application instance.
    """
    # 1. Environment Setup
    env_cfg = parse_env_cfg(args.task, device=getattr(args, "device", "cpu"))
    
    # Apply CLI seed settings for environment initialization.
    if hasattr(env_cfg, "use_random_seed"):
        env_cfg.use_random_seed = args.use_random_seed

    if args.use_random_seed:
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = None
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = None
        logger.info("Using random seed (no fixed seed)")
    else:
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        if hasattr(env_cfg, "random_seed"):
            env_cfg.random_seed = args.seed
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    # 2. Instantiate Environment
    env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped
    validate_task_and_device(args, env)
    
    # 3. Setup Interface and Callbacks
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(teleop_interface, recording_enabled=args.enable_record)
    teleop_interface.reset()
    
    # 4. Dataset and Solver Setup
    dataset, json_path, ee_solver, is_bi_arm = create_dataset_if_needed(args, env)
    
    # State tracking variables
    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose: Optional[Dict[str, Any]] = None

    # 5. Main Loop
    try:
        while simulation_app.is_running():
            # Evaluation and recording usually don't need gradients
            with torch.inference_mode():
                # Loop Branch A: Waiting to start (Idle Phase)
                if not flags["start"]:
                    pose, count_render = run_idle_phase(env, teleop_interface, args, count_render)
                    if pose: 
                        object_initial_pose = pose
                    
                    if flags["abort"]: # Handle ESC during idle phase
                        break
                    
                    # Print instructions once stabilization is done
                    if count_render > 0:
                        idle_frame_counter += 1
                        if idle_frame_counter == 20 and not printed_instructions:
                            logger.info("=" * 60 + "\n🎮 CONTROL INSTRUCTIONS 🎮\n" + str(teleop_interface) + "\n" + "=" * 60)
                            printed_instructions = True
                
                # Loop Branch B: Active Recording (S key pressed and --enable_record set)
                elif args.enable_record and dataset is not None:
                    object_initial_pose = run_recording_phase(
                        env, teleop_interface, args, flags, dataset, json_path, 
                        object_initial_pose, ee_solver, is_bi_arm
                    )
                    break # All episodes finished
                
                # Loop Branch C: Live Control (S key pressed but no recording)
                else:
                    run_live_control_without_record(env, teleop_interface, args, flags)
                    if flags["abort"]: # Handle ESC during live control
                        break
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        if args.enable_record and dataset is not None and flags["start"]:
            logger.info("Finalizing dataset safely...")
            dataset.clear_episode_buffer()
            dataset.finalize()
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        # Crucial: Always release simulator resources
        env.close()
