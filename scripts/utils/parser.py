"""Argument parser setup functions for LeHome scripts subcommands."""

import argparse


def setup_record_parser(
    subparsers: argparse.ArgumentParser,
    parent_parsers: list[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for 'record' subcommand."""
    parser = subparsers.add_parser(
        "record",
        help="Record teleoperation data",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

    # Teleoperation parameters
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Device for interacting with environment",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device.")
    parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0", help="Port for the left teleop device.")
    parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1", help="Port for the right teleop device.")
    parser.add_argument("--recalibrate", action="store_true", default=False, help="Recalibrate teleop device.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

    # Task configuration
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--use_random_seed", action="store_true", default=False, help="Use random seed.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument("--log_success", action="store_true", default=False, help="Log success information.")

    # Recording parameters
    parser.add_argument("--enable_record", action="store_true", default=False, help="Enable dataset recording.")
    parser.add_argument("--step_hz", type=int, default=120, help="Environment stepping rate in Hz.")
    parser.add_argument("--num_episode", type=int, default=20, help="Max episodes to record.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record",
        help="Root directory for saving datasets.",
    )
    parser.add_argument("--disable_depth", action="store_true", default=False, help="Disable depth observation.")
    parser.add_argument("--enable_pointcloud", action="store_true", default=False, help="Enable pointcloud observation.")
    parser.add_argument(
        "--task_description",
        type=str,
        default="",
        help="Clear text description of the task objective.",
    )
    parser.add_argument("--record_ee_pose", action="store_true", default=False, help="Record EE pose online.")
    parser.add_argument("--ee_urdf_path", type=str, default="Assets/robots/so101_new_calib.urdf", help="URDF path for EE pose computation (single SO-101 arm).")
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint unit for kinematics (default: rad).",
    )

    return parser


def setup_replay_parser(
    subparsers: argparse.ArgumentParser,
    parent_parsers: list[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for 'replay' subcommand."""
    parser = subparsers.add_parser(
        "replay",
        help="Replay dataset",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task environment.")
    parser.add_argument("--step_hz", type=int, default=60, help="Stepping rate in Hz.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--output_root", type=str, default=None, help="Directory to save replayed episodes.")
    parser.add_argument("--num_replays", type=int, default=1, help="Number of times to replay each episode.")
    parser.add_argument("--save_successful_only", action="store_true", default=False, help="Only save successful replays.")
    parser.add_argument("--start_episode", type=int, default=0, help="Starting episode index.")
    parser.add_argument("--end_episode", type=int, default=None, help="Ending episode index.")
    parser.add_argument("--use_random_seed", action="store_true", default=False, help="Use random seed.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument(
        "--task_description",
        type=str,
        default="",
        help="Description of the task objective.",
    )
    parser.add_argument("--use_ee_pose", action="store_true", default=False, help="Use Cartesian control via IK.")
    parser.add_argument("--ee_urdf_path", type=str, default="Assets/robots/so101_new_calib.urdf", help="URDF file path for IK solver (single SO-101 arm).")
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint unit for IK solver.",
    )
    parser.add_argument("--disable_depth", action="store_true", default=False, help="Disable depth observation.")

    return parser


def setup_inspect_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'inspect' subcommand."""
    parser = subparsers.add_parser("inspect", help="Inspect dataset metadata")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--show_frames", type=int, default=None, help="Display first N frames")
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_read_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'read' subcommand."""
    parser = subparsers.add_parser("read", help="Read dataset states")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to read")
    parser.add_argument("--episode", type=int, default=None, help="Specific episode index")
    parser.add_argument("--output_csv", type=str, default=None, help="Export to CSV")
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_augment_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'augment' subcommand."""
    parser = subparsers.add_parser("augment", help="Add EE pose to dataset")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--urdf_path", type=str, required=True, help="URDF file path")
    parser.add_argument("--state_unit", type=str, default="rad", choices=["rad", "deg"], help="Joint angle unit")
    parser.add_argument("--output_root", type=str, default=None, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing ee_pose columns and overwrite existing output files")
    return parser


def setup_merge_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'merge' subcommand."""
    parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    parser.add_argument("--source_roots", type=str, required=True, help="List of source directories (Python list string)")
    parser.add_argument("--output_root", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--output_repo_id", type=str, default="merged_dataset", help="Repository ID")
    parser.add_argument("--merge_custom_meta", action="store_true", default=True, help="Merge episode_metadata.json files")
    return parser


def setup_eval_parser() -> argparse.ArgumentParser:
    """Setup parser for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate policy in LeHome manipulation environments.")

    # Core arguments
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Episodes per variant.")
    parser.add_argument("--step_hz", type=int, default=120, help="Stepping rate in Hz.")

    # Evaluation parameters
    parser.add_argument("--use_random_seed", action="store_true", default=False, help="Use random seed.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
    parser.add_argument("--task_description", type=str, default="", help="Task goal for VLA models.")

    # Record parameters
    parser.add_argument("--save_video", action="store_true", help="Save evaluation videos.")
    parser.add_argument("--video_dir", type=str, default="outputs/eval_videos", help="Directory for videos.")
    parser.add_argument("--save_datasets", action="store_true", help="Save success episodes as dataset.")
    parser.add_argument("--eval_dataset_path", type=str, default="Datasets/eval", help="Path for saving evaluation data.")

    # Policy arguments
    parser.add_argument("--policy_type", type=str, default="lerobot", help="Type of policy (e.g., 'lerobot').")
    parser.add_argument("--policy_path", type=str, default=None, help="Path to pretrained policy checkpoint.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path of training dataset (for metadata).")
    parser.add_argument("--use_ee_pose", action="store_true", help="Policy outputs EE poses; IK will be used.")
    parser.add_argument("--ee_urdf_path", type=str, default="Assets/robots/so101_new_calib.urdf", help="URDF path for IK solver (single SO-101 arm).")

    return parser
