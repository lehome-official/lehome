import torch
from typing import Any

import isaaclab.envs.mdp as mdp

# Lekiwi joint limits.
LEKIWI_JOINT_LIMITS = {
    # Mobile base omni wheels are continuous in teleoperation.
    "wheel_1": (-3.14159, 3.14159),
    "wheel_2": (-3.14159, 3.14159),
    "wheel_3": (-3.14159, 3.14159),
    # Arm and gripper joints.
    "shoulder_pitch": (-1.91986, 1.91986),
    "shoulder_roll": (-3.31533, 0.17533),
    "elbow": (-0.00080, 3.31533),
    "wrist_pitch": (-0.08806, 3.22806),
    "wrist_roll": (-1.22253, 4.36253),
    "gripper": (-1.74533, 0.17453),
}

LEKIWI_CONTROLLED_JOINTS = {
    # Mobile-base wheels are continuous joints and are not clamped here.
    "STS3215_03a_v1_Revolute_45": "shoulder_pitch",
    "STS3215_03a_v1_1_Revolute_49": "shoulder_roll",
    "STS3215_03a_v1_2_Revolute_51": "elbow",
    "STS3215_03a_v1_3_Revolute_53": "wrist_pitch",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": "wrist_roll",
    "STS3215_03a_v1_4_Revolute_57": "gripper",
}


def clamp_lekiwi_joint_targets(
    joint_pos: torch.Tensor,
    joint_names: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Clamp lekiwi arm/gripper targets by runtime joint name."""
    squeeze = False
    if joint_pos.dim() == 1:
        joint_pos = joint_pos.unsqueeze(0)
        squeeze = True

    for usd_joint_name, limit_name in LEKIWI_CONTROLLED_JOINTS.items():
        if usd_joint_name not in joint_names:
            continue
        joint_idx = joint_names.index(usd_joint_name)
        lower, upper = LEKIWI_JOINT_LIMITS[limit_name]
        joint_pos[:, joint_idx] = torch.clamp(
            joint_pos[:, joint_idx],
            min=float(lower),
            max=float(upper),
        )

    return joint_pos.squeeze(0) if squeeze else joint_pos


def init_lekiwi_action_cfg(action_cfg, device):
    """Initialize Lekiwi action configuration."""
    if device in ['keyboard', 'lekiwi_keyboard']:
        # Keyboard control uses relative joint-position actions.
        action_cfg.base_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "ST3215_Servo_Motor_v1_2_Revolute_60",
                "ST3215_Servo_Motor_v1_1_Revolute_62",
                "ST3215_Servo_Motor_v1_Revolute_64"
            ],
            scale=2.0,
        )
        action_cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "STS3215_03a_v1_Revolute_45",
                "STS3215_03a_v1_1_Revolute_49",
                "STS3215_03a_v1_2_Revolute_51",
                "STS3215_03a_v1_3_Revolute_53",
                "STS3215_03a_Wrist_Roll_v1_Revolute_55",
            ],
            scale=3.0,
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["STS3215_03a_v1_4_Revolute_57"],
            scale=2.0,
        )
    elif device in ['hybrid', 'lekiwi_hybrid']:
        # Hybrid control: relative keyboard base, absolute leader arm.
        action_cfg.base_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "ST3215_Servo_Motor_v1_2_Revolute_60",
                "ST3215_Servo_Motor_v1_1_Revolute_62",
                "ST3215_Servo_Motor_v1_Revolute_64"
            ],
            scale=2.0,
        )
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "STS3215_03a_v1_Revolute_45",
                "STS3215_03a_v1_1_Revolute_49",
                "STS3215_03a_v1_2_Revolute_51",
                "STS3215_03a_v1_3_Revolute_53",
                "STS3215_03a_Wrist_Roll_v1_Revolute_55",
            ],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["STS3215_03a_v1_4_Revolute_57"],
            scale=1.0,
        )
    else:
        action_cfg.base_action = None
        action_cfg.arm_action = None
        action_cfg.gripper_action = None

    return action_cfg


# Joint-name to action-index mapping.
lekiwi_joint_names_to_motor_ids = {
    # Mobile base (0-2).
    "ST3215_Servo_Motor_v1_2_Revolute_60": 0,  # wheel_1
    "ST3215_Servo_Motor_v1_1_Revolute_62": 1,  # wheel_2
    "ST3215_Servo_Motor_v1_Revolute_64": 2,  # wheel_3

    # Arm (3-7).
    "STS3215_03a_v1_Revolute_45": 3,  # shoulder_pitch
    "STS3215_03a_v1_1_Revolute_49": 4,  # shoulder_roll
    "STS3215_03a_v1_2_Revolute_51": 5,  # elbow
    "STS3215_03a_v1_3_Revolute_53": 6,  # wrist_pitch
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": 7,  # wrist_roll

    # Gripper (8).
    "STS3215_03a_v1_4_Revolute_57": 8,  # gripper
}


def preprocess_lekiwi_device_action(
    action: dict[str, Any], teleop_device
) -> torch.Tensor:
    """Preprocess Lekiwi device actions."""
    if (action.get('keyboard') is not None or
            action.get('lekiwi_keyboard') is not None):
        # Keyboard actions already match the environment action layout.
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 9,
            device=teleop_device.env.device
        )
        processed_action[:, :] = action['joint_state']
    elif action.get('lekiwi_hybrid') is not None:
        # Hybrid actions already match the environment action layout.
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 9,
            device=teleop_device.env.device
        )
        processed_action[:, :] = action['joint_state']
    else:
        raise NotImplementedError(
            "Only keyboard and hybrid teleoperation are supported "
            "for lekiwi."
        )

    return processed_action


def get_lekiwi_action_space_size():
    """Return the Lekiwi action-space size."""
    return 9  # 3 wheels + 5 arm joints + 1 gripper


def get_lekiwi_joint_names():
    """Return all Lekiwi joint names."""
    return list(lekiwi_joint_names_to_motor_ids.keys())


def get_lekiwi_joint_limits():
    """Return Lekiwi joint limits."""
    return LEKIWI_JOINT_LIMITS.copy()
