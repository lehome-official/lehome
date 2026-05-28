import os
import torch
from typing import Any

import isaaclab.envs.mdp as mdp
from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
from lehome.assets.robots.xlerobot import XLEROBOT_JOINT_LIMITS as XLEROBOT_ASSET_JOINT_LIMITS

# 统一使用机器人资产配置中的关节限位，避免多处定义不一致。
XLEROBOT_JOINT_LIMITS = XLEROBOT_ASSET_JOINT_LIMITS


def init_xlerobot_action_cfg(action_cfg, device):
    """初始化xlerobot的动作配置"""
    if device in ['keyboard']:
        # 键盘控制 - 移除底盘动作配置，使用直接位置控制
        # 注释掉或删除base_action配置
        # action_cfg.base_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
        #     scale=2.0,
        # )
        
        # 只保留手臂和头部的动作配置
        action_cfg.left_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=3.0,
        )
        action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=2.0,
        )
        action_cfg.right_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=3.0,
        )
        action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=2.0,
        )
        action_cfg.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=2.0,
        )
        
        # 清空base_action，避免冲突
        action_cfg.base_action = None
    elif device in ["hybrid", "arms_only"]:
        # xlerobot direct env在环境内部自行处理15维动作向量；
        # 这里显式关闭manager动作项，避免“未定义分支”语义不清晰。
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
        
    elif device in ['xlerobot_leader']:
        # 物理设备控制 - 使用绝对位置控制
        action_cfg.base_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
            scale=1.0,
        )
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=1.0,
        )
        action_cfg.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=1.0,
        )
    elif device in ['xbox', 'gamepad']:
        # Xbox 控制 - 使用统一的动作空间
        action_cfg.unified_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "root_x_axis_joint", "root_z_rotation_joint",  # 底盘
                "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",  # 左臂
                "Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2",  # 右臂
                "head_pan_joint", "head_tilt_joint"  # 头部
            ],
            scale=1.0,
        )
        # 清空其他动作配置
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    else:
        # 默认配置
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    
    return action_cfg


# 更新关节映射索引
xlerobot_joint_names_to_motor_ids = {
    # 旋转关节 (0)
    "root_z_rotation_joint": 0,
    
    # 左臂 (1-5)
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6,  # 左夹爪
    
    # 右臂 (7-11)
    "Rotation_2": 7,
    "Pitch_2": 8,
    "Elbow_2": 9,
    "Wrist_Pitch_2": 10,
    "Wrist_Roll_2": 11,
    "Jaw_2": 12,  # 右夹爪
    
    # 头部 (13-14)
    "head_pan_joint": 13,
    "head_tilt_joint": 14,
}


_SO101_ARM_JOINT_TO_INDEX = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}

_SO101_ARM_DIRECTION_CORRECTION = {
    "shoulder_pan": 1,
    "shoulder_lift": -1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_roll": -1,
    "gripper": 1,
}

_SO101_ARM_ZERO_OFFSET = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.57,
    "elbow_flex": 1.57,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def _is_lerobot_compat_enabled() -> bool:
    """Whether to align SO101->xlerobot arm mapping with lerobot behavior."""
    return os.getenv("LEHOME_XLEROBOT_LEROBOT_COMPAT", "0") == "1"


# lerobot兼容模式下仍需保留的关键几何对齐（与历史可用控制对齐）。
_LEROBOT_COMPAT_DIRECTION_CORRECTION = {
    "shoulder_pan": 1,
    "shoulder_lift": -1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_roll": -1,
    "gripper": 1,
}

_LEROBOT_COMPAT_ZERO_OFFSET = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.57,
    "elbow_flex": 1.57,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def _convert_so101_arm_to_xlerobot_arm(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    teleop_device,
) -> torch.Tensor:
    """Convert a single SO101 arm state into xlerobot arm action [6]."""
    processed_action = torch.zeros(
        teleop_device.env.num_envs, 6, device=teleop_device.env.device
    )
    if not joint_state or not motor_limits:
        return processed_action

    lerobot_compat = _is_lerobot_compat_enabled()
    for joint_name, index in _SO101_ARM_JOINT_TO_INDEX.items():
        if joint_name in joint_state and joint_name in motor_limits:
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]

            processed_degree = (
                (joint_state[joint_name] - motor_limit_range[0])
                / (motor_limit_range[1] - motor_limit_range[0])
                * (joint_limit_range[1] - joint_limit_range[0])
                + joint_limit_range[0]
            )
            processed_radius = processed_degree / 180.0 * torch.pi
            if lerobot_compat:
                # lerobot兼容 + xlerobot几何对齐：保留必要方向/零位修正，和历史控制行为对齐。
                processed_action[:, index] = (
                    processed_radius * _LEROBOT_COMPAT_DIRECTION_CORRECTION[joint_name]
                    + _LEROBOT_COMPAT_ZERO_OFFSET[joint_name]
                )
            else:
                processed_action[:, index] = (
                    processed_radius * _SO101_ARM_DIRECTION_CORRECTION[joint_name]
                    + _SO101_ARM_ZERO_OFFSET[joint_name]
                )

    # 安全夹紧：确保SO101映射后不越过xlerobot机械臂/夹爪真实关节限位。
    arm_joint_order = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    lower = torch.tensor(
        [XLEROBOT_JOINT_LIMITS[name][0] for name in arm_joint_order],
        device=teleop_device.env.device,
        dtype=processed_action.dtype,
    )
    upper = torch.tensor(
        [XLEROBOT_JOINT_LIMITS[name][1] for name in arm_joint_order],
        device=teleop_device.env.device,
        dtype=processed_action.dtype,
    )
    processed_action = torch.max(torch.min(processed_action, upper), lower)
    return processed_action


def convert_action_from_xlerobot_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    """从xlerobot leader设备转换动作"""
    processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
    
    for joint_name, motor_id in xlerobot_joint_names_to_motor_ids.items():
        if joint_name in joint_state and joint_name in motor_limits:
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = XLEROBOT_JOINT_LIMITS[joint_name]
            
            # 将设备范围映射到关节范围
            processed_value = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]

            # XLEROBOT_JOINT_LIMITS 已使用仿真关节单位（旋转为rad，平移为m），这里不再二次角度转换。
            processed_action[:, motor_id] = processed_value
    
    return processed_action


def preprocess_xlerobot_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    """预处理xlerobot设备动作"""
    if action.get('hybrid_controller') is not None:
        # 混合控制器动作处理
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xlerobot_leader') is not None or action.get('so101_leader') is not None:
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 15, device=teleop_device.env.device
        )
        joint_state = action.get('joint_state')
        motor_limits = action.get('motor_limits', {})
        if isinstance(joint_state, dict):
            if "shoulder_pan" in joint_state:
                processed_action[:, 1:7] = _convert_so101_arm_to_xlerobot_arm(
                    joint_state, motor_limits, teleop_device
                )
            else:
                processed_action = convert_action_from_xlerobot_leader(
                    joint_state, motor_limits, teleop_device
                )
        else:
            processed_action[:, :] = joint_state
    elif action.get('keyboard') is not None:
        # 键盘动作直接使用
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xbox') is not None:
        # Xbox 控制器动作处理 
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_xlerobot_leader') is not None or action.get('bi_so101_leader') is not None:
        # 双臂xlerobot leader设备
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 15, device=teleop_device.env.device
        )
        joint_state = action.get('joint_state', {})
        motor_limits = action.get('motor_limits', {})
        left_joint_state = joint_state.get('left_arm', {})
        right_joint_state = joint_state.get('right_arm', {})
        left_motor_limits = motor_limits.get('left_arm', {})
        right_motor_limits = motor_limits.get('right_arm', {})
        processed_action[:, 1:7] = _convert_so101_arm_to_xlerobot_arm(
            left_joint_state, left_motor_limits, teleop_device
        )
        processed_action[:, 7:13] = _convert_so101_arm_to_xlerobot_arm(
            right_joint_state, right_motor_limits, teleop_device
        )
    else:
        raise NotImplementedError("Only teleoperation with xlerobot_leader, bi_xlerobot_leader, keyboard, xbox, hybrid_controller is supported for xlerobot.")
    
    return processed_action


def get_xlerobot_action_space_size():
    """获取xlerobot动作空间大小"""
    return 15  # 1(旋转) + 5(左臂) + 1(左夹爪) + 5(右臂) + 1(右夹爪) + 2(头部)


def get_xlerobot_joint_names():
    """获取xlerobot所有关节名称"""
    return list(xlerobot_joint_names_to_motor_ids.keys())


def get_xlerobot_joint_limits():
    """获取xlerobot关节限制"""
    return XLEROBOT_JOINT_LIMITS.copy()
