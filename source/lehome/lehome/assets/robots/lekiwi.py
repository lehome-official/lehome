from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from lehome.utils.constant import ASSETS_ROOT

# Lekiwi USD文件路径
LEKIWI_ASSET_PATH = Path(ASSETS_ROOT) / "lekiwi" / "lekiwi_final_best.usd"

# 友好的关节名称映射（从URDF中的复杂名称映射到简单名称）
LEKIWI_JOINT_MAPPING = {
    # 移动底盘 (3个全向轮)
    "wheel_1": "ST3215_Servo_Motor_v1_2_Revolute_60",
    "wheel_2": "ST3215_Servo_Motor_v1_1_Revolute_62",
    "wheel_3": "ST3215_Servo_Motor_v1_Revolute_64",
    # 机械臂 (6个关节)
    "shoulder_pitch": "STS3215_03a_v1_Revolute_45",
    "shoulder_roll": "STS3215_03a_v1_1_Revolute_49",
    "elbow": "STS3215_03a_v1_2_Revolute_51",
    "wrist_pitch": "STS3215_03a_v1_3_Revolute_53",
    "wrist_roll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "gripper": "STS3215_03a_v1_4_Revolute_57",
}

# 反向映射（从URDF名称到友好名称）
LEKIWI_JOINT_MAPPING_REVERSE = {v: k for k, v in LEKIWI_JOINT_MAPPING.items()}

LEKIWI_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LEKIWI_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # 移动底盘 - 3个全向轮
            "ST3215_Servo_Motor_v1_2_Revolute_60": 0.0,
            "ST3215_Servo_Motor_v1_1_Revolute_62": 0.0,
            "ST3215_Servo_Motor_v1_Revolute_64": 0.0,
            # 机械臂 - 6个关节
            "STS3215_03a_v1_Revolute_45": 0.0,
            "STS3215_03a_v1_1_Revolute_49": 0.0,
            "STS3215_03a_v1_2_Revolute_51": 0.0,
            "STS3215_03a_v1_3_Revolute_53": 0.0,
            "STS3215_03a_Wrist_Roll_v1_Revolute_55": 0.0,
            "STS3215_03a_v1_4_Revolute_57": 0.0,
        }
    ),
    actuators={
        # 移动底盘执行器 - 3个全向轮
        "mobile_base": ImplicitActuatorCfg(
            joint_names_expr=[
                "ST3215_Servo_Motor_v1_2_Revolute_60",
                "ST3215_Servo_Motor_v1_1_Revolute_62",
                "ST3215_Servo_Motor_v1_Revolute_64"
            ],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        # 机械臂执行器 - 6个关节
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "STS3215_03a_v1_Revolute_45",
                "STS3215_03a_v1_1_Revolute_49",
                "STS3215_03a_v1_2_Revolute_51",
                "STS3215_03a_v1_3_Revolute_53",
                "STS3215_03a_Wrist_Roll_v1_Revolute_55",
            ],
            effort_limit_sim=200,
            velocity_limit_sim=10.0,
            stiffness=3000.0,
            damping=100.0,
        ),
        # 夹爪执行器
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["STS3215_03a_v1_4_Revolute_57"],
            effort_limit_sim=50,
            velocity_limit_sim=5.0,
            stiffness=1500.0,
            damping=50.0,
        ),
    },
)

# 关节限制（根据URDF和实际硬件调整）
LEKIWI_JOINT_LIMITS = {
    # 移动底盘 - 全向轮（连续旋转）
    "wheel_1": (-3.14159, 3.14159),
    "wheel_2": (-3.14159, 3.14159),
    "wheel_3": (-3.14159, 3.14159),
    # 机械臂关节
    "shoulder_pitch": (-1.91986, 1.91986),
    "shoulder_roll": (-3.31533, 0.17533),
    "elbow": (-0.00080, 3.31533),
    "wrist_pitch": (-0.08806, 3.22806),
    "wrist_roll": (-1.22253, 4.36253),
    "gripper": (-1.74533, 0.17453),
}

# 电机限制（需要根据实际硬件调整）
LEKIWI_MOTOR_LIMITS = {
    "wheel_1": (-100.0, 100.0),
    "wheel_2": (-100.0, 100.0),
    "wheel_3": (-100.0, 100.0),
    "shoulder_pitch": (-100.0, 100.0),
    "shoulder_roll": (-100.0, 100.0),
    "elbow": (-100.0, 100.0),
    "wrist_pitch": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}
