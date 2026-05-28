from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from lehome.utils.constant import ASSETS_ROOT

# Lekiwi USD asset path.
LEKIWI_ASSET_PATH = Path(ASSETS_ROOT) / "lekiwi" / "lekiwi_final_best.usd"

# Friendly joint-name aliases for the verbose USD joint names.
LEKIWI_JOINT_MAPPING = {
    # Mobile base omni wheels.
    "wheel_1": "ST3215_Servo_Motor_v1_2_Revolute_60",
    "wheel_2": "ST3215_Servo_Motor_v1_1_Revolute_62",
    "wheel_3": "ST3215_Servo_Motor_v1_Revolute_64",
    # Arm and gripper joints.
    "shoulder_pitch": "STS3215_03a_v1_Revolute_45",
    "shoulder_roll": "STS3215_03a_v1_1_Revolute_49",
    "elbow": "STS3215_03a_v1_2_Revolute_51",
    "wrist_pitch": "STS3215_03a_v1_3_Revolute_53",
    "wrist_roll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "gripper": "STS3215_03a_v1_4_Revolute_57",
}

# Reverse lookup from USD joint names to friendly aliases.
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
            # Mobile base omni wheels.
            "ST3215_Servo_Motor_v1_2_Revolute_60": 0.0,
            "ST3215_Servo_Motor_v1_1_Revolute_62": 0.0,
            "ST3215_Servo_Motor_v1_Revolute_64": 0.0,
            # Arm and gripper joints.
            "STS3215_03a_v1_Revolute_45": 0.0,
            "STS3215_03a_v1_1_Revolute_49": 0.0,
            "STS3215_03a_v1_2_Revolute_51": 0.0,
            "STS3215_03a_v1_3_Revolute_53": 0.0,
            "STS3215_03a_Wrist_Roll_v1_Revolute_55": 0.0,
            "STS3215_03a_v1_4_Revolute_57": 0.0,
        }
    ),
    actuators={
        # Mobile base omni-wheel actuators.
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
        # Arm joint actuators.
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
        # Gripper actuator.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["STS3215_03a_v1_4_Revolute_57"],
            effort_limit_sim=50,
            velocity_limit_sim=5.0,
            stiffness=1500.0,
            damping=50.0,
        ),
    },
)

# Joint limits, aligned with the SO101 arm calibration and USD model.
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

# Nominal motor command limits.
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
