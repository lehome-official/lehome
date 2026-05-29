from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from lehome.utils.constant import ASSETS_ROOT

XLEROBOT_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "xlerobot" / "xlerobot" / "xlerobot_final_best.usd"

XLEROBOT_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(XLEROBOT_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Keep self-collision disabled for the release teleop path.
            # The current xlerobot USD has dense arm/gripper collisions that can
            # stall PhysX startup when whole-body self-collision is enabled.
            enabled_self_collisions=False,
            # Use higher solver iterations to improve contact stability.
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.0, -2.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # Mobile base joints.
            "root_x_axis_joint": 0.0,
            "root_y_axis_joint": 0.0,
            "root_z_rotation_joint": 0.0,
            # Left arm joints.
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 0.0,
            # Right arm joints.
            "Rotation_2": 0.0,
            "Pitch_2": 0.0,
            "Elbow_2": 0.0,
            "Wrist_Pitch_2": 0.0,
            "Wrist_Roll_2": 0.0,
            "Jaw_2": 0.0,
            # Head joints.
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
        }
    ),
    actuators={
        "mobile_base": ImplicitActuatorCfg(
            joint_names_expr=["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"],
            effort_limit_sim=5000,
            velocity_limit_sim=2.0,
            stiffness=1200.0,
            damping=300.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            effort_limit_sim=200,
            velocity_limit_sim=10.0,
            stiffness=3000.0,
            damping=100.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=50,
            velocity_limit_sim=5.0,
            stiffness=1500.0,
            damping=50.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            effort_limit_sim=200,
            velocity_limit_sim=10.0,
            stiffness=3000.0,
            damping=100.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw_2"],
            effort_limit_sim=50,
            velocity_limit_sim=5.0,
            stiffness=1500.0,
            damping=50.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan_joint", "head_tilt_joint"],
            effort_limit_sim=30,
            velocity_limit_sim=5.0,
            stiffness=1500.0,
            damping=50.0,
        ),
    },
)

# Joint limits extracted from the source robot model.
XLEROBOT_JOINT_LIMITS = {
    # Mobile base joints.
    "root_x_axis_joint": (-20.0, 20.0),
    "root_y_axis_joint": (-20.0, 20.0),
    "root_z_rotation_joint": (-3.14159, 3.14159),

    # Left arm joints.
    "Rotation": (-2.1, 2.1),
    "Pitch": (-0.1, 3.45),
    "Elbow": (-0.2, 3.14159),
    "Wrist_Pitch": (-1.8, 1.8),
    "Wrist_Roll": (-3.14159, 3.14159),
    "Jaw": (0.0, 1.7),

    # Right arm joints.
    "Rotation_2": (-2.1, 2.1),
    "Pitch_2": (-0.1, 3.45),
    "Elbow_2": (-0.2, 3.14159),
    "Wrist_Pitch_2": (-1.8, 1.8),
    "Wrist_Roll_2": (-3.14159, 3.14159),
    "Jaw_2": (0.0, 1.7),

    # Head joints.
    "head_pan_joint": (-1.57, 1.57),
    "head_tilt_joint": (-0.76, 1.45),
}

# Nominal motor command limits.
XLEROBOT_MOTOR_LIMITS = {
    # Mobile base.
    "root_x_axis_joint": (-100.0, 100.0),
    "root_y_axis_joint": (-100.0, 100.0),
    "root_z_rotation_joint": (-100.0, 100.0),
    # Left arm.
    "Rotation": (-100.0, 100.0),
    "Pitch": (-100.0, 100.0),
    "Elbow": (-100.0, 100.0),
    "Wrist_Pitch": (-100.0, 100.0),
    "Wrist_Roll": (-100.0, 100.0),
    "Jaw": (0.0, 100.0),
    # Right arm.
    "Rotation_2": (-100.0, 100.0),
    "Pitch_2": (-100.0, 100.0),
    "Elbow_2": (-100.0, 100.0),
    "Wrist_Pitch_2": (-100.0, 100.0),
    "Wrist_Roll_2": (-100.0, 100.0),
    "Jaw_2": (0.0, 100.0),
    # Head.
    "head_pan_joint": (-100.0, 100.0),
    "head_tilt_joint": (-100.0, 100.0),
}
