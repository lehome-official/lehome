from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
#移除这行：from isaaclab.sim.spawners.from_files import UrdfFileCfg
from lehome.utils.constant import ASSETS_ROOT

# 修正文件路径
XLEROBOT_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "xlerobot" / "xlerobot" / "xlerobot_final_best.usd"

XLEROBOT_CFG = ArticulationCfg(
    # 修正prim_path - 根据USD文件结构，应该是xlerobot作为根节点
    prim_path="/World/Robot",  # 正确的articulation根节点路径
    # 使用UsdFileCfg
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
            # 提高接触求解迭代，降低“接触抖动/弹飞”概率。
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            # fix_root_link=False,  # 移动底盘需要设置为False
        ),
    ),
    # 其余配置保持不变
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.0, -2.0, 0.0),  
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # 恢复所有原始关节
            "root_x_axis_joint": 0.0,
            "root_y_axis_joint": 0.0,
            "root_z_rotation_joint": 0.0,
            # 左臂关节
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 0.0,
            # 右臂关节
            "Rotation_2": 0.0,
            "Pitch_2": 0.0,
            "Elbow_2": 0.0,
            "Wrist_Pitch_2": 0.0,
            "Wrist_Roll_2": 0.0,
            "Jaw_2": 0.0,
            # 头部关节
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
        }
    ),
    actuators={
        # 恢复移动底盘执行器，包含所有3个关节
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

# 关节限制（从URDF提取）
XLEROBOT_JOINT_LIMITS = {
    # 移动底盘关节
    "root_x_axis_joint": (-20.0, 20.0),      # X轴平移
    "root_y_axis_joint": (-20.0, 20.0),      # Y轴平移  
    "root_z_rotation_joint": (-3.14159, 3.14159),  # Z轴旋转
    
    # 左臂关节
    "Rotation": (-2.1, 2.1),                 # 肩部旋转
    "Pitch": (-0.1, 3.45),                   # 肩部抬升
    "Elbow": (-0.2, 3.14159),                # 肘部弯曲
    "Wrist_Pitch": (-1.8, 1.8),              # 腕部弯曲
    "Wrist_Roll": (-3.14159, 3.14159),       # 腕部旋转
    "Jaw": (0.0, 1.7),                       # 左臂夹爪（与URDF一致）
    
    # 右臂关节
    "Rotation_2": (-2.1, 2.1),               # 右肩部旋转
    "Pitch_2": (-0.1, 3.45),                 # 右肩部抬升
    "Elbow_2": (-0.2, 3.14159),              # 右肘部弯曲
    "Wrist_Pitch_2": (-1.8, 1.8),            # 右腕部弯曲
    "Wrist_Roll_2": (-3.14159, 3.14159),     # 右腕部旋转
    "Jaw_2": (0.0, 1.7),                     # 右臂夹爪（与URDF一致）
    
    # 头部关节
    "head_pan_joint": (-1.57, 1.57),         # 头部水平旋转
    "head_tilt_joint": (-0.76, 1.45),        # 头部垂直倾斜
}

# 电机限制（需要根据实际硬件调整）
XLEROBOT_MOTOR_LIMITS = {
    # 移动底盘
    "root_x_axis_joint": (-100.0, 100.0),
    "root_y_axis_joint": (-100.0, 100.0),
    "root_z_rotation_joint": (-100.0, 100.0),
    # 左臂
    "Rotation": (-100.0, 100.0),
    "Pitch": (-100.0, 100.0),
    "Elbow": (-100.0, 100.0),
    "Wrist_Pitch": (-100.0, 100.0),
    "Wrist_Roll": (-100.0, 100.0),
    "Jaw": (0.0, 100.0),
    # 右臂
    "Rotation_2": (-100.0, 100.0),
    "Pitch_2": (-100.0, 100.0),
    "Elbow_2": (-100.0, 100.0),
    "Wrist_Pitch_2": (-100.0, 100.0),
    "Wrist_Roll_2": (-100.0, 100.0),
    "Jaw_2": (0.0, 100.0),
    # 头部
    "head_pan_joint": (-100.0, 100.0),
    "head_tilt_joint": (-100.0, 100.0),
}
