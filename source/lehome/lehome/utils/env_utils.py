import os
import torch


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device):
    # xlerobot默认关闭“按最近物体质量动态改夹爪力矩”，
    # 避免抓轻小刚体（如cube）时力矩被压得过低导致“碰到了但抓不住”。
    if teleop_device == "xlerobot" and os.getenv("LEHOME_XLEROBOT_DYNAMIC_GRIPPER_EFFORT", "0") != "1":
        return

    need_to_set = []
    if teleop_device in ["bi-so101leader", "bi-keyboard"]:
        need_to_set = [
            env.scene.articulations["left_arm"],
            env.scene.articulations["right_arm"],
        ]
    elif teleop_device in ["so101leader", "keyboard", "xlerobot", "lekiwi_keyboard", "lekiwi_hybrid"]:
        need_to_set = [env.scene["robot"]]

    for arm in need_to_set:
        write_gripper_effort_limit_sim(env, arm)


def write_gripper_effort_limit_sim(env, env_arm):
    gripper_pos = env_arm.data.body_link_pos_w[:, -1]  # [num_envs, 3]
    num_envs = gripper_pos.shape[0]

    object_positions = []
    object_masses = []

    for _, obj in env.scene._rigid_objects.items():
        object_positions.append(obj.data.body_link_pos_w[:, 0])  # [num_envs, 3]
        object_masses.append(obj.data.default_mass)  # [num_envs, 1]

    if not object_positions:
        return

    object_positions = torch.stack(object_positions)  # [num_objects, num_envs, 3]
    object_masses = torch.stack(object_masses)  # [num_objects, num_envs, 1]
    distances = torch.sqrt(torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))
    _, min_indices = torch.min(distances, dim=0)  # [num_envs]
    target_masses = object_masses[min_indices.cpu(), 0, 0]  # [num_envs]
    target_effort_limits = (target_masses / 0.15).to(env_arm._data.joint_effort_limits.device)

    # 夹爪力矩保护范围，避免过低夹不住或过高导致抖动。
    effort_min = float(os.getenv("LEHOME_GRIPPER_EFFORT_MIN", "5.0"))
    effort_max = float(os.getenv("LEHOME_GRIPPER_EFFORT_MAX", "200.0"))
    if effort_max < effort_min:
        effort_max = effort_min
    target_effort_limits = torch.clamp(target_effort_limits, min=effort_min, max=effort_max)

    gripper_joint_id = _infer_gripper_joint_id(env_arm)
    current_effort_limit_sim = env_arm._data.joint_effort_limits[:, gripper_joint_id]
    need_update = torch.abs(target_effort_limits - current_effort_limit_sim) > 0.1

    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]
        env_arm.write_joint_effort_limit_to_sim(
            limits=new_limits,
            joint_ids=[gripper_joint_id for _ in range(num_envs)],
        )


def _infer_gripper_joint_id(env_arm) -> int:
    """Infer gripper joint index from joint names, fallback to last joint."""
    joint_names = [str(name).lower() for name in env_arm.data.joint_names]
    for key in ("gripper", "jaw", "finger"):
        for idx, name in enumerate(joint_names):
            if key in name:
                return idx
    return len(joint_names) - 1


def get_task_type(task: str) -> str:
    """
    Make sure the task type is in the supported teleop devices.
    """
    if "BiArm" in task:
        return "bi-so101leader"
    if "Xlerobot" in task:
        return "xlerobot"
    if "Lekiwi" in task:
        return "lekiwi_keyboard"
    return "so101leader"
