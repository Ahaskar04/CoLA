# APPROACH and PICKUP phase 
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from mink import SO3

def compensate_gravity(model, data, subtree_ids):
    qfrc_applied = data.qfrc_applied
    qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac

def check_gripper_box_contact(model, data, box_geom_name="middle_box_geom"):
    finger_geom_names = {"left/left_g0", "left/left_g1", "left/left_g2",
                          "left/right_g0", "left/right_g1", "left/right_g2"}
    box_geom_id = model.geom(box_geom_name).id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = model.geom(contact.geom1).name
        geom2_name = model.geom(contact.geom2).name
        if (contact.geom1 == box_geom_id and geom2_name in finger_geom_names) or \
           (contact.geom2 == box_geom_id and geom1_name in finger_geom_names):
            return True
    return False

def check_gripper_box_contact_right(model, data, box_geom_name="middle_box_geom"):
    # Both fingers, mirroring check_gripper_box_contact above. The right-finger
    # names were previously listed twice and the left-finger ones omitted, so a
    # box seated against arm B's LEFT finger was held but never detected: grip_b
    # waits on this predicate with no timeout, so those episodes froze mid-
    # handover until max_steps. It cost 31/200 episodes in the first v3, all of
    # them spawns in x ~ [-0.01, +0.04] -- exactly where RIGHT_TARGET_OFFSET
    # (+0.03 in x) seats the box against the unlisted finger.
    finger_geom_names = {"right/left_g0", "right/left_g1", "right/left_g2",
                          "right/right_g0", "right/right_g1", "right/right_g2"}
    box_geom_id = model.geom(box_geom_name).id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = model.geom(contact.geom1).name
        geom2_name = model.geom(contact.geom2).name
        if (contact.geom1 == box_geom_id and geom2_name in finger_geom_names) or \
           (contact.geom2 == box_geom_id and geom1_name in finger_geom_names):
            return True
    return False

def check_gripper_box_contact_third(model, data, box_geom_name="middle_box_geom"):
    # BOTH finger sets, deliberately. The right-arm version of this function
    # once listed the right-finger geoms twice and omitted the left ones, so a
    # box seated against the unlisted finger was held but never DETECTED --
    # grip_b waits on the predicate with no timeout, so those episodes froze
    # until max_steps. It cost 31/200 episodes before it was caught.
    finger_geom_names = {"third/left_g0", "third/left_g1", "third/left_g2",
                          "third/right_g0", "third/right_g1", "third/right_g2"}
    box_geom_id = model.geom(box_geom_name).id

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = model.geom(contact.geom1).name
        geom2_name = model.geom(contact.geom2).name
        if (contact.geom1 == box_geom_id and geom2_name in finger_geom_names) or \
           (contact.geom2 == box_geom_id and geom1_name in finger_geom_names):
            return True
    return False

def setup_dual_arm_ik(xml_path):
    # Name kept for compatibility: every caller imports setup_dual_arm_ik, and
    # it now builds THREE arms when the scene provides them.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    left_ee_task = mink.FrameTask(
        frame_name="left/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
        gain = 0.05
    )

    right_ee_task = mink.FrameTask(
        frame_name="right/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
        gain = 0.05
    )
    third_ee_task = mink.FrameTask(
        frame_name="third/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
        gain = 0.05
    )
    posture_task = mink.PostureTask(model, cost=1e-4)
    # third_ee_task must be in this list or mink never solves for arm C: the
    # phases would write third/target and nothing would follow it.
    tasks = [left_ee_task, right_ee_task, third_ee_task, posture_task]

    left_joint_names = [
        f"left/{n}" for n in
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ]

    right_joint_names = [
        f"right/{n}" for n in
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ]

    third_joint_names = [
        f"third/{n}" for n in
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ]

    # Arm C must be in all_joint_names or its velocity limit is never applied and
    # it moves arbitrarily fast while the other two are capped.
    all_joint_names = left_joint_names + right_joint_names + third_joint_names
    velocity_limits = {name: 1.0 for name in all_joint_names}  # radians/sec, tune this number
    limits = [mink.VelocityLimit(model, velocity_limits)]

    left_dof_ids = np.array([model.joint(name).id for name in left_joint_names])
    left_actuator_ids = np.array([model.actuator(name).id for name in left_joint_names])

    right_dof_ids = np.array([model.joint(name).id for name in right_joint_names])
    right_actuator_ids = np.array([model.actuator(name).id for name in right_joint_names])

    third_dof_ids = np.array([model.joint(name).id for name in third_joint_names])
    third_actuator_ids = np.array([model.actuator(name).id for name in third_joint_names])

    solver = "daqp"
    left_mocap_id = model.body("left/target").mocapid[0]
    left_gripper_actuator_id = model.actuator("left/gripper").id
    right_mocap_id = model.body("right/target").mocapid[0]
    right_gripper_actuator_id = model.actuator("right/gripper").id
    third_mocap_id = model.body("third/target").mocapid[0]
    third_gripper_actuator_id = model.actuator("third/gripper").id
    GRIPPER_OPEN = 0.037
    GRIPPER_CLOSED = 0.002
    left_subtree_id = model.body("left/base_link").id
    right_subtree_id = model.body("right/base_link").id
    # Same gravity compensation as the other two: without it arm C sags under
    # its own weight while A and B hold their pose.
    third_subtree_id = model.body("third/base_link").id

    # --- newly added: task-specific constants, previously hardcoded in the main script ---
    GRASP_ORIENTATION_MATRIX = np.array([
        [ 0.217446,  0.0,        0.976072],
        [ 0.0,       1.0,        0.0     ],
        [-0.976072,  0.0,        0.217446],
    ])
    GRASP_ORIENTATION = SO3.from_matrix(GRASP_ORIENTATION_MATRIX)
    LIFT_HEIGHT = 0.25
    box_joint_id = model.joint('middle_box_joint').qposadr[0]
    box_spawn_height = 0.03

    return {
        "model": model,
        "data": data,
        "configuration": configuration,
        "tasks": tasks,
        "left_ee_task": left_ee_task,
        "right_ee_task": right_ee_task,
        "third_ee_task": third_ee_task,
        "posture_task": posture_task,
        "left_dof_ids": left_dof_ids,
        "left_actuator_ids": left_actuator_ids,
        "right_dof_ids": right_dof_ids,
        "right_actuator_ids": right_actuator_ids,
        "third_dof_ids": third_dof_ids,
        "third_actuator_ids": third_actuator_ids,
        "limits": limits,
        "solver": solver,
        "left_mocap_id": left_mocap_id,
        "left_gripper_actuator_id": left_gripper_actuator_id,
        "right_mocap_id": right_mocap_id,
        "right_gripper_actuator_id": right_gripper_actuator_id,
        "third_mocap_id": third_mocap_id,
        "third_gripper_actuator_id": third_gripper_actuator_id,
        "GRIPPER_OPEN": GRIPPER_OPEN,
        "GRIPPER_CLOSED": GRIPPER_CLOSED,
        "left_subtree_id": left_subtree_id,
        "right_subtree_id": right_subtree_id,
        "third_subtree_id": third_subtree_id,
        "GRASP_ORIENTATION": GRASP_ORIENTATION,
        "LIFT_HEIGHT": LIFT_HEIGHT,
        "box_joint_id": box_joint_id,
        "box_spawn_height": box_spawn_height,
    }