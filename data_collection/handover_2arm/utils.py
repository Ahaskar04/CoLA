"""IK rig, gravity compensation, contact checks and constants for the scripted expert."""
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
    # Both finger pads.
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

def setup_dual_arm_ik(xml_path):
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
    posture_task = mink.PostureTask(model, cost=1e-4)
    tasks = [left_ee_task, right_ee_task, posture_task]

    left_joint_names = [
        f"left/{n}" for n in
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ]

    right_joint_names = [
        f"right/{n}" for n in
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ]

    all_joint_names = left_joint_names + right_joint_names
    velocity_limits = {name: 1.0 for name in all_joint_names}  # rad/s
    limits = [mink.VelocityLimit(model, velocity_limits)]

    left_dof_ids = np.array([model.joint(name).id for name in left_joint_names])
    left_actuator_ids = np.array([model.actuator(name).id for name in left_joint_names])

    right_dof_ids = np.array([model.joint(name).id for name in right_joint_names])
    right_actuator_ids = np.array([model.actuator(name).id for name in right_joint_names])

    solver = "daqp"
    left_mocap_id = model.body("left/target").mocapid[0]
    left_gripper_actuator_id = model.actuator("left/gripper").id
    right_mocap_id = model.body("right/target").mocapid[0]
    right_gripper_actuator_id = model.actuator("right/gripper").id
    GRIPPER_OPEN = 0.037
    GRIPPER_CLOSED = 0.002
    left_subtree_id = model.body("left/base_link").id
    right_subtree_id = model.body("right/base_link").id

    # Task constants.
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
        "posture_task": posture_task,
        "left_dof_ids": left_dof_ids,
        "left_actuator_ids": left_actuator_ids,
        "right_dof_ids": right_dof_ids,
        "right_actuator_ids": right_actuator_ids,
        "limits": limits,
        "solver": solver,
        "left_mocap_id": left_mocap_id,
        "left_gripper_actuator_id": left_gripper_actuator_id,
        "right_mocap_id": right_mocap_id,
        "right_gripper_actuator_id": right_gripper_actuator_id,
        "GRIPPER_OPEN": GRIPPER_OPEN,
        "GRIPPER_CLOSED": GRIPPER_CLOSED,
        "left_subtree_id": left_subtree_id,
        "right_subtree_id": right_subtree_id,
        "GRASP_ORIENTATION": GRASP_ORIENTATION,
        "LIFT_HEIGHT": LIFT_HEIGHT,
        "box_joint_id": box_joint_id,
        "box_spawn_height": box_spawn_height,
    }