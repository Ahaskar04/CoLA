"""Scripted expert for the hidden-marker handover: pick, hand over, place in the marker's tray."""

from pathlib import Path
import mujoco
import mujoco.viewer
import os
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from mink import SO3
from utils import setup_dual_arm_ik, compensate_gravity, check_gripper_box_contact, check_gripper_box_contact_right

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent  
_XML = _PROJECT_ROOT / "envs" / "handover_marker" / "scene.xml"

setup = setup_dual_arm_ik(_XML)

box_x_range = (0.0, 0.0)
box_y_range = (0.0, 0.0)


def render_camera(model, data, renderer, camera_name):
    """Render one frame from a named camera."""
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()


class EpisodeData:
    """Container for one episode's recorded data."""

    def __init__(self):
        self.image_overhead = []
        self.image_wrist_a = []
        self.image_wrist_b = []
        self.state_a = []
        self.state_b = []
        self.action_a = []
        self.action_b = []
        self.box_pos = []
        self.phase = []
        # Marker colour for the episode (needed to score correct-tray placement).
        self.marker_color = None

    def add_step(self, image_overhead, image_wrist_a, image_wrist_b,
                 state_a, state_b, action_a, action_b, box_pos, phase):
        self.image_overhead.append(image_overhead)
        self.image_wrist_a.append(image_wrist_a)
        self.image_wrist_b.append(image_wrist_b)
        self.state_a.append(state_a)
        self.state_b.append(state_b)
        self.action_a.append(action_a)
        self.action_b.append(action_b)
        self.box_pos.append(box_pos)
        self.phase.append(phase)

    def to_dict(self):
        return {
            "image_overhead": np.array(self.image_overhead),
            "image_wrist_a": np.array(self.image_wrist_a),
            "image_wrist_b": np.array(self.image_wrist_b),
            "state_a": np.array(self.state_a),
            "state_b": np.array(self.state_b),
            "action_a": np.array(self.action_a),
            "action_b": np.array(self.action_b),
            "box_pos": np.array(self.box_pos),
            "phase": np.array(self.phase, dtype="S20"),
            # Fixed-width bytes, like `phase` (h5py can't write a bare str).
            "marker_color": np.array(self.marker_color or "unknown", dtype="S10"),
        }



class ExpertState:
    """The expert's state carried between steps (phase, IK targets, mink configuration)."""

    __slots__ = ('phase', 'target_pos', 'target_pos_b', 'gripper_ctrl',
                 'right_gripper_ctrl', 'left_neutral_pos', 'left_neutral_quat',
                 'right_neutral_pos', 'right_neutral_quat', 'chosen_color')

    def __init__(self, phase, target_pos, target_pos_b, gripper_ctrl,
                 right_gripper_ctrl, left_neutral_pos, left_neutral_quat,
                 right_neutral_pos, right_neutral_quat, chosen_color=None):
        self.phase = phase
        self.target_pos = target_pos
        self.target_pos_b = target_pos_b
        self.gripper_ctrl = gripper_ctrl
        self.right_gripper_ctrl = right_gripper_ctrl
        self.left_neutral_pos = left_neutral_pos
        self.left_neutral_quat = left_neutral_quat
        self.right_neutral_pos = right_neutral_pos
        self.right_neutral_quat = right_neutral_quat
        self.chosen_color = chosen_color

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)


# Constants shared by expert_step() and run_episode().
DT = 1.0 / 200.0
APPROACH_HEIGHT_OFFSET = 0.10
POS_THRESHOLD = 0.02
B_APPROACH_HEIGHT_OFFSET = 0.08
# Offset of B's grasp target from A's gripper site.
RIGHT_TARGET_OFFSET = np.array([-0.03, 0.0, 0.03])  # 3 cm deeper toward A, 3 cm higher
# B must reach its target this closely before closing its gripper.
B_GRIP_THRESHOLD = 0.005
A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0])
# x at which arm A presents the box, near the midline between the arms.
PRESENT_X = -0.02
# Camera for the review video (COLA_REVIEW_CAMERA=wrist_cam_left shows arm A's view).
REVIEW_CAMERA = os.environ.get("COLA_REVIEW_CAMERA", "overhead_cam")

# Tray drop points (on arm B's side), one per marker colour.
TRAY_POSITIONS = {
    "blue":   np.array([0.36, -0.247, 0.20]),
    "green":  np.array([0.36,  0.0,   0.20]),
    "yellow": np.array([0.36,  0.247, 0.20]),
}


def expert_step(setup, est, sync_configuration=False):
    """One expert step: choose targets, solve IK, advance the phase.

    Returns (action_a, action_b, est), each action [6 joint targets, gripper].
    Doesn't write data.ctrl or step physics. sync_configuration=True re-seeds
    mink from data.qpos (needed if something else moved the arms). Writes the
    mocap targets that mink reads.
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]
    tasks = setup["tasks"]
    left_ee_task = setup["left_ee_task"]
    right_ee_task = setup["right_ee_task"]
    left_dof_ids = setup["left_dof_ids"]
    right_dof_ids = setup["right_dof_ids"]
    limits = setup["limits"]
    solver = setup["solver"]
    left_mocap_id = setup["left_mocap_id"]
    right_mocap_id = setup["right_mocap_id"]
    gripper_open = setup["GRIPPER_OPEN"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    GRASP_ORIENTATION = setup["GRASP_ORIENTATION"]
    LIFT_HEIGHT = setup["LIFT_HEIGHT"]

    if sync_configuration:
        configuration.update(data.qpos)

    box_pos = data.joint('middle_box_joint').qpos[:3]
    phase = est.phase

    if phase == "approach":
        est.target_pos = box_pos + A_GRASP_OFFSET + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])

    elif phase == "open_gripper":
        est.gripper_ctrl = gripper_open

    elif phase == "descend":
        est.target_pos = box_pos.copy() + A_GRASP_OFFSET

    elif phase == "grasp":
        est.gripper_ctrl = gripper_closed

    elif phase == "lift":
        est.target_pos = np.array([box_pos[0] + A_GRASP_OFFSET[0], box_pos[1], LIFT_HEIGHT])

    elif phase == "present_a":
        # Override x only; y and z keep the presentation pose.
        est.target_pos = np.array([PRESENT_X, est.target_pos[1], est.target_pos[2]])

    elif phase == "move_arm_B":
        # Stage B outward along x, so it approaches the held box from the side.
        est.target_pos_b = est.target_pos + RIGHT_TARGET_OFFSET + np.array([B_APPROACH_HEIGHT_OFFSET, 0.0, 0.0])
        data.mocap_pos[right_mocap_id] = est.target_pos_b
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
        est.right_gripper_ctrl = gripper_open

    elif phase == "approach_b":
        est.target_pos_b = est.target_pos + RIGHT_TARGET_OFFSET
        data.mocap_pos[right_mocap_id] = est.target_pos_b
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "grip_b":
        est.right_gripper_ctrl = gripper_closed

    elif phase == "release_a":
        est.gripper_ctrl = gripper_open

    elif phase == "retract_b":
        data.mocap_pos[right_mocap_id] = est.right_neutral_pos
        data.mocap_quat[right_mocap_id] = est.right_neutral_quat
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "move_to_tray":
        # Move arm B over the tray matching the marker.
        if est.chosen_color is not None and est.chosen_color in TRAY_POSITIONS:
            tray_pos = TRAY_POSITIONS[est.chosen_color]
            est.target_pos_b = tray_pos.copy()
            data.mocap_pos[right_mocap_id] = est.target_pos_b
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "drop_box":
        # Open the gripper to drop the box into the tray.
        est.right_gripper_ctrl = gripper_open

    elif phase == "wait_drop":
        # Keep the gripper open while the box falls.
        est.right_gripper_ctrl = gripper_open

    if phase not in ("retract_b", "move_to_tray", "drop_box", "wait_drop"):
        target_quat = GRASP_ORIENTATION.wxyz
        data.mocap_pos[left_mocap_id] = est.target_pos
        data.mocap_quat[left_mocap_id] = target_quat
        left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    vel = mink.solve_ik(configuration, tasks, DT, limits=limits, solver=solver, damping=1e-5)
    configuration.integrate_inplace(vel, DT)

    gripper_qpos = data.joint('left/left_finger').qpos[0]
    gripper_is_open = abs(gripper_qpos - gripper_open) < 0.002
    current_gripper_pos = data.site('left/gripper').xpos
    dist_to_target = np.linalg.norm(current_gripper_pos - est.target_pos)
    right_gripper_pos = data.site('right/gripper').xpos
    dist_right_to_target = np.linalg.norm(right_gripper_pos - est.target_pos_b)

    if phase == "present_a" and dist_to_target < POS_THRESHOLD:
        est.phase = "move_arm_B"
    elif phase == "approach" and dist_to_target < POS_THRESHOLD:
        est.phase = "open_gripper"
    elif phase == "open_gripper" and gripper_is_open:
        est.phase = "descend"
    elif phase == "descend" and dist_to_target < POS_THRESHOLD:
        est.phase = "grasp"
    elif phase == "grasp" and check_gripper_box_contact(model, data):
        est.phase = "lift"
    elif phase == "lift" and dist_to_target < POS_THRESHOLD:
        est.phase = "move_arm_B"
    elif phase == "move_arm_B" and dist_right_to_target < POS_THRESHOLD:
        est.phase = "approach_b"
    elif phase == "approach_b" and dist_right_to_target < B_GRIP_THRESHOLD:
        est.phase = "grip_b"
    elif phase == "grip_b" and check_gripper_box_contact_right(model, data):
        est.phase = "release_a"
    elif phase == "release_a" and gripper_is_open:
        est.phase = "retract_b"
    elif phase == "retract_b" and np.linalg.norm(right_gripper_pos - est.right_neutral_pos) < POS_THRESHOLD:
        # After retracting, move to the marker's tray.
        est.phase = "move_to_tray"
    elif phase == "move_to_tray":
        # Looser threshold (5 cm) for tray positioning.
        if dist_right_to_target < 0.05:
            est.phase = "drop_box"
    elif phase == "drop_box":
        # Once the right gripper is open, wait for the drop.
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]
        right_gripper_is_open = abs(right_gripper_qpos - gripper_open) < 0.01
        if right_gripper_is_open:
            est.phase = "wait_drop"
            # Start the drop-wait counter.
            if not hasattr(expert_step, '_drop_wait_count'):
                expert_step._drop_wait_count = 0
            expert_step._drop_wait_count = 0
    elif phase == "wait_drop":
        # Wait 100 steps (0.5 s at 200 Hz) for the box to fall.
        if not hasattr(expert_step, '_drop_wait_count'):
            expert_step._drop_wait_count = 0
        expert_step._drop_wait_count += 1
        if expert_step._drop_wait_count >= 100:
            est.phase = "done"

    action_a = np.concatenate([configuration.q[left_dof_ids], [est.gripper_ctrl]])
    action_b = np.concatenate([configuration.q[right_dof_ids], [est.right_gripper_ctrl]])
    return action_a, action_b, est


def run_episode(setup, box_x_range, box_y_range, max_steps=2000,
                 record_training_data=True, record_every_n_steps=10,
                 record_review_video=True,
                 review_video_every_n_steps=10,
                 handover_only=False,
                 episode_idx=None):
    """Run one full pick-and-handover episode headlessly.

    Returns (episode_data, success, review_frames); episode_data and
    review_frames are None when not recorded.
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]
    tasks = setup["tasks"]
    left_ee_task = setup["left_ee_task"]
    right_ee_task = setup["right_ee_task"]
    posture_task = setup["posture_task"]
    left_dof_ids = setup["left_dof_ids"]
    left_actuator_ids = setup["left_actuator_ids"]
    right_dof_ids = setup["right_dof_ids"]
    right_actuator_ids = setup["right_actuator_ids"]
    limits = setup["limits"]
    solver = setup["solver"]
    left_mocap_id = setup["left_mocap_id"]
    right_mocap_id = setup["right_mocap_id"]
    left_subtree_id = setup["left_subtree_id"]
    right_subtree_id = setup["right_subtree_id"]
    gripper_open = setup["GRIPPER_OPEN"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    left_gripper_actuator_id = setup["left_gripper_actuator_id"]
    right_gripper_actuator_id = setup["right_gripper_actuator_id"]
    GRASP_ORIENTATION = setup["GRASP_ORIENTATION"]
    LIFT_HEIGHT = setup["LIFT_HEIGHT"]
    box_joint_id = setup["box_joint_id"]
    box_spawn_height = setup["box_spawn_height"]

    dt = 1.0 / 200.0
    APPROACH_HEIGHT_OFFSET = 0.10
    POS_THRESHOLD = 0.02
    B_APPROACH_HEIGHT_OFFSET = 0.08
    right_target_offset = np.array([0.03, 0.0, 0.0])

    # One marker is made visible per episode (the others stay at alpha 0).
    MARKER_COLORS = ["blue", "green", "yellow"]
    marker_geom_ids = {
        color: model.geom(f"marker_{color}").id for color in MARKER_COLORS
    }

    if handover_only:
        # Handover-only: start with arm A holding the box.
        mujoco.mj_resetDataKeyframe(model, data, model.key("handover_start").id)
        mujoco.mj_forward(model, data)

        # Marker colour: cycles with episode_idx when given (balanced), else random.
        if episode_idx is not None:
            chosen_color = MARKER_COLORS[episode_idx % len(MARKER_COLORS)]
        else:
            chosen_color = np.random.choice(MARKER_COLORS)
        for color in MARKER_COLORS:
            gid = marker_geom_ids[color]
            if color == chosen_color:
                model.geom_rgba[gid, 3] = 1.0  # visible
            else:
                model.geom_rgba[gid, 3] = 0.0  # invisible
        mujoco.mj_forward(model, data)

        grip0 = data.site('left/gripper').xpos.copy()
        box0 = data.qpos[box_joint_id: box_joint_id + 3].copy()
        # Record A's rest pose before the mocap is retargeted (retract_a returns to it).
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        rest_pos_a = data.mocap_pos[left_mocap_id].copy()
        rest_quat_a = data.mocap_quat[left_mocap_id].copy()
        # Zero presentation offset: every episode starts from the same pose, so
        # the marker colour is the only thing that varies.
        offset = np.array([
            0.0,
            0.0,
            0.0,
        ])

        # Skip IK for a zero offset (the keyframe pose is already correct).
        if np.any(offset != 0):
            # Move A's gripper (and the held box) to the offset pose with IK.
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)
            data.mocap_pos[left_mocap_id] = grip0 + offset
            data.mocap_quat[left_mocap_id] = GRASP_ORIENTATION.wxyz
            left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            goal_a = grip0 + offset
            for _ in range(1500):
                vel = mink.solve_ik(configuration, tasks, dt, limits=limits,
                                    solver=solver, damping=1e-5)
                configuration.integrate_inplace(vel, dt)
                data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
                data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
                data.ctrl[left_gripper_actuator_id] = gripper_closed
                data.ctrl[right_gripper_actuator_id] = gripper_closed
                mujoco.mj_step(model, data)
                if np.linalg.norm(data.site('left/gripper').xpos - goal_a) < 0.005:
                    break

        # Reject episodes where the box was dropped during the reset.
        if not check_gripper_box_contact(model, data):
            return None, False, None

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        mujoco.mj_forward(model, data)
    else:
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        rand_x = np.random.uniform(*box_x_range)
        rand_y = np.random.uniform(*box_y_range)
        data.qpos[box_joint_id: box_joint_id + 3] = [rand_x, rand_y, box_spawn_height]
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        mujoco.mj_forward(model, data)
        # No marker in full-task mode.
        chosen_color = None

    
    if handover_only:
        left_neutral_pos = rest_pos_a
        left_neutral_quat = rest_quat_a
    else:
        left_neutral_pos = data.mocap_pos[left_mocap_id].copy()
        left_neutral_quat = data.mocap_quat[left_mocap_id].copy()

    right_neutral_pos = data.mocap_pos[right_mocap_id].copy()
    right_neutral_quat = data.mocap_quat[right_mocap_id].copy()

    phase = "present_a" if handover_only else "approach"
    gripper_ctrl = gripper_closed
    right_gripper_ctrl = gripper_closed
    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    episode_data = EpisodeData() if record_training_data else None
    if episode_data is not None:
        # chosen_color only exists in handover-only mode.
        episode_data.marker_color = locals().get('chosen_color')
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    step_count = 0
    target_pos = data.site('left/gripper').xpos.copy()  
    target_pos_b = data.site('right/gripper').xpos.copy()
    A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0]) 

    # The phase machine lives in expert_step().
    est = ExpertState(
        phase=phase,
        target_pos=target_pos,
        target_pos_b=target_pos_b,
        gripper_ctrl=gripper_ctrl,
        right_gripper_ctrl=right_gripper_ctrl,
        left_neutral_pos=left_neutral_pos,
        left_neutral_quat=left_neutral_quat,
        right_neutral_pos=right_neutral_pos,
        right_neutral_quat=right_neutral_quat,
        chosen_color=chosen_color,
    )

    while est.phase != "done" and step_count < max_steps:
        box_pos = data.joint('middle_box_joint').qpos[:3]
        action_a, action_b, est = expert_step(setup, est)
        phase = est.phase
        gripper_ctrl = est.gripper_ctrl
        right_gripper_ctrl = est.right_gripper_ctrl
        gripper_qpos = data.joint('left/left_finger').qpos[0]
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]

        data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
        data.ctrl[left_gripper_actuator_id] = gripper_ctrl
        data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
        data.ctrl[right_gripper_actuator_id] = right_gripper_ctrl

        compensate_gravity(model, data, [left_subtree_id, right_subtree_id])
        mujoco.mj_step(model, data)

        # Capture training data every step
        if record_training_data and step_count % record_every_n_steps == 0:
            state_a = np.concatenate([data.qpos[left_dof_ids], [gripper_qpos]])
            state_b = np.concatenate([data.qpos[right_dof_ids], [right_gripper_qpos]])
            action_a = np.concatenate([configuration.q[left_dof_ids], [gripper_ctrl]])
            action_b = np.concatenate([configuration.q[right_dof_ids], [right_gripper_ctrl]])

            episode_data.add_step(
                image_overhead=render_camera(model, data, renderer, "overhead_cam"),
                image_wrist_a=render_camera(model, data, renderer, "wrist_cam_left"),
                image_wrist_b=render_camera(model, data, renderer, "wrist_cam_right"),
                state_a=state_a,
                state_b=state_b,
                action_a=action_a,
                action_b=action_b,
                box_pos=box_pos.copy(),
                phase=phase,
            )

        # Capture review video, sparsely (not every step, for speed)
        if record_review_video and step_count % review_video_every_n_steps == 0:
            renderer.update_scene(data, camera=REVIEW_CAMERA)
            review_frames.append(renderer.render())

        step_count += 1

    if renderer is not None:
        renderer.close()

    success = (phase == "done")
    return episode_data, success, review_frames

       