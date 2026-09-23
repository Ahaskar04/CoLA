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
_XML = _PROJECT_ROOT / "environments" / "handover_2arm_marker" / "scene.xml"

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
        # Which marker is visible this episode. Constant for the episode, so it
        # is a scalar rather than a per-step column. Set by run_episode after it
        # draws the colour. WITHOUT THIS the dataset cannot be scored for
        # CORRECT-tray placement, only "landed in some tray" -- which discards
        # the 33% chance baseline the whole task is built around.
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
            # Fixed-width bytes, matching how `phase` is stored: h5py cannot
            # write a bare Python str, and "yellow" is the longest value.
            "marker_color": np.array(self.marker_color or "unknown", dtype="S10"),
        }



class ExpertState:
    """The scripted expert's carried state.

    The expert is a phase machine, so it cannot be queried from a bare
    (model, data) snapshot: `phase` says which stage it believes it is in, and
    `target_pos` / `target_pos_b` are IK goals that each phase MUTATES rather
    than recomputing from scratch. `configuration` is mink's own copy of the
    joint state. All of it has to be carried between steps.
    """

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


# Constants the phase machine uses. Module level so expert_step() and
# run_episode() cannot drift apart.
DT = 1.0 / 200.0
APPROACH_HEIGHT_OFFSET = 0.10
POS_THRESHOLD = 0.02
B_APPROACH_HEIGHT_OFFSET = 0.08
# How close B's gripper site ends up to A's. 0.03 was tuned for B reaching
# DOWN onto a box lying flat; coming in horizontally at a box held out
# sideways it left B short, closing just outside the box. 0.01 brings it
# deeper still: at +0.01 two of five episodes sat in grip_b for 140+ steps
# closing on air, so B's target now sits 1cm PAST A's gripper site and the
# fingers straddle the box. The box is 8cm long, so there is room.
# 1cm deeper than the 0.03 the dataset was collected at. Earlier depth tests
# (0.01 -> 2/5, -0.01 -> 0/5, B pushing the box away) predate the
# B_GRIP_THRESHOLD fix, when B closed 4.5cm short no matter where its
# target sat -- so they are confounded and worth redoing.
RIGHT_TARGET_OFFSET = np.array([-0.03, 0.0, 0.03])  # x-3cm reaches deeper toward Arm A, z+3cm raises handover point
# approach_b -> grip_b used POS_THRESHOLD (0.02), so B started closing while
# still 2cm short of its own target -- which is itself 3cm from A's gripper.
# Tracing grip_b showed B shutting its fingers 4.5cm from the box with ZERO
# right/* contacts: it closed on air, then A released into nothing and the
# box fell. Require B to actually arrive before it grips.
B_GRIP_THRESHOLD = 0.005
A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0])
# Where arm A holds the box out for the handover, on the axis between the
# two arms (A base x=-0.55, B base x=+0.55, midline x=0). Handover-only
# episodes start with A at neutral, so without a phase that moves it the
# whole transfer is done by B travelling across the table.
PRESENT_X = -0.02
# Camera the human-review video is rendered from. COLA_REVIEW_CAMERA=wrist_cam_left
# shows arm A's own point of view -- what the policy actually sees.
REVIEW_CAMERA = os.environ.get("COLA_REVIEW_CAMERA", "overhead_cam")

# Tray positions for the marker task: Arm B drops the box in the matching tray.
# All trays are at x=0.36 (Arm B's side), height z=0.20 (above tray surface).
TRAY_POSITIONS = {
    "blue":   np.array([0.36, -0.247, 0.20]),
    "green":  np.array([0.36,  0.0,   0.20]),
    "yellow": np.array([0.36,  0.247, 0.20]),
}


def expert_step(setup, est, sync_configuration=False):
    """One step of the scripted expert: decide, solve IK, advance the phase.

    Returns (action_a, action_b, est) where each action is
    [6 joint targets, gripper] -- exactly the layout collect_demos.py records.

    Does NOT write data.ctrl and does NOT step physics, so it can run alongside
    a learned policy that owns the arms (DAgger observer mode).

    sync_configuration=True re-seeds mink from data.qpos before solving. Needed
    whenever something OTHER than this function moved the arms -- otherwise the
    expert computes a correction from a pose the arm is not actually in.

    NOTE: this writes data.mocap_pos/quat, because mink reads the IK target back
    out of the mocap bodies. The marker geoms are alpha 0, so this does not
    change what any camera sees.
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
        # Only x is overridden: y and z keep whatever the reset randomised,
        # so the presentation pose still varies episode to episode.
        est.target_pos = np.array([PRESENT_X, est.target_pos[1], est.target_pos[2]])

    elif phase == "move_arm_B":
        # Stage OUTWARD along x (B's own side), not above. The old
        # [0, 0, B_APPROACH_HEIGHT_OFFSET] put B 8cm over the box so approach_b
        # then dropped onto it -- correct for a box lying on the table, wrong
        # for one held out horizontally, where B should come in from the side.
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
        # Move Arm B over the correct colored tray based on the marker
        if est.chosen_color is not None and est.chosen_color in TRAY_POSITIONS:
            tray_pos = TRAY_POSITIONS[est.chosen_color]
            est.target_pos_b = tray_pos.copy()
            data.mocap_pos[right_mocap_id] = est.target_pos_b
            right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "drop_box":
        # Open gripper to drop the box into the tray
        est.right_gripper_ctrl = gripper_open

    elif phase == "wait_drop":
        # Keep gripper open and wait for box to fall
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
        # After retract, move to the correct tray based on marker color
        est.phase = "move_to_tray"
    elif phase == "move_to_tray":
        # Looser threshold for tray positioning (0.05 = 5cm) since exact placement isn't critical
        if dist_right_to_target < 0.05:
            est.phase = "drop_box"
    elif phase == "drop_box":
        # Check if right gripper is open - once open, transition to wait_drop
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]
        right_gripper_is_open = abs(right_gripper_qpos - gripper_open) < 0.01
        if right_gripper_is_open:
            est.phase = "wait_drop"
            # Initialize drop wait counter
            if not hasattr(expert_step, '_drop_wait_count'):
                expert_step._drop_wait_count = 0
            expert_step._drop_wait_count = 0
    elif phase == "wait_drop":
        # Wait 100 steps (~0.5 sec at 200Hz) for box to actually fall
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
    """
    Run one full pick-and-handover episode, headless (no live viewer).

    Returns:
        episode_data: EpisodeData object (or None if record_training_data=False)
        success: bool, True if the episode reached the "done" phase
        review_frames: list of rendered frames for a human-review video (or None)
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

    # Marker colors: randomly select one to be visible each episode.
    # The marker geoms start with alpha=0 (invisible); we set the chosen one to alpha=1.
    MARKER_COLORS = ["blue", "green", "yellow"]
    marker_geom_ids = {
        color: model.geom(f"marker_{color}").id for color in MARKER_COLORS
    }

    if handover_only:
        # Start mid-task: arm A already holds the box, so the episode is the
        # handover alone. The keyframe is one settled equilibrium pose; the
        # randomisation below moves the whole grasp (arm + box together) so
        # every episode presents the box somewhere different.
        mujoco.mj_resetDataKeyframe(model, data, model.key("handover_start").id)
        mujoco.mj_forward(model, data)

        # Select marker color: cycle through colors for balanced distribution.
        # If episode_idx is provided, use it to deterministically pick color.
        # Otherwise fall back to random selection.
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
        # Capture A's REST pose now, before the mocap is repointed at the IK
        # goal below. left_neutral_pos is read off the mocap further down, so
        # without this it would be set to the handover pose and `retract_a`
        # would have nowhere to retract to.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        rest_pos_a = data.mocap_pos[left_mocap_id].copy()
        rest_quat_a = data.mocap_quat[left_mocap_id].copy()
        # x is the axis SEPARATING the two arms (left base at x=-0.55, right at
        # x=+0.55), so it sets how far apart they meet. Hold it fixed: the arms
        # should always rendezvous at the same point along that axis, and the
        # variation should be in where A presents the box within its own
        # workspace -- left/right (y) and up/down (z).
        #
        # y is capped at 0.07, not the 0.10 first tried: at y=+0.108 arm B could
        # not reach the presentation point at all and the episode sat in
        # move_arm_B for its full 200 steps. Every episode that completed had
        # |y| <= 0.065, so beyond ~0.07 the failures are unreachable geometry
        # rather than hard coordination.
        #
        # z is the axis with NO prior variation: every full-task demo lifts to
        # LIFT_HEIGHT, giving recorded handover height std 0.003, so arm B has
        # never seen the box presented at a different height.
        # Capped at 0.06: every failure across the tuning runs was a HIGH
        # presentation (z 0.33-0.35) where B cannot reach, while every
        # success was z <= 0.31. At +-0.08 those high draws fail and get
        # retried with lower ones, so the kept episodes skew low instead of
        # spanning the range -- less usable height diversity, not more.
        offset = np.array([
            0.0,
            0.0,
            0.0,
        ])

        # Only run IK if there's an actual offset to move to. When offset is
        # zero, the keyframe pose is already correct and running IK can cause
        # the solver to find an alternative joint configuration that reaches
        # the same Cartesian position -- making the arm "dance" unnecessarily.
        if np.any(offset != 0):
            # Drive A's gripper to the offset pose with IK, carrying the box: the
            # box is only held by contact, so moving it independently would push it
            # out of the fingers.
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

        # Reject rather than record a dropped box: an episode that starts with
        # nothing in the gripper trains the policy on an impossible task.
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
        # For non-handover mode, we don't use marker-based tray placement
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
        # chosen_color is bound only in the handover_only branch above -- the
        # full-task branch never draws a marker -- so read it defensively
        # rather than by bare name, which would raise NameError there.
        episode_data.marker_color = locals().get('chosen_color')
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    step_count = 0
    target_pos = data.site('left/gripper').xpos.copy()  
    target_pos_b = data.site('right/gripper').xpos.copy()
    A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0]) 

    # Single source of truth: the phase machine lives in expert_step() so the
    # DAgger collector queries exactly the same expert this records.
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

        # --- capture training data every step ---
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

        # --- capture review video, sparsely (not every step, for speed) ---
        if record_review_video and step_count % review_video_every_n_steps == 0:
            renderer.update_scene(data, camera=REVIEW_CAMERA)
            review_frames.append(renderer.render())

        step_count += 1

    if renderer is not None:
        renderer.close()

    success = (phase == "done")
    return episode_data, success, review_frames

       