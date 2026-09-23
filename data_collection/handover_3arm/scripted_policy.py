from pathlib import Path
import mujoco
import mujoco.viewer
import os
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from mink import SO3
from utils import setup_dual_arm_ik, compensate_gravity, check_gripper_box_contact, check_gripper_box_contact_right, check_gripper_box_contact_third

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
_XML = _PROJECT_ROOT / "environments" / "handover_3arm" / "scene.xml"

setup = setup_dual_arm_ik(_XML)

# Random box spawn position
box_x_range = (-0.12, 0.12)
box_y_range = (-0.10, 0.10)

# Debug counters (module-level since ExpertState uses __slots__)
_turn_b_step_count = 0
_turn_b_transition_check_count = 0
_turn_b_initialized = False
_grip_c_step_count = 0
_approach_c_initialized = False
_approach_c_step_count = 0


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
        self.image_wrist_c = []
        self.state_a = []
        self.state_b = []
        self.state_c = []
        self.action_a = []
        self.action_b = []
        self.action_c = []
        self.box_pos = []
        self.phase = []

    def add_step(self, image_overhead, image_wrist_a, image_wrist_b, image_wrist_c,
                 state_a, state_b, state_c, action_a, action_b, action_c,
                 box_pos, phase):
        self.image_overhead.append(image_overhead)
        self.image_wrist_a.append(image_wrist_a)
        self.image_wrist_b.append(image_wrist_b)
        self.image_wrist_c.append(image_wrist_c)
        self.state_a.append(state_a)
        self.state_b.append(state_b)
        self.state_c.append(state_c)
        self.action_a.append(action_a)
        self.action_b.append(action_b)
        self.action_c.append(action_c)
        self.box_pos.append(box_pos)
        self.phase.append(phase)

    def to_dict(self):
        return {
            "image_overhead": np.array(self.image_overhead),
            "image_wrist_a": np.array(self.image_wrist_a),
            "image_wrist_b": np.array(self.image_wrist_b),
            "image_wrist_c": np.array(self.image_wrist_c),
            "state_a": np.array(self.state_a),
            "state_b": np.array(self.state_b),
            "state_c": np.array(self.state_c),
            "action_a": np.array(self.action_a),
            "action_b": np.array(self.action_b),
            "action_c": np.array(self.action_c),
            "box_pos": np.array(self.box_pos),
            "phase": np.array(self.phase, dtype="S20"),
        }



class ExpertState:
    """The scripted expert's carried state.

    The expert is a phase machine, so it cannot be queried from a bare
    (model, data) snapshot: `phase` says which stage it believes it is in, and
    `target_pos` / `target_pos_b` are IK goals that each phase MUTATES rather
    than recomputing from scratch. `configuration` is mink's own copy of the
    joint state. All of it has to be carried between steps.
    """

    __slots__ = ('phase', 'target_pos', 'target_pos_b', 'target_pos_c', 'gripper_ctrl',
                 'right_gripper_ctrl', 'third_gripper_ctrl',
                 'left_neutral_pos', 'left_neutral_quat',
                 'right_neutral_pos', 'right_neutral_quat',
                 'third_neutral_pos', 'third_neutral_quat',
                 'waist_b_target_angle', 'waist_b_start_angle')

    def __init__(self, phase, target_pos, target_pos_b, gripper_ctrl,
                 right_gripper_ctrl, left_neutral_pos, left_neutral_quat,
                 right_neutral_pos, right_neutral_quat,
                 target_pos_c=None, third_gripper_ctrl=None,
                 third_neutral_pos=None, third_neutral_quat=None,
                 waist_b_target_angle=None, waist_b_start_angle=None):
        self.phase = phase
        self.target_pos = target_pos
        self.target_pos_b = target_pos_b
        self.gripper_ctrl = gripper_ctrl
        self.right_gripper_ctrl = right_gripper_ctrl
        self.left_neutral_pos = left_neutral_pos
        self.left_neutral_quat = left_neutral_quat
        self.right_neutral_pos = right_neutral_pos
        self.right_neutral_quat = right_neutral_quat
        # Arm C. Default None so 2-arm callers construct an ExpertState
        # unchanged; the B->C phases fill these in.
        self.target_pos_c = target_pos_c
        self.third_gripper_ctrl = third_gripper_ctrl
        self.third_neutral_pos = third_neutral_pos
        self.third_neutral_quat = third_neutral_quat
        # Waist rotation tracking for turn_b phase
        self.waist_b_target_angle = waist_b_target_angle
        self.waist_b_start_angle = waist_b_start_angle

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
RIGHT_TARGET_OFFSET = np.array([0.02, 0.0, 0.0])
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
# Where arm B holds the box out for the SECOND handover. Arm bases sit at
# -0.55 / +0.55 / +1.65, so B is exactly midway between A and C and the two
# 1.10 m gaps are identical: the B->C meet point is the A->B one mirrored
# about B, i.e. PRESENT_X reflected through x=+0.55.
PRESENT_X_C = 2 * 0.55 - PRESENT_X          # +1.12
# B must turn to face C before presenting. Verified in simulation: the box
# survives a full 180 deg waist sweep with contact held at every 15 deg step.
B_TURN_RAD = np.pi
# Camera the human-review video is rendered from. COLA_REVIEW_CAMERA=wrist_cam_left
# shows arm A's own point of view -- what the policy actually sees.
# For 3-arm: side_cam shows all three arms
REVIEW_CAMERA = os.environ.get("COLA_REVIEW_CAMERA", "side_cam")


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
    third_ee_task = setup["third_ee_task"]
    third_dof_ids = setup["third_dof_ids"]
    third_mocap_id = setup["third_mocap_id"]
    gripper_open = setup["GRIPPER_OPEN"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    GRASP_ORIENTATION = setup["GRASP_ORIENTATION"]
    LIFT_HEIGHT = setup["LIFT_HEIGHT"]

    # During turn_b, don't reset configuration from data.qpos - we're manually controlling the waist
    phase = est.phase
    if sync_configuration and phase != "turn_b":
        configuration.update(data.qpos)

    # Override waist angle during turn_b phase - do this AFTER configuration.update()
    # but BEFORE IK solve, so we manually control the waist and keep other joints frozen
    if phase == "turn_b":
        global _turn_b_step_count, _turn_b_initialized
        if est.waist_b_target_angle is not None:
            # Use DOF ID not qposadr! right_dof_ids[0] is the waist (first in the joint list)
            waist_dof_id = right_dof_ids[0]
            waist_qpos_id = model.joint('right/waist').qposadr[0]
            current_angle = data.qpos[waist_qpos_id]

            # Track the commanded angle in configuration.data.qpos, incrementing it each step
            # NOTE: configuration.q is a property that returns a COPY, so we must modify data.qpos directly!
            if not _turn_b_initialized:
                _turn_b_initialized = True
                _turn_b_step_count = 0
                print(f"\n=== TURN_B DEBUG ===")
                print(f"Waist DOF ID: {waist_dof_id}, qpos ID: {waist_qpos_id}")
                print(f"Start angle: {est.waist_b_start_angle:.4f} rad ({np.degrees(est.waist_b_start_angle):.2f}°)")
                print(f"Target angle: {est.waist_b_target_angle:.4f} rad ({np.degrees(est.waist_b_target_angle):.2f}°)")
                print(f"Total rotation needed: {B_TURN_RAD:.4f} rad ({np.degrees(B_TURN_RAD):.2f}°)")

            _turn_b_step_count += 1

            # Increment the commanded angle in configuration.data.qpos directly
            commanded_angle = configuration.data.qpos[waist_qpos_id]
            angle_remaining = est.waist_b_target_angle - commanded_angle

            # Rotate in small increments (0.05 rad per step = ~2.86 degrees)
            WAIST_ROTATION_SPEED = 0.05
            if abs(angle_remaining) > WAIST_ROTATION_SPEED:
                # Still rotating - increment the commanded angle directly in data.qpos
                old_val = configuration.data.qpos[waist_qpos_id]
                configuration.data.qpos[waist_qpos_id] += np.sign(angle_remaining) * WAIST_ROTATION_SPEED
                new_val = configuration.data.qpos[waist_qpos_id]
                if _turn_b_step_count % 10 == 0:
                    print(f"Step {_turn_b_step_count}: current={current_angle:.4f}, commanded: {old_val:.4f} → {new_val:.4f}, remaining={angle_remaining:.4f}")
            else:
                # Rotation complete, set exact target
                configuration.data.qpos[waist_qpos_id] = est.waist_b_target_angle
                if _turn_b_step_count % 50 == 0:  # Only print every 50 steps to reduce spam
                    print(f"Rotation complete at step {_turn_b_step_count}!")
        else:
            print("WARNING: turn_b phase but waist_b_target_angle is None!")

    box_pos = data.joint('middle_box_joint').qpos[:3]

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

    # ---- second leg: B hands on to C ----
    elif phase == "turn_b":
        # Rotate B to face C. The waist is driven directly rather than through
        # IK: asking the solver for a pose on the far side of a 180 deg sweep
        # invites it to take an arbitrary path, and the box is held by friction
        # alone. Everything else is left where it is so the grasp is undisturbed.

        # Initialize rotation on first entry to this phase
        if est.waist_b_start_angle is None:
            waist_joint_id = model.joint('right/waist').qposadr[0]
            est.waist_b_start_angle = data.qpos[waist_joint_id]
            est.waist_b_target_angle = est.waist_b_start_angle + B_TURN_RAD

        # Keep the IK target tracking the current gripper position
        # so other joints don't try to move to compensate for waist rotation
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        est.target_pos_b = data.mocap_pos[right_mocap_id].copy()
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "present_b2":
        # Mirror of present_a: only x is overridden, so whatever y/z the box
        # ended up at after the first handover carries through and the
        # presentation to C still varies episode to episode.
        est.target_pos_b = np.array([PRESENT_X_C, est.target_pos_b[1], est.target_pos_b[2]])
        data.mocap_pos[right_mocap_id] = est.target_pos_b
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

    elif phase == "move_arm_C":
        # C stages OUTWARD along +x on its own side, mirroring move_arm_B.
        # After B rotated 180°, the box orientation flipped, so C approaches from +x
        # (same direction as B did) to reach the opposite side of the box.
        est.target_pos_c = (est.target_pos_b + RIGHT_TARGET_OFFSET
                            + np.array([B_APPROACH_HEIGHT_OFFSET, 0.0, 0.0]))
        data.mocap_pos[third_mocap_id] = est.target_pos_c
        third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))
        est.third_gripper_ctrl = gripper_open

    elif phase == "approach_c":
        global _approach_c_initialized, _approach_c_step_count
        if not _approach_c_initialized:
            _approach_c_initialized = True
            _approach_c_step_count = 0
            box_pos = data.joint('middle_box_joint').qpos[:3]
            print(f"\n=== APPROACH_C DEBUG ===")
            print(f"Box position: {box_pos}")
            print(f"Arm B target pos: {est.target_pos_b}")
            print(f"RIGHT_TARGET_OFFSET: {RIGHT_TARGET_OFFSET}")
            print(f"Arm C target will be: {est.target_pos_b + RIGHT_TARGET_OFFSET}")
            print(f"Arm C gripper current pos: {data.site('third/gripper').xpos}")
            print(f"B_GRIP_THRESHOLD: {B_GRIP_THRESHOLD}")
        _approach_c_step_count += 1
        # Set target position and shift mocap directly to avoid gripper collision
        base_target = est.target_pos_b + RIGHT_TARGET_OFFSET
        # Shift the mocap position in X to avoid collision with arm B's gripper
        # Need enough space for both grippers to fit without jamming
        mocap_pos = base_target.copy()
        mocap_pos[0] += 0.015  # 15mm X offset (arm C is to the right along X-axis)
        est.target_pos_c = mocap_pos  # Store the actual target position for distance calculations
        data.mocap_pos[third_mocap_id] = mocap_pos
        third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))
        # Debug: print distance to target every 50 steps
        if _approach_c_step_count % 50 == 0:
            current_gripper_pos = data.site('third/gripper').xpos
            current_dist = np.linalg.norm(current_gripper_pos - est.target_pos_c)
            print(f"approach_c step {_approach_c_step_count}: dist_to_target={current_dist:.4f}, threshold={B_GRIP_THRESHOLD}")

    elif phase == "grip_c":
        est.third_gripper_ctrl = gripper_closed

    elif phase == "release_b":
        est.right_gripper_ctrl = gripper_open
        # Debug: print gripper state during release
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]
        print(f"release_b: setting ctrl={gripper_open:.4f}, current qpos={right_gripper_qpos:.4f}")

    elif phase == "retract_c":
        data.mocap_pos[third_mocap_id] = est.third_neutral_pos
        data.mocap_quat[third_mocap_id] = est.third_neutral_quat
        third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))

    if phase != "retract_b":
        target_quat = GRASP_ORIENTATION.wxyz
        data.mocap_pos[left_mocap_id] = est.target_pos
        data.mocap_quat[left_mocap_id] = target_quat
        left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    # Skip IK during turn_b - we manually control the waist and keep other joints frozen
    if phase != "turn_b":
        vel = mink.solve_ik(configuration, tasks, DT, limits=limits, solver=solver, damping=1e-5)
        configuration.integrate_inplace(vel, DT)

    gripper_qpos = data.joint('left/left_finger').qpos[0]
    gripper_is_open = abs(gripper_qpos - gripper_open) < 0.002
    current_gripper_pos = data.site('left/gripper').xpos
    dist_to_target = np.linalg.norm(current_gripper_pos - est.target_pos)
    right_gripper_pos = data.site('right/gripper').xpos
    dist_right_to_target = np.linalg.norm(right_gripper_pos - est.target_pos_b)
    # Arm C, and B's own gripper state -- release_b waits on B opening, the way
    # release_a waits on A. gripper_is_open above tracks arm A only.
    third_gripper_pos = data.site('third/gripper').xpos
    dist_third_to_target = (np.linalg.norm(third_gripper_pos - est.target_pos_c)
                            if est.target_pos_c is not None else np.inf)
    right_gripper_qpos = data.joint('right/left_finger').qpos[0]
    right_gripper_is_open = abs(right_gripper_qpos - gripper_open) < 0.002

    # Phase transitions with debug prints
    old_phase = phase

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
        # In the 2-arm task this ended the episode. Now B still holds the box
        # and hands it on to C.
        est.phase = "turn_b"
    elif phase == "turn_b":
        # Check if rotation is complete by comparing to target angle
        global _turn_b_transition_check_count
        if est.waist_b_target_angle is not None:
            current_waist_angle = data.qpos[model.joint('right/waist').qposadr[0]]
            angle_remaining = abs(est.waist_b_target_angle - current_waist_angle)
            _turn_b_transition_check_count += 1
            if _turn_b_transition_check_count % 20 == 0:
                print(f"Turn_b transition check: angle_remaining={angle_remaining:.4f} rad ({np.degrees(angle_remaining):.2f}°), threshold=0.1 rad")
            if angle_remaining < 0.1:  # Within 0.1 radians (~5.7 degrees) of target
                est.phase = "present_b2"
                print(f"Turn_b → present_b2: rotation complete!")
    elif phase == "present_b2" and dist_right_to_target < POS_THRESHOLD:
        est.phase = "move_arm_C"
    elif phase == "move_arm_C" and dist_third_to_target < POS_THRESHOLD:
        est.phase = "approach_c"
    elif phase == "approach_c" and dist_third_to_target < B_GRIP_THRESHOLD:
        # Same tight threshold as approach_b: at POS_THRESHOLD (0.02) the
        # receiving arm starts closing while still short of the box and grips air.
        est.phase = "grip_c"
    elif phase == "grip_c":
        global _grip_c_step_count
        _grip_c_step_count += 1
        has_contact = check_gripper_box_contact_third(model, data)
        third_gripper_qpos = data.joint('third/left_finger').qpos[0]
        box_pos = data.joint('middle_box_joint').qpos[:3]
        gripper_c_pos = data.site('third/gripper').xpos
        dist_to_box = np.linalg.norm(gripper_c_pos - box_pos)
        # Print first 10 steps, then every 20 steps
        if _grip_c_step_count <= 10 or _grip_c_step_count % 20 == 0:
            print(f"grip_c step {_grip_c_step_count}: contact={has_contact}, dist_to_box={dist_to_box:.4f}, gripper_ctrl={est.third_gripper_ctrl:.4f}, qpos={third_gripper_qpos:.4f}, target={gripper_closed:.4f}")
        # Wait at least 50 steps before checking contact to allow gripper to close
        if has_contact and _grip_c_step_count > 50:
            print(f"grip_c → release_b: contact detected after {_grip_c_step_count} steps")
            est.phase = "release_b"
    elif phase == "release_b" and right_gripper_is_open:
        est.phase = "retract_c"
    elif phase == "retract_c" and np.linalg.norm(third_gripper_pos - est.third_neutral_pos) < POS_THRESHOLD:
        est.phase = "done"

    # Print when phase changes
    if est.phase != old_phase:
        print(f"Phase transition: {old_phase} → {est.phase}")

    action_a = np.concatenate([configuration.q[left_dof_ids], [est.gripper_ctrl]])
    action_b = np.concatenate([configuration.q[right_dof_ids], [est.right_gripper_ctrl]])
    action_c = np.concatenate([configuration.q[third_dof_ids],
                               [est.third_gripper_ctrl if est.third_gripper_ctrl
                                is not None else gripper_open]])
    return action_a, action_b, action_c, est


# 2000 sim steps at record_every_n_steps=10 gave exactly 200 recorded frames,
# which is what both smoke-test episodes hit -- they were not stalling, they
# ran out of budget mid-rotation. A 2-arm episode takes ~1680 sim steps for
# ONE handover leg; three arms add a 180 deg turn and a second leg, so 4500
# leaves headroom without letting a genuinely stuck episode run forever.
def run_episode(setup, box_x_range, box_y_range, max_steps=4500,
                 record_training_data=True, record_every_n_steps=10,
                 record_review_video=True,
                 review_video_every_n_steps=3,
                 handover_only=False):
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
    third_ee_task = setup["third_ee_task"]
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

    if handover_only:
        # Start mid-task: arm A already holds the box, so the episode is the
        # handover alone. The keyframe is one settled equilibrium pose; the
        # randomisation below moves the whole grasp (arm + box together) so
        # every episode presents the box somewhere different.
        mujoco.mj_resetDataKeyframe(model, data, model.key("handover_start").id)
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
            np.random.uniform(-0.10, 0.10),
            np.random.uniform(-0.06, 0.06),
        ])

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
        mink.move_mocap_to_frame(model, data, "third/target", "third/gripper", "site")
        third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))

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
        mink.move_mocap_to_frame(model, data, "third/target", "third/gripper", "site")
        mujoco.mj_forward(model, data)

    
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
    third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))

    episode_data = EpisodeData() if record_training_data else None
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    step_count = 0
    target_pos = data.site('left/gripper').xpos.copy()
    target_pos_b = data.site('right/gripper').xpos.copy()
    target_pos_c = data.site('third/gripper').xpos.copy()
    A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0])

    # Get third arm neutral positions
    third_mocap_id = setup["third_mocap_id"]
    third_neutral_pos = data.mocap_pos[third_mocap_id].copy()
    third_neutral_quat = data.mocap_quat[third_mocap_id].copy()

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
        target_pos_c=target_pos_c,
        third_gripper_ctrl=gripper_closed,
        third_neutral_pos=third_neutral_pos,
        third_neutral_quat=third_neutral_quat,
    )

    # Get third arm IDs from setup
    third_dof_ids = setup["third_dof_ids"]
    third_actuator_ids = setup["third_actuator_ids"]
    third_subtree_id = setup["third_subtree_id"]
    third_gripper_actuator_id = setup["third_gripper_actuator_id"]

    while est.phase != "done" and step_count < max_steps:
        box_pos = data.joint('middle_box_joint').qpos[:3]
        action_a, action_b, action_c, est = expert_step(setup, est)
        phase = est.phase
        gripper_ctrl = est.gripper_ctrl
        right_gripper_ctrl = est.right_gripper_ctrl
        third_gripper_ctrl = est.third_gripper_ctrl if est.third_gripper_ctrl is not None else gripper_open
        gripper_qpos = data.joint('left/left_finger').qpos[0]
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]
        third_gripper_qpos = data.joint('third/left_finger').qpos[0]

        data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
        data.ctrl[left_gripper_actuator_id] = gripper_ctrl
        data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
        data.ctrl[right_gripper_actuator_id] = right_gripper_ctrl
        data.ctrl[third_actuator_ids] = configuration.q[third_dof_ids]
        data.ctrl[third_gripper_actuator_id] = third_gripper_ctrl

        compensate_gravity(model, data, [left_subtree_id, right_subtree_id, third_subtree_id])
        mujoco.mj_step(model, data)

        # --- capture training data every step ---
        if record_training_data and step_count % record_every_n_steps == 0:
            state_a = np.concatenate([data.qpos[left_dof_ids], [gripper_qpos]])
            state_b = np.concatenate([data.qpos[right_dof_ids], [right_gripper_qpos]])
            # Arm C's proprioception, same layout: six joint positions then the
            # left finger. expert_step already computes and actuates action_c;
            # without state_c and action_c recorded, C would be trained with
            # images and no labels.
            state_c = np.concatenate([data.qpos[third_dof_ids], [third_gripper_qpos]])
            action_a = np.concatenate([configuration.q[left_dof_ids], [gripper_ctrl]])
            action_b = np.concatenate([configuration.q[right_dof_ids], [right_gripper_ctrl]])

            episode_data.add_step(
                image_overhead=render_camera(model, data, renderer, "overhead_cam"),
                image_wrist_a=render_camera(model, data, renderer, "wrist_cam_left"),
                image_wrist_b=render_camera(model, data, renderer, "wrist_cam_right"),
                image_wrist_c=render_camera(model, data, renderer, "wrist_cam_third"),
                state_a=state_a,
                state_b=state_b,
                state_c=state_c,
                action_a=action_a,
                action_b=action_b,
                action_c=action_c,
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


def main():
    """
    Main function for testing the 3-arm handover environment.

    You can customize the parameters below to test different scenarios:
    - handover_only: Start with arm A already holding the box (True) or do full pick-up (False)
    - max_steps: Maximum steps before timeout
    - record_training_data: Whether to record episode data
    - record_review_video: Whether to record video frames
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test 3-arm handover scripted policy")
    parser.add_argument("--handover-only", action="store_true",
                       help="Start with arm A already holding box (skip pickup phase)")
    parser.add_argument("--max-steps", type=int, default=4000,
                       help="Maximum steps per episode (default: 4000 for 3-arm)")
    parser.add_argument("--no-video", action="store_true",
                       help="Disable review video recording")
    parser.add_argument("--viewer", action="store_true",
                       help="Launch interactive viewer (real-time visualization)")
    parser.add_argument("--num-episodes", type=int, default=1,
                       help="Number of episodes to run (default: 1)")

    args = parser.parse_args()

    print("=" * 70)
    print("3-ARM HANDOVER SCRIPTED POLICY TEST")
    print("=" * 70)
    print(f"Scene XML: {_XML}")
    print(f"Handover only mode: {args.handover_only}")
    print(f"Max steps: {args.max_steps}")
    print(f"Record video: {not args.no_video}")
    print(f"Interactive viewer: {args.viewer}")
    print(f"Number of episodes: {args.num_episodes}")
    print("=" * 70)

    if args.viewer:
        # Interactive viewer mode - run one episode with live visualization
        print("\nLaunching interactive viewer...")
        print("Controls:")
        print("  - Double-click to rotate")
        print("  - Right-click to pan")
        print("  - Scroll to zoom")
        print("  - Press ESC to exit")
        print()

        run_episode_with_viewer(
            setup=setup,
            box_x_range=box_x_range,
            box_y_range=box_y_range,
            max_steps=args.max_steps,
            handover_only=args.handover_only
        )
    else:
        # Headless mode - run episodes without viewer
        success_count = 0

        for episode_num in range(args.num_episodes):
            print(f"\n--- Episode {episode_num + 1}/{args.num_episodes} ---")

            episode_data, success, review_frames = run_episode(
                setup=setup,
                box_x_range=box_x_range,
                box_y_range=box_y_range,
                max_steps=args.max_steps,
                record_training_data=True,
                record_every_n_steps=10,
                record_review_video=not args.no_video,
                review_video_every_n_steps=10,
                handover_only=args.handover_only
            )

            if success:
                success_count += 1
                print(f"✓ Episode {episode_num + 1} SUCCESS")
                if episode_data:
                    print(f"  - Recorded {len(episode_data.image_overhead)} timesteps")
                if review_frames:
                    print(f"  - Captured {len(review_frames)} video frames")
            else:
                print(f"✗ Episode {episode_num + 1} FAILED")

            # Save review video if enabled
            if review_frames and not args.no_video:
                try:
                    import mediapy
                    video_path = f"test_episode_{episode_num + 1:04d}_review.mp4"
                    mediapy.write_video(video_path, review_frames, fps=20)
                    print(f"  - Saved review video: {video_path}")
                except ImportError:
                    print("  - mediapy not available, skipping video save")

        print("\n" + "=" * 70)
        print(f"RESULTS: {success_count}/{args.num_episodes} episodes succeeded")
        print(f"Success rate: {100.0 * success_count / args.num_episodes:.1f}%")
        print("=" * 70)


def run_episode_with_viewer(setup, box_x_range, box_y_range, max_steps=2000, handover_only=False):
    """
    Run one episode with the interactive MuJoCo viewer for real-time visualization.
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]
    tasks = setup["tasks"]
    left_ee_task = setup["left_ee_task"]
    right_ee_task = setup["right_ee_task"]
    third_ee_task = setup["third_ee_task"]
    posture_task = setup["posture_task"]
    left_dof_ids = setup["left_dof_ids"]
    left_actuator_ids = setup["left_actuator_ids"]
    right_dof_ids = setup["right_dof_ids"]
    right_actuator_ids = setup["right_actuator_ids"]
    third_dof_ids = setup["third_dof_ids"]
    third_actuator_ids = setup["third_actuator_ids"]
    limits = setup["limits"]
    solver = setup["solver"]
    left_mocap_id = setup["left_mocap_id"]
    right_mocap_id = setup["right_mocap_id"]
    third_mocap_id = setup["third_mocap_id"]
    left_subtree_id = setup["left_subtree_id"]
    right_subtree_id = setup["right_subtree_id"]
    third_subtree_id = setup["third_subtree_id"]
    gripper_open = setup["GRIPPER_OPEN"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    left_gripper_actuator_id = setup["left_gripper_actuator_id"]
    right_gripper_actuator_id = setup["right_gripper_actuator_id"]
    third_gripper_actuator_id = setup["third_gripper_actuator_id"]
    GRASP_ORIENTATION = setup["GRASP_ORIENTATION"]
    LIFT_HEIGHT = setup["LIFT_HEIGHT"]
    box_joint_id = setup["box_joint_id"]
    box_spawn_height = setup["box_spawn_height"]

    # Reset to initial state
    if handover_only:
        mujoco.mj_resetDataKeyframe(model, data, model.key("handover_start").id)
        mujoco.mj_forward(model, data)

        grip0 = data.site('left/gripper').xpos.copy()
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        rest_pos_a = data.mocap_pos[left_mocap_id].copy()
        rest_quat_a = data.mocap_quat[left_mocap_id].copy()

        offset = np.array([0.0, np.random.uniform(-0.10, 0.10), np.random.uniform(-0.06, 0.06)])
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        data.mocap_pos[left_mocap_id] = grip0 + offset
        data.mocap_quat[left_mocap_id] = GRASP_ORIENTATION.wxyz
        left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
        mink.move_mocap_to_frame(model, data, "third/target", "third/gripper", "site")
        third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))

        goal_a = grip0 + offset
        for _ in range(1500):
            vel = mink.solve_ik(configuration, tasks, DT, limits=limits, solver=solver, damping=1e-5)
            configuration.integrate_inplace(vel, DT)
            data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
            data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
            data.ctrl[third_actuator_ids] = configuration.q[third_dof_ids]
            data.ctrl[left_gripper_actuator_id] = gripper_closed
            data.ctrl[right_gripper_actuator_id] = gripper_closed
            data.ctrl[third_gripper_actuator_id] = gripper_closed
            mujoco.mj_step(model, data)
            if np.linalg.norm(data.site('left/gripper').xpos - goal_a) < 0.005:
                break

        if not check_gripper_box_contact(model, data):
            print("WARNING: Box not in gripper at start!")

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        mink.move_mocap_to_frame(model, data, "third/target", "third/gripper", "site")
        mujoco.mj_forward(model, data)

        left_neutral_pos = rest_pos_a
        left_neutral_quat = rest_quat_a
        phase = "present_a"
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
        mink.move_mocap_to_frame(model, data, "third/target", "third/gripper", "site")
        mujoco.mj_forward(model, data)

        left_neutral_pos = data.mocap_pos[left_mocap_id].copy()
        left_neutral_quat = data.mocap_quat[left_mocap_id].copy()
        phase = "approach"

    right_neutral_pos = data.mocap_pos[right_mocap_id].copy()
    right_neutral_quat = data.mocap_quat[right_mocap_id].copy()
    third_neutral_pos = data.mocap_pos[third_mocap_id].copy()
    third_neutral_quat = data.mocap_quat[third_mocap_id].copy()

    gripper_ctrl = gripper_closed if handover_only else gripper_closed
    right_gripper_ctrl = gripper_closed
    third_gripper_ctrl = gripper_closed

    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
    third_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "third/target"))

    target_pos = data.site('left/gripper').xpos.copy()
    target_pos_b = data.site('right/gripper').xpos.copy()
    target_pos_c = data.site('third/gripper').xpos.copy()

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
        target_pos_c=target_pos_c,
        third_gripper_ctrl=third_gripper_ctrl,
        third_neutral_pos=third_neutral_pos,
        third_neutral_quat=third_neutral_quat,
    )

    # Launch viewer - use the simpler launch_passive API
    print("\nRunning episode in viewer...")
    print(f"Starting phase: {est.phase}")
    print("Note: Viewer will run for max_steps or until episode completes\n")

    step_count = 0
    # Use launch_passive which works on all platforms
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Main simulation loop
        while viewer.is_running() and est.phase != "done" and step_count < max_steps:
            action_a, action_b, action_c, est = expert_step(setup, est)

            data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
            data.ctrl[left_gripper_actuator_id] = est.gripper_ctrl
            data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
            data.ctrl[right_gripper_actuator_id] = est.right_gripper_ctrl
            data.ctrl[third_actuator_ids] = configuration.q[third_dof_ids]
            data.ctrl[third_gripper_actuator_id] = est.third_gripper_ctrl if est.third_gripper_ctrl is not None else gripper_open

            compensate_gravity(model, data, [left_subtree_id, right_subtree_id, third_subtree_id])
            mujoco.mj_step(model, data)

            viewer.sync()
            step_count += 1

            # Print phase changes
            if step_count % 50 == 0:
                print(f"Step {step_count}: Phase = {est.phase}")

        # Context manager handles closing automatically

    if est.phase == "done":
        print(f"\n✓ Episode completed successfully in {step_count} steps!")
    else:
        print(f"\n✗ Episode did not complete. Final phase: {est.phase} (step {step_count})")


if __name__ == "__main__":
    main()
