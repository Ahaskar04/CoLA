"""Scripted expert, rigid two-joint scanning variant.

Same pick-and-handover demonstration as scripted_policy.py, with a genuine
opening search driven by exactly TWO joints held under direct position control:

    left/wrist_angle  -- ELEVATION. Pitches the camera onto the spawn strip.
    left/waist        -- AZIMUTH.   Sweeps it left to right until the box appears.

Every other joint is frozen at the neutral keyframe for the whole search, and no
IK runs until the box has been found.

Why these two. Measured, not assumed (scripts/probe_scan_joints.pbs): over its
full range wrist_angle swings the optical axis 157 deg while translating the
camera only 0.24 m, where shoulder and elbow pitch comparably but drag the
camera 0.54 m and 0.76 m. waist is the only joint that pans -- 360 deg of
azimuth with zero elevation change. wrist_rotate merely rolls the image about
the view direction and forearm_roll does almost nothing to either angle.

Why this geometry. From the neutral pose the camera sits at [-0.38, -0.02, 0.40]
with its axis 19 deg below horizontal, looking out ACROSS the table -- the spawn
strip lies about 47 deg down, so it never enters the frame. That is why the
earlier vantage-based search timed out on all 232 collection attempts. Pitching
wrist_angle to SCAN_ELEVATION drops the axis to -47 deg, from which all 25
probed spawns are visible at some azimuth; the waist then sweeps a range chosen
to start with NOTHING in view and end past the last findable spawn.

What that buys. The sweep is a function of the clock alone, so the opening of
every episode is identical no matter where the box spawned, and the arm's own
joints are near-constant during it -- a policy cannot read the sweep clock out
of proprioception, it has to look. Detection azimuth then varies ~26 deg across
the spawn range and tracks box y monotonically, so WHEN the sweep stops says
where the box is along y, and where the box sits in frame says the rest.

Once found, the box is tracked by the wrist camera for the rest of the episode
(LOOK_AT_DEADBAND), so it stays visible from detection through the grasp and a
policy can servo on the current view instead of remembering the sweep.

`scan_found` is recorded per step. Episodes whose sweep timed out reached for a
box the camera never saw -- exactly the unobservable label this phase exists to
remove -- so filter on it before training.

Public API is identical to scripted_policy.py -- `setup`, `run_episode`,
`box_x_range`, `box_y_range`, `expert_step`, `ExpertState` -- so collect_demos.py
picks this up by swapping its import.
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from mink import SO3
from utils import setup_dual_arm_ik, compensate_gravity, check_gripper_box_contact, check_gripper_box_contact_right

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent  
_XML = _PROJECT_ROOT / "environments" / "handover_2arm" / "scene.xml"

setup = setup_dual_arm_ik(_XML)

box_x_range = (-0.12, 0.12)
box_y_range = (-0.10, 0.10)


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
        self.scan_found = []

    def add_step(self, image_overhead, image_wrist_a, image_wrist_b,
                 state_a, state_b, action_a, action_b, box_pos, phase,
                 scan_found):
        self.image_overhead.append(image_overhead)
        self.image_wrist_a.append(image_wrist_a)
        self.image_wrist_b.append(image_wrist_b)
        self.state_a.append(state_a)
        self.state_b.append(state_b)
        self.action_a.append(action_a)
        self.action_b.append(action_b)
        self.box_pos.append(box_pos)
        self.phase.append(phase)
        self.scan_found.append(scan_found)

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
            # False for the whole episode when the sweep timed out: the arm then
            # reached for a box the camera never saw. Filter these out.
            "scan_found": np.array(self.scan_found, dtype=bool),
        }



class ExpertState:
    """The scripted expert's carried state.

    The expert is a phase machine, so it cannot be queried from a bare
    (model, data) snapshot: `phase` says which stage it believes it is in, and
    `target_pos` / `target_pos_b` are IK goals that each phase MUTATES rather
    than recomputing from scratch. `configuration` is mink's own copy of the
    joint state. All of it has to be carried between steps.

    `scan_step` is the sweep clock for the opening search phase, and `scan_found`
    records whether that search ended by finding the box or by running out of
    sweep. Neither can be recovered from a (model, data) snapshot, so both are
    carried like everything else.
    """

    __slots__ = ('phase', 'target_pos', 'target_pos_b', 'gripper_ctrl',
                 'right_gripper_ctrl', 'left_neutral_pos', 'left_neutral_quat',
                 'right_neutral_pos', 'right_neutral_quat', 'scan_step',
                 'scan_found', 'left_neutral_q', 'right_neutral_q', 'ik_synced')

    def __init__(self, phase, target_pos, target_pos_b, gripper_ctrl,
                 right_gripper_ctrl, left_neutral_pos, left_neutral_quat,
                 right_neutral_pos, right_neutral_quat, scan_step=0,
                 scan_found=False, left_neutral_q=None, right_neutral_q=None,
                 ik_synced=False):
        self.phase = phase
        self.target_pos = target_pos
        self.target_pos_b = target_pos_b
        self.gripper_ctrl = gripper_ctrl
        self.right_gripper_ctrl = right_gripper_ctrl
        self.left_neutral_pos = left_neutral_pos
        self.left_neutral_quat = left_neutral_quat
        self.right_neutral_pos = right_neutral_pos
        self.right_neutral_quat = right_neutral_quat
        self.scan_step = scan_step
        self.scan_found = scan_found
        # Joint-space neutral pose. The search holds every joint here except the
        # two it sweeps, and the right arm holds it for the whole search.
        self.left_neutral_q = left_neutral_q
        self.right_neutral_q = right_neutral_q
        # mink's copy of the joint state goes stale while the search drives the
        # arm directly, so it is re-seeded from data.qpos on the way out.
        self.ik_synced = ik_synced

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)


# Constants the phase machine uses. Module level so expert_step() and
# run_episode() cannot drift apart.
DT = 1.0 / 200.0
APPROACH_HEIGHT_OFFSET = 0.10
POS_THRESHOLD = 0.02
B_APPROACH_HEIGHT_OFFSET = 0.08
RIGHT_TARGET_OFFSET = np.array([0.03, 0.0, 0.0])
A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0])
# Arm B takes the +x half of the box while arm A holds the -x half, mirroring
# A_GRASP_OFFSET. The box is 8 cm long in x, so +/-0.02 puts a gripper on each
# half without the two colliding.
#
# This replaces deriving arm B's target from arm A's COMMANDED target. That
# chained A_GRASP_OFFSET and RIGHT_TARGET_OFFSET onto a target_pos frozen at the
# lift -- carrying LIFT_HEIGHT (0.25) even though the box hangs in arm A's
# fingers ~1.7 cm lower -- so arm B closed on empty air for spawns near
# x ~ +0.02: 31/200 episodes with ZERO arm-B contacts, frozen in grip_b until
# max_steps. Reading the box directly is what arm A already does.
B_GRASP_OFFSET = np.array([0.02, 0.0, 0.0])

# --- opening search scan ----------------------------------------------------
# This phase exists so that the WRIST camera is what locates the box, which means
# every part of the motion has to be independent of where the box actually is --
# an arm that already flies to the box has not searched for anything, and a
# policy could only reproduce it by knowing the answer in advance.
#
# So: the arm travels to a FIXED vantage over the table and pans its gaze across
# the spawn region on a fixed schedule. The box enters the frame at a moment, and
# at a place in the frame, that depend on where it spawned -- that is the signal.
# The sweep ends when the box is found, so its DURATION carries information too.
# Joint-space search. Indices into the left arm's 6 joints, in the order
# setup_dual_arm_ik builds them: waist, shoulder, elbow, forearm_roll,
# wrist_angle, wrist_rotate.
WAIST_IDX = 0
WRIST_ANGLE_IDX = 4

# ELEVATION. 0.183 rad puts the optical axis 47 deg below horizontal, the only
# band from which all 25 probed spawns are reachable (0.08 -> 23/25, 0.18 ->
# 25/25, 0.29 -> 20/25). Held fixed for the whole search.
SCAN_ELEVATION = 0.183

# AZIMUTH. Spawns become visible between waist -0.400 and +0.050, so the sweep
# starts below that window and ends above it: nothing is in frame when the sweep
# begins (there would be no search otherwise) and every findable spawn is passed
# before it ends.
SCAN_AZ_START = -0.60
SCAN_AZ_END = 0.25
SCAN_STEPS = 600                              # 3.0 s of sweep at DT = 1/200

# Ramp from the neutral keyframe into the search pose. Both scan joints move
# together; the sweep clock does not start until they have arrived, so every
# episode gets the same full sweep regardless of settling time.
# The approach into the sweep start runs in TWO stages, and the order matters.
# Ramping both joints together walks the waist from 0 down to SCAN_AZ_START --
# straight through the detection window -- so the box was found during the ramp
# and 6 of 16 episodes never swept at all. Rotating FIRST at the neutral pitch is
# safe because at wrist_angle=-0.33 the camera is too flat to see any spawn at
# any azimuth (probe: 0/25), and pitching second happens at SCAN_AZ_START, which
# is outside the window. Nothing is findable until the sweep proper begins.
SCAN_YAW_STEPS = 300                          # stage 1: waist -> SCAN_AZ_START
SCAN_AIM_STEPS = 200                          # stage 2: wrist_angle -> SCAN_ELEVATION
SCAN_AIM_TOLERANCE = 0.02                     # rad, per joint

# "Found" = the box is within this cone of the optical axis. Well inside the
# ~29 deg half-FOV, so it means "squarely in frame", not "clipped the edge".
SCAN_DETECT_CONE = np.deg2rad(18.0)
# ...and genuinely visible: the ray from camera to box must arrive without
# hitting the arm on the way.
SCAN_REQUIRE_UNOCCLUDED = True

# --- wrist camera aiming ----------------------------------------------------
WRIST_CAM_NAME = "wrist_cam_left"
EE_SITE_NAME = "left/gripper"
# Once the box has been found, the camera tracks it for the rest of the episode,
# correcting only the part of the aiming error that exceeds this cone. Reaching
# down at the box leaves a ~14 deg residual error, so a 12 deg cone swallows
# almost all of it and the pick still runs on the original scripted grasp
# orientation, with the box near the middle of a ~29 deg half-FOV anyway.
# During the search the camera follows the sweep schedule instead -- see
# scan_gaze_point() for why it must not track the box there.
LOOK_AT_DEADBAND = np.deg2rad(12.0)
SCAN_PHASES = ("scan_yaw", "scan_aim", "scan")
# During retract_b the box belongs to arm B and arm A is finished; the original
# policy freezes arm A's target there. Flip this on to have arm A keep tracking
# the box with its wrist while arm B carries it away.
TRACK_BOX_DURING_RETRACT = False


def wrist_cam_axis_in_site_frame(model, data, cam_name=WRIST_CAM_NAME, site_name=EE_SITE_NAME):
    """The wrist camera's optical axis, expressed in EE-site coordinates.

    The camera hangs off the same rigid chain as the `left/gripper` site with no
    joint in between, so this direction is a constant of the model -- read it off
    once rather than re-deriving the mesh eulers by hand. MuJoCo cameras look
    down their own -Z.

    Runs mj_forward because a fresh MjData has no kinematics yet; that only
    recomputes derived quantities, it does not touch qpos/qvel.
    """
    mujoco.mj_forward(model, data)
    cam_id = model.camera(cam_name).id
    R_site = data.site(site_name).xmat.reshape(3, 3)
    R_cam = data.cam_xmat[cam_id].reshape(3, 3)
    return cam_id, R_site.T @ (-R_cam[:, 2])


def _wrist_cam_frame(setup):
    """Cached (cam_id, axis_local) for whichever setup dict is in play."""
    if "wrist_cam_axis_local" not in setup:
        cam_id, axis_local = wrist_cam_axis_in_site_frame(setup["model"], setup["data"])
        setup["wrist_cam_id"] = cam_id
        setup["wrist_cam_axis_local"] = axis_local
    return setup["wrist_cam_id"], setup["wrist_cam_axis_local"]


def look_at_orientation(base_rot, cam_pos, look_at, cam_axis_local,
                        deadband=LOOK_AT_DEADBAND):
    """`base_rot` nudged just far enough to put `look_at` in the wrist camera.

    Rotates the scripted grasp orientation about the axis that swings the
    camera's optical axis onto the target point, by the part of the aiming error
    that exceeds `deadband`. With the point already inside the cone this returns
    `base_rot` untouched, which is what keeps the descend/grasp part of the demo
    on the original scripted pose.

    The target is the box once it has been found, and the sweep's gaze point
    while the search is still running.

    `cam_pos` is where the camera ACTUALLY is this step, not where the IK has
    been told to put it. The two differ a lot while the arm is still swinging
    into the scan, and aiming from the real position is what stops the box
    sliding out of frame during that transit -- it makes the commanded
    orientation lead the lag instead of trailing it. Once the arm has converged
    the two coincide and this is just the aim.
    """
    to_target = look_at - cam_pos
    dist = np.linalg.norm(to_target)
    if dist < 1e-6:
        return base_rot
    desired = to_target / dist

    current = base_rot @ cam_axis_local
    angle = np.arccos(np.clip(current @ desired, -1.0, 1.0))
    if angle <= deadband:
        return base_rot
    axis = np.cross(current, desired)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return base_rot

    quat = np.empty(4)
    delta = np.empty(9)
    mujoco.mju_axisAngle2Quat(quat, axis / axis_norm, angle - deadband)
    mujoco.mju_quat2Mat(delta, quat)
    return delta.reshape(3, 3) @ base_rot


def _mat_to_wxyz(mat):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=np.float64).reshape(9))
    return quat


def rigid_scan_joints(neutral_q, scan_step, phase):
    """The left arm's 6 joint targets during the search.

    Only two entries ever move -- WAIST_IDX and WRIST_ANGLE_IDX -- and they move
    in three stages so that nothing is findable before the sweep:

      scan_yaw : waist rotates to SCAN_AZ_START at the NEUTRAL pitch, where the
                 camera is too flat to see any spawn at any azimuth.
      scan_aim : wrist_angle pitches to SCAN_ELEVATION, held at SCAN_AZ_START,
                 which is outside the azimuth window where spawns are visible.
      scan     : waist pans SCAN_AZ_START -> SCAN_AZ_END. This is the search.

    A function of the CLOCK ALONE -- no box position anywhere in it. That is what
    makes the opening identical in every episode, executable from an observation
    that does not yet contain the box, and therefore real evidence rather than a
    restatement of an answer the expert already had.
    """
    q = np.asarray(neutral_q, dtype=float).copy()
    if phase == "scan_yaw":
        t = min(scan_step / SCAN_YAW_STEPS, 1.0)
        q[WAIST_IDX] = neutral_q[WAIST_IDX] + t * (SCAN_AZ_START - neutral_q[WAIST_IDX])
    elif phase == "scan_aim":
        q[WAIST_IDX] = SCAN_AZ_START
        t = min(scan_step / SCAN_AIM_STEPS, 1.0)
        q[WRIST_ANGLE_IDX] = (neutral_q[WRIST_ANGLE_IDX]
                              + t * (SCAN_ELEVATION - neutral_q[WRIST_ANGLE_IDX]))
    else:
        progress = min(scan_step / SCAN_STEPS, 1.0)
        q[WRIST_ANGLE_IDX] = SCAN_ELEVATION
        q[WAIST_IDX] = SCAN_AZ_START + (SCAN_AZ_END - SCAN_AZ_START) * progress
    return q


def box_is_found(model, data, cam_id, box_pos, cone=SCAN_DETECT_CONE):
    """Is the box squarely inside the wrist camera's view this step?

    Stands in for the detector a vision policy would learn. It reads the box's
    true position, but only to answer the yes/no question the IMAGE could answer
    for itself, and only to decide WHEN the sweep stops -- never where the arm
    goes while it is sweeping. A pixel test on the rendered frame would be more
    faithful still, but expert_step() has to stay render-free to run as a DAgger
    observer, so the geometry answers it instead.
    """
    cam_pos = data.cam_xpos[cam_id]
    to_box = box_pos - cam_pos
    dist = np.linalg.norm(to_box)
    if dist < 1e-6:
        return False
    direction = to_box / dist

    optical_axis = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    if np.arccos(np.clip(optical_axis @ direction, -1.0, 1.0)) > cone:
        return False
    if not SCAN_REQUIRE_UNOCCLUDED:
        return True

    # mj_ray reports the first geom the ray meets; anything but the box means the
    # arm is in the way and the camera cannot actually see it yet.
    geomid = np.array([-1], dtype=np.int32)
    mujoco.mj_ray(model, data,
                  np.ascontiguousarray(cam_pos, dtype=np.float64),
                  np.ascontiguousarray(direction, dtype=np.float64),
                  None, 1, -1, geomid)
    return geomid[0] != -1 and model.geom(geomid[0]).name == "middle_box_geom"


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
    wrist_cam_id, cam_axis_local = _wrist_cam_frame(setup)

    if sync_configuration:
        configuration.update(data.qpos)

    box_pos = data.joint('middle_box_joint').qpos[:3]
    phase = est.phase

    # ---- rigid search: two joints under direct position control, no IK -------
    if phase in SCAN_PHASES:
        q_left = rigid_scan_joints(est.left_neutral_q, est.scan_step, phase)
        est.scan_step += 1

        def arrived(idx):
            return abs(data.qpos[left_dof_ids[idx]] - q_left[idx]) < SCAN_AIM_TOLERANCE

        found = box_is_found(model, data, wrist_cam_id, box_pos)
        if found:
            # Stop the moment the box is in view. Stages 1 and 2 are built so
            # this cannot fire before the sweep; it is a safety net, and
            # scan_found records that the box was genuinely seen.
            est.scan_found = True
            est.phase = "approach"
        elif phase == "scan_yaw":
            if est.scan_step >= SCAN_YAW_STEPS and arrived(WAIST_IDX):
                est.scan_step = 0
                est.phase = "scan_aim"
        elif phase == "scan_aim":
            if est.scan_step >= SCAN_AIM_STEPS and arrived(WRIST_ANGLE_IDX):
                est.scan_step = 0          # restart the clock for the sweep proper
                est.phase = "scan"
        elif est.scan_step >= SCAN_STEPS:
            # Timed out. Fall through so the episode still yields a demo, but
            # scan_found stays False and it should be filtered out: the arm is
            # about to reach for a box the camera never saw.
            est.phase = "approach"

        if est.phase not in SCAN_PHASES:
            # Hand the arm back to mink from wherever the search actually left it.
            configuration.update(data.qpos)
            est.ik_synced = True

        action_a = np.concatenate([q_left, [est.gripper_ctrl]])
        action_b = np.concatenate([est.right_neutral_q, [est.right_gripper_ctrl]])
        return action_a, action_b, est

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

    elif phase == "move_arm_B":
        est.target_pos_b = box_pos + B_GRASP_OFFSET + np.array([0.0, 0.0, B_APPROACH_HEIGHT_OFFSET])
        data.mocap_pos[right_mocap_id] = est.target_pos_b
        right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
        est.right_gripper_ctrl = gripper_open

    elif phase == "approach_b":
        est.target_pos_b = box_pos + B_GRASP_OFFSET
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

    # The search returns above, so every phase reaching here tracks the box.
    aim_point = box_pos
    aim_deadband = LOOK_AT_DEADBAND

    if phase != "retract_b" or TRACK_BOX_DURING_RETRACT:
        # Inside the aim cone this IS the fixed scripted grasp orientation.
        target_rot = look_at_orientation(
            GRASP_ORIENTATION.as_matrix(), data.cam_xpos[wrist_cam_id], aim_point,
            cam_axis_local, deadband=aim_deadband,
        )
        data.mocap_pos[left_mocap_id] = est.target_pos
        data.mocap_quat[left_mocap_id] = _mat_to_wxyz(target_rot)
        left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    vel = mink.solve_ik(configuration, tasks, DT, limits=limits, solver=solver, damping=1e-5)
    configuration.integrate_inplace(vel, DT)

    gripper_qpos = data.joint('left/left_finger').qpos[0]
    gripper_is_open = abs(gripper_qpos - gripper_open) < 0.002
    current_gripper_pos = data.site('left/gripper').xpos
    dist_to_target = np.linalg.norm(current_gripper_pos - est.target_pos)
    right_gripper_pos = data.site('right/gripper').xpos
    dist_right_to_target = np.linalg.norm(right_gripper_pos - est.target_pos_b)

    if phase == "approach" and dist_to_target < POS_THRESHOLD:
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
    elif phase == "approach_b" and dist_right_to_target < POS_THRESHOLD:
        est.phase = "grip_b"
    elif phase == "grip_b" and check_gripper_box_contact_right(model, data):
        est.phase = "release_a"
    elif phase == "release_a" and gripper_is_open:
        est.phase = "retract_b"
    elif phase == "retract_b" and np.linalg.norm(right_gripper_pos - est.right_neutral_pos) < POS_THRESHOLD:
        est.phase = "done"

    action_a = np.concatenate([configuration.q[left_dof_ids], [est.gripper_ctrl]])
    action_b = np.concatenate([configuration.q[right_dof_ids], [est.right_gripper_ctrl]])
    return action_a, action_b, est


def run_episode(setup, box_x_range, box_y_range, max_steps=3200,
                 record_training_data=True, record_every_n_steps=10, 
                 record_review_video=True,
                 review_video_every_n_steps=10):
    """
    Run one full scan-then-pick-and-handover episode, headless (no live viewer).

    max_steps is larger than the non-scanning policy's 2000 because the opening
    search costs a reach to the vantage plus up to SCAN_STEPS of sweep before the
    pick even starts.

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
    
    left_neutral_pos = data.mocap_pos[left_mocap_id].copy()
    left_neutral_quat = data.mocap_quat[left_mocap_id].copy()

    right_neutral_pos = data.mocap_pos[right_mocap_id].copy()
    right_neutral_quat = data.mocap_quat[right_mocap_id].copy()

    phase = "scan_yaw"
    gripper_ctrl = gripper_closed
    right_gripper_ctrl = gripper_closed
    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    episode_data = EpisodeData() if record_training_data else None
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    step_count = 0
    target_pos = data.site('left/gripper').xpos.copy()  
    target_pos_b = data.site('right/gripper').xpos.copy()

    # Single source of truth: the phase machine lives in expert_step() so the
    # DAgger collector queries exactly the same expert this records.
    # Joint-space neutral pose, read after the keyframe reset. The rigid search
    # holds every joint here except the two it sweeps.
    left_neutral_q = data.qpos[left_dof_ids].copy()
    right_neutral_q = data.qpos[right_dof_ids].copy()

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
        scan_step=0,
        left_neutral_q=left_neutral_q,
        right_neutral_q=right_neutral_q,
    )

    while est.phase != "done" and step_count < max_steps:
        box_pos = data.joint('middle_box_joint').qpos[:3]
        action_a, action_b, est = expert_step(setup, est)
        phase = est.phase
        gripper_ctrl = est.gripper_ctrl
        right_gripper_ctrl = est.right_gripper_ctrl
        gripper_qpos = data.joint('left/left_finger').qpos[0]
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]

        # Drive from the returned actions rather than configuration.q: during the
        # rigid search no IK runs, so mink's copy of the joint state is stale and
        # would command the arm back to wherever it last solved. In the IK phases
        # action_a[:6] IS configuration.q[left_dof_ids], so this is equivalent.
        data.ctrl[left_actuator_ids] = action_a[:6]
        data.ctrl[left_gripper_actuator_id] = gripper_ctrl
        data.ctrl[right_actuator_ids] = action_b[:6]
        data.ctrl[right_gripper_actuator_id] = right_gripper_ctrl

        compensate_gravity(model, data, [left_subtree_id, right_subtree_id])
        mujoco.mj_step(model, data)

        # --- capture training data every step ---
        if record_training_data and step_count % record_every_n_steps == 0:
            state_a = np.concatenate([data.qpos[left_dof_ids], [gripper_qpos]])
            state_b = np.concatenate([data.qpos[right_dof_ids], [right_gripper_qpos]])
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
                scan_found=est.scan_found,
            )

        # --- capture review video, sparsely (not every step, for speed) ---
        if record_review_video and step_count % review_video_every_n_steps == 0:
            renderer.update_scene(data, camera="overhead_cam")
            review_frames.append(renderer.render())

        step_count += 1

    if renderer is not None:
        renderer.close()

    success = (phase == "done")
    return episode_data, success, review_frames
