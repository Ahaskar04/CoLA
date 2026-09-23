"""Scripted expert, frozen-arm scanning variant (azimuth sweep, table-aimed).

Same pick-and-handover demonstration as scripted_policy.py, with a genuine
opening search.

    Arm A reaches a fixed vantage over the table, FREEZES every joint at
    whatever that reach converged to, and then sweeps its wrist camera along the
    table until the box comes into view.

The sweep is one dimensional: azimuth is the clock, and elevation is SOLVED at
each step so the camera's optical axis meets the table plane. That solve uses
only the camera's current pose and the table height -- both episode-invariant --
so the whole opening is byte-identical no matter where the box spawned, which is
what makes the wrist view during the search real evidence about the box rather
than a restatement of an answer the expert already had.

Why elevation is solved rather than tiered. An earlier version rastered a few
discrete elevation tiers across a wide range and detected nothing. The reason is
geometric: for a fixed camera position and a fixed target point, the set of
(azimuth, elevation) that puts the target inside a 10 deg cone is a thin CURVE
in the two-dimensional angle space, not a region. Tiers spaced further apart
than the cone width step straight over it. Covering a 4 rad elevation span at
0.175 rad spacing would need ~24 tiers and ~6000 steps. Solving elevation from
the table plane rides along that curve instead, and one pass covers the whole
spawn strip.

Why the arm is frozen. When the sweep runs through IK, mink is free to move any
joint to satisfy the end-effector task, so the camera translates as well as
rotates and "box at pixel (u,v)" no longer pins down where the box is. Held
still, the camera POSITION is constant for the whole sweep and box position maps
to a bearing along a known ray. Freezing also keeps arm A's proprioception
nearly constant during the search, so a policy cannot use joint state as a proxy
for the sweep clock -- it has to read the image, which is the behaviour this
phase exists to teach.

run_episode returns `scan_found` alongside `success`. Episodes where the sweep
timed out MUST BE DISCARDED: the phase machine falls through to `approach` and
reaches for a box the camera never saw, which is exactly the unobservable label
this whole phase exists to remove -- concentrated, worse, in the spawns that are
hardest to see.

BEFORE COLLECTING ANYTHING:

    python scripted_policy_scan.py --probe    # what azimuths see each spawn
    python scripted_policy_scan.py            # coverage of the configured sweep

--probe searches a window AROUND the frozen pose, not the joint's full range.
Full-range search finds wrap-around solutions -- waist swung 3 rad backwards
with the wrist flipped to compensate -- which are real, useless, and poison any
min/max summary of where to sweep.
"""
from pathlib import Path
import argparse
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
        }


class ExpertState:
    """The scripted expert's carried state.

    The expert is a phase machine, so it cannot be queried from a bare
    (model, data) snapshot: `phase` says which stage it believes it is in, and
    `target_pos` / `target_pos_b` are IK goals that each phase MUTATES rather
    than recomputing from scratch. `configuration` is mink's own copy of the
    joint state. All of it has to be carried between steps.

    The scan fields likewise: `scan_step` is the sweep clock, `scan_hold_qpos`
    is the frozen configuration the sweep rotates within, `scan_dof_azim` and
    `scan_dof_elev` are the two panning joints' indices inside it,
    `scan_elev_hint` is the previous step's solved elevation (the seed for this
    step's solve), `scan_dwell` counts consecutive in-cone steps, and
    `scan_found` records whether the search ended by finding the box or by
    running out of sweep.
    """

    __slots__ = ('phase', 'target_pos', 'target_pos_b', 'gripper_ctrl',
                 'right_gripper_ctrl', 'left_neutral_pos', 'left_neutral_quat',
                 'right_neutral_pos', 'right_neutral_quat', 'scan_step',
                 'scan_found', 'scan_hold_qpos', 'scan_dof_azim',
                 'scan_dof_elev', 'scan_elev_hint', 'scan_dwell',
                 'approach_ramp')

    def __init__(self, phase, target_pos, target_pos_b, gripper_ctrl,
                 right_gripper_ctrl, left_neutral_pos, left_neutral_quat,
                 right_neutral_pos, right_neutral_quat, scan_step=0,
                 scan_found=False):
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
        self.scan_hold_qpos = None
        self.scan_dof_azim = None
        self.scan_dof_elev = None
        self.scan_elev_hint = None
        self.scan_dwell = 0
        self.approach_ramp = 0

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)


# Constants the phase machine uses. Module level so expert_step() and
# run_episode() cannot drift apart.
DT = 1.0 / 200.0
APPROACH_HEIGHT_OFFSET = 0.10
POS_THRESHOLD = 0.02
B_APPROACH_HEIGHT_OFFSET = 0.08
A_GRASP_OFFSET = np.array([-0.02, 0.0, 0.0])
# Arm B takes the +x half of the box while arm A holds the -x half, mirroring
# A_GRASP_OFFSET. The box is 8 cm long in x, so +/-0.02 puts a gripper on each
# half without the two colliding.
B_GRASP_OFFSET = np.array([0.02, 0.0, 0.0])

# --- opening search ---------------------------------------------------------
SCAN_VANTAGE = np.array([-0.20, 0.0, 0.28])   # fixed EE position for the search

# Azimuth is the sweep clock: rotating the whole arm about vertical pans the
# camera along the table with almost no change in camera position.
SCAN_AZIM_JOINT = 'left/waist'
# Elevation is SOLVED each step so the optical axis meets the table plane. This
# joint tilts the camera up and down about the wrist -- the smallest camera
# translation of any joint that changes pitch. Shoulder or elbow would swing the
# camera bodily across the workspace and destroy the fixed-viewpoint property.
SCAN_ELEV_JOINT = 'left/wrist_angle'

# Azimuth sweep limits, in radians, RELATIVE to the frozen pose's azimuth. The
# sweep runs start -> end. Run --probe and widen until every spawn is covered;
# relative offsets are used so these stay meaningful if SCAN_VANTAGE moves.
SCAN_AZIM_REL_START = -0.70
SCAN_AZIM_REL_END = 0.70
SCAN_STEPS = 600                              # 3.0 s of sweep at DT = 1/200

# The plane the camera is aimed at while sweeping: the box's resting centre
# height on the table. Episode-invariant, so using it introduces no dependence
# on where the box actually is.
SCAN_TABLE_Z = 0.02
# The elevation solve is a bounded bisection about the previous step's answer.
# The camera moves a fraction of a degree per step, so a tight window converges
# in a few iterations and cannot jump to a wrap-around branch.
SCAN_ELEV_SEARCH_HALFWIDTH = 0.35
SCAN_ELEV_ITERS = 24

# "Found" = the box is within this cone of the optical axis. Tight, so the box
# sits near the MIDDLE of the frame at the moment the action changes -- a far
# more distinctive image for a policy to key on than one clipped at the edge of
# a ~29 deg half-FOV.
SCAN_DETECT_CONE = np.deg2rad(10.0)
# ...held for this many consecutive steps. Removes the knife edge where one
# in-cone frame flips the phase and behavioural cloning has to resolve an
# ambiguity band around the transition.
SCAN_DWELL_STEPS = 8
# ...and genuinely visible: the ray from camera to box must arrive without
# hitting the arm on the way.
SCAN_REQUIRE_UNOCCLUDED = True

# Steps over which the approach target is ramped out of the scan pose. Without
# this, target_pos jumps from the vantage to the box in a single step and the
# policy has to time a large discontinuity exactly.
APPROACH_RAMP_STEPS = 20

# --- wrist camera aiming ----------------------------------------------------
WRIST_CAM_NAME = "wrist_cam_left"
EE_SITE_NAME = "left/gripper"
LOOK_AT_DEADBAND = np.deg2rad(12.0)
TRACK_BOX_DURING_RETRACT = False


def wrist_cam_axis_in_site_frame(model, data, cam_name=WRIST_CAM_NAME, site_name=EE_SITE_NAME):
    """The wrist camera's optical axis, expressed in EE-site coordinates.

    The camera hangs off the same rigid chain as the `left/gripper` site with no
    joint in between, so this direction is a constant of the model -- read it off
    once rather than re-deriving the mesh eulers by hand. MuJoCo cameras look
    down their own -Z.
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
    `base_rot` untouched, which keeps descend/grasp on the original scripted pose.
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


def scan_azimuth(scan_step, azim0):
    """Commanded azimuth at `scan_step`, absolute, given the frozen pose's azimuth.

    A function of the CLOCK ALONE apart from azim0, which is itself
    box-independent -- the reach to the vantage is the same motion in every
    episode. Nothing here reads the box.
    """
    progress = min(max(scan_step, 0) / max(SCAN_STEPS, 1), 1.0)
    return azim0 + SCAN_AZIM_REL_START + (SCAN_AZIM_REL_END - SCAN_AZIM_REL_START) * progress


def _optical_axis_pitch(model, data, cam_id, hold, dof_elev, elev):
    """Set the elevation joint, refresh kinematics, return (cam_pos, axis).
    """
    saved = float(data.qpos[dof_elev])
    data.qpos[dof_elev] = elev
    mujoco.mj_forward(model, data)
    axis = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    cam_pos = data.cam_xpos[cam_id].copy()
    data.qpos[dof_elev] = saved
    return cam_pos, axis


def solve_elevation_for_table(model, data, cam_id, hold, dof_elev,
                              hint, table_z=SCAN_TABLE_Z,
                              halfwidth=SCAN_ELEV_SEARCH_HALFWIDTH,
                              iters=SCAN_ELEV_ITERS):
    """Elevation whose optical axis meets the plane z = table_z.

    Bisection on the axis's z-component: too shallow and the ray never comes
    down to the table, too steep and it hits the near floor in front of the arm.
    The bracket is a narrow window around `hint` (the previous step's answer),
    which keeps the solve on the same branch -- a full-range search would happily
    return a wrap-around pose with the wrist flipped a whole turn.

    NOTE this reads NOTHING about the box. The table height is a property of the
    scene, identical in every episode, so aiming at it leaves the sweep entirely
    box-independent.

    Returns the solved elevation, or `hint` unchanged if no bracket exists at
    this azimuth (the ray cannot be made to meet the table from here).
    """
    lo, hi = hint - halfwidth, hint + halfwidth

    def dz(elev):
        # How far below the camera the axis points, minus how far the table is.
        # Zero when the ray, extended, lands on the table plane.
        cam_pos, axis = _optical_axis_pitch(model, data, cam_id, hold, dof_elev, elev)
        if axis[2] > -1e-3:
            # Pointing level or up: the ray never reaches the table.
            return 1e3
        t = (table_z - cam_pos[2]) / axis[2]
        return t - np.linalg.norm(cam_pos[:2] - (cam_pos[:2] + axis[:2] * t))

    f_lo, f_hi = dz(lo), dz(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return hint

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = dz(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def aim_elevation_at_point(model, data, cam_id, hold, dof_elev, point,
                           hint, halfwidth=SCAN_ELEV_SEARCH_HALFWIDTH,
                           iters=SCAN_ELEV_ITERS):
    """Elevation that minimises the angle between the optical axis and `point`.

    Used ONLY by the diagnostics below, never by expert_step -- it reads the
    target's position, which the sweep itself must not do.
    """
    def err(elev):
        cam_pos, axis = _optical_axis_pitch(model, data, cam_id, hold, dof_elev, elev)
        to_p = point - cam_pos
        n = np.linalg.norm(to_p)
        if n < 1e-9:
            return np.pi
        return np.arccos(np.clip(axis @ (to_p / n), -1.0, 1.0))

    lo, hi = hint - halfwidth, hint + halfwidth
    # Golden-section, since err() is unimodal in elevation near the solution.
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = err(c), err(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = err(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = err(d)
    best = 0.5 * (a + b)
    return best, err(best)


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

    NOTE: this writes data.mocap_pos/quat, because mink reads the IK target back
    out of the mocap bodies. The marker geoms are alpha 0, so this does not
    change what any camera sees. Worth verifying by eye -- a visible marker at
    the IK target would be a near-perfect label the policy learns to read, and it
    stops moving at rollout.
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

    # ---- decide the target for this phase --------------------------------
    if phase == "scan_start":
        est.target_pos = SCAN_VANTAGE.copy()

    elif phase == "scan":
        if est.scan_hold_qpos is None:
            # Freeze HERE: whatever scan_start converged to is the configuration
            # the whole sweep rotates within. Captured once, never recomputed --
            # re-reading it each step would let the arm drift.
            est.scan_hold_qpos = np.array(configuration.q).copy()
            est.scan_dof_azim = int(model.joint(SCAN_AZIM_JOINT).qposadr[0])
            est.scan_dof_elev = int(model.joint(SCAN_ELEV_JOINT).qposadr[0])
            est.scan_elev_hint = float(est.scan_hold_qpos[est.scan_dof_elev])
        est.scan_step += 1

    elif phase == "approach":
        goal = box_pos + A_GRASP_OFFSET + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])
        # Ramp out of the scan pose rather than jumping. The arm ends the sweep
        # at an azimuth that depends on where the box was found, so this
        # transition starts from a spawn-dependent pose.
        if est.approach_ramp < APPROACH_RAMP_STEPS:
            est.approach_ramp += 1
            alpha = est.approach_ramp / APPROACH_RAMP_STEPS
            start = data.site(EE_SITE_NAME).xpos.copy()
            est.target_pos = start + alpha * (goal - start)
        else:
            est.target_pos = goal

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

    # ---- move arm A ------------------------------------------------------
    if phase == "scan":
        # Direct joint command, NO IK. This is what freezes the arm: mink is
        # never asked to satisfy an end-effector task during the sweep, so it
        # cannot move any joint but the two below.
        azim0 = float(est.scan_hold_qpos[est.scan_dof_azim])
        azim = scan_azimuth(est.scan_step, azim0)

        q = est.scan_hold_qpos.copy()
        q[est.scan_dof_azim] = azim
        # Solve elevation at THIS azimuth so the axis meets the table. Seeded
        # from the previous step's answer, which keeps the solve on one branch.
        elev = solve_elevation_for_table(
            model, data, wrist_cam_id, q, est.scan_dof_elev, est.scan_elev_hint)
        est.scan_elev_hint = elev
        q[est.scan_dof_elev] = elev

        configuration.update(q)
        data.qpos[est.scan_dof_azim] = azim
        data.qpos[est.scan_dof_elev] = elev
        mujoco.mj_forward(model, data)
    else:
        if phase != "retract_b" or TRACK_BOX_DURING_RETRACT:
            aim_point = SCAN_VANTAGE if phase == "scan_start" else box_pos
            aim_deadband = 0.0 if phase == "scan_start" else LOOK_AT_DEADBAND
            target_rot = look_at_orientation(
                GRASP_ORIENTATION.as_matrix(), data.cam_xpos[wrist_cam_id],
                aim_point, cam_axis_local, deadband=aim_deadband,
            )
            data.mocap_pos[left_mocap_id] = est.target_pos
            data.mocap_quat[left_mocap_id] = _mat_to_wxyz(target_rot)
            left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

        vel = mink.solve_ik(configuration, tasks, DT, limits=limits,
                            solver=solver, damping=1e-5)
        configuration.integrate_inplace(vel, DT)

    # ---- advance the phase -----------------------------------------------
    gripper_qpos = data.joint('left/left_finger').qpos[0]
    gripper_is_open = abs(gripper_qpos - gripper_open) < 0.002
    current_gripper_pos = data.site('left/gripper').xpos
    dist_to_target = np.linalg.norm(current_gripper_pos - est.target_pos)
    right_gripper_pos = data.site('right/gripper').xpos
    dist_right_to_target = np.linalg.norm(right_gripper_pos - est.target_pos_b)

    if phase == "scan_start" and dist_to_target < POS_THRESHOLD:
        est.phase = "scan"
    elif phase == "scan":
        if box_is_found(model, data, wrist_cam_id, box_pos):
            est.scan_dwell += 1
        else:
            est.scan_dwell = 0

        if est.scan_dwell >= SCAN_DWELL_STEPS:
            est.scan_found = True
            est.phase = "approach"
        elif est.scan_step >= SCAN_STEPS:
            # Timed out. The episode continues so the harness does not stall,
            # but scan_found stays False and run_episode reports it -- these MUST
            # be discarded, not trained on.
            est.phase = "approach"
    elif phase == "approach" and dist_to_target < POS_THRESHOLD \
            and est.approach_ramp >= APPROACH_RAMP_STEPS:
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


def run_episode(setup, box_x_range, box_y_range, max_steps=3600,
                record_training_data=True, record_every_n_steps=10,
                record_review_video=True,
                review_video_every_n_steps=10):
    """
    Run one full scan-then-pick-and-handover episode, headless.

    max_steps allows for the reach to the vantage plus up to SCAN_STEPS of sweep
    before the pick even starts.

    Returns:
        episode_data:  EpisodeData object (or None if record_training_data=False)
        success:       bool, True if the episode reached the "done" phase
        review_frames: rendered frames for a human-review video (or None)
        scan_found:    bool, False if the sweep timed out. DISCARD THOSE.
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]
    right_ee_task = setup["right_ee_task"]
    left_ee_task = setup["left_ee_task"]
    posture_task = setup["posture_task"]
    left_dof_ids = setup["left_dof_ids"]
    left_actuator_ids = setup["left_actuator_ids"]
    right_dof_ids = setup["right_dof_ids"]
    right_actuator_ids = setup["right_actuator_ids"]
    left_mocap_id = setup["left_mocap_id"]
    right_mocap_id = setup["right_mocap_id"]
    left_subtree_id = setup["left_subtree_id"]
    right_subtree_id = setup["right_subtree_id"]
    gripper_closed = setup["GRIPPER_CLOSED"]
    left_gripper_actuator_id = setup["left_gripper_actuator_id"]
    right_gripper_actuator_id = setup["right_gripper_actuator_id"]
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

    right_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))
    left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))

    episode_data = EpisodeData() if record_training_data else None
    review_frames = [] if record_review_video else None
    renderer = mujoco.Renderer(model, 256, 256) if (record_training_data or record_review_video) else None

    est = ExpertState(
        phase="scan_start",
        target_pos=data.site('left/gripper').xpos.copy(),
        target_pos_b=data.site('right/gripper').xpos.copy(),
        gripper_ctrl=gripper_closed,
        right_gripper_ctrl=gripper_closed,
        left_neutral_pos=left_neutral_pos,
        left_neutral_quat=left_neutral_quat,
        right_neutral_pos=right_neutral_pos,
        right_neutral_quat=right_neutral_quat,
        scan_step=0,
    )

    step_count = 0
    while est.phase != "done" and step_count < max_steps:
        box_pos = data.joint('middle_box_joint').qpos[:3].copy()
        # The phase that PRODUCED this step's action, captured before
        # expert_step advances the machine. Recording est.phase afterwards would
        # label each frame with the NEXT phase and shift every phase-bucketed
        # analysis by one recorded step.
        phase_now = est.phase

        action_a, action_b, est = expert_step(setup, est)

        gripper_qpos = data.joint('left/left_finger').qpos[0]
        right_gripper_qpos = data.joint('right/left_finger').qpos[0]

        data.ctrl[left_actuator_ids] = configuration.q[left_dof_ids]
        data.ctrl[left_gripper_actuator_id] = est.gripper_ctrl
        data.ctrl[right_actuator_ids] = configuration.q[right_dof_ids]
        data.ctrl[right_gripper_actuator_id] = est.right_gripper_ctrl

        compensate_gravity(model, data, [left_subtree_id, right_subtree_id])
        mujoco.mj_step(model, data)

        if record_training_data and step_count % record_every_n_steps == 0:
            state_a = np.concatenate([data.qpos[left_dof_ids], [gripper_qpos]])
            state_b = np.concatenate([data.qpos[right_dof_ids], [right_gripper_qpos]])

            episode_data.add_step(
                image_overhead=render_camera(model, data, renderer, "overhead_cam"),
                image_wrist_a=render_camera(model, data, renderer, "wrist_cam_left"),
                image_wrist_b=render_camera(model, data, renderer, "wrist_cam_right"),
                state_a=state_a,
                state_b=state_b,
                # The actions expert_step actually returned, not a second
                # reconstruction -- one definition of the layout, not two.
                action_a=action_a,
                action_b=action_b,
                box_pos=box_pos,
                phase=phase_now,
            )

        # One frame per review_video_every_n_steps sim steps = one frame per
        # 0.02 s, so these play back real-time at fps=50, not 20 or 30.
        if record_review_video and step_count % review_video_every_n_steps == 0:
            renderer.update_scene(data, camera="overhead_cam")
            review_frames.append(renderer.render())

        step_count += 1

    if renderer is not None:
        renderer.close()

    success = (est.phase == "done")
    return episode_data, success, review_frames, bool(est.scan_found)


# --------------------------------------------------------------------------
# Diagnostics -- run BOTH before collecting a dataset
# --------------------------------------------------------------------------

def _reach_vantage_and_freeze(setup, x, y):
    """Drive arm A to the vantage with the box at (x, y), then freeze.

    Returns (frozen_qpos, azim_dof, elev_dof), or None if the arm never arrives.
    """
    model = setup["model"]
    data = setup["data"]
    configuration = setup["configuration"]

    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    data.qpos[setup["box_joint_id"]: setup["box_joint_id"] + 3] = [
        x, y, setup["box_spawn_height"]]
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
    mujoco.mj_forward(model, data)

    # Same initialisation run_episode does. Without it the RIGHT arm's FrameTask
    # has no target and solve_ik raises TargetNotSet on the first call -- the
    # diagnostics never reach a phase that would set it.
    setup["posture_task"].set_target_from_configuration(configuration)
    setup["left_ee_task"].set_target(
        mink.SE3.from_mocap_name(model, data, "left/target"))
    setup["right_ee_task"].set_target(
        mink.SE3.from_mocap_name(model, data, "right/target"))

    est = ExpertState(
        phase="scan_start",
        target_pos=SCAN_VANTAGE.copy(),
        target_pos_b=data.site('right/gripper').xpos.copy(),
        gripper_ctrl=setup["GRIPPER_CLOSED"],
        right_gripper_ctrl=setup["GRIPPER_CLOSED"],
        left_neutral_pos=data.mocap_pos[setup["left_mocap_id"]].copy(),
        left_neutral_quat=data.mocap_quat[setup["left_mocap_id"]].copy(),
        right_neutral_pos=data.mocap_pos[setup["right_mocap_id"]].copy(),
        right_neutral_quat=data.mocap_quat[setup["right_mocap_id"]].copy(),
    )

    for _ in range(3000):
        expert_step(setup, est)
        data.qpos[:] = configuration.q
        mujoco.mj_forward(model, data)
        if est.phase == "scan":
            break

    if est.phase != "scan":
        return None
    return (np.array(configuration.q).copy(),
            int(model.joint(SCAN_AZIM_JOINT).qposadr[0]),
            int(model.joint(SCAN_ELEV_JOINT).qposadr[0]))


def scan_probe_grid(setup, nx=3, ny=3, n_azim=241, verbose=True):
    """For each spawn, the azimuth OFFSET at which the box comes into view.

    At each candidate azimuth the elevation is solved to aim at the box -- the
    best case that azimuth can achieve. So this reports whether an azimuth-only
    sweep with a table-aimed elevation can ever see the spawn, and where.

    Deliberately searches a WINDOW around the frozen pose rather than the joint's
    full range. Full-range search finds wrap-around solutions -- the waist swung
    3 rad backwards with the wrist flipped to compensate -- which are real,
    useless, and make a min/max summary meaningless.

    The offsets printed here are what to paste into SCAN_AZIM_REL_START/END.
    """
    model = setup["model"]
    data = setup["data"]
    wrist_cam_id, _ = _wrist_cam_frame(setup)

    window = max(abs(SCAN_AZIM_REL_START), abs(SCAN_AZIM_REL_END)) * 1.5 + 0.3
    if verbose:
        print(f'PROBE: azimuth offsets within +/-{window:.2f} rad of the frozen pose')
        print(f'  azim {SCAN_AZIM_JOINT}, {n_azim} samples')
        print(f'  elev {SCAN_ELEV_JOINT}, solved per azimuth to aim at the box')
        print(f'  detect cone {np.rad2deg(SCAN_DETECT_CONE):.0f} deg\n')
        print(f'  {"x":>8}{"y":>8}{"n_hit":>7}{"offs_lo":>10}{"offs_hi":>10}'
              f'{"best_off":>10}{"elev":>9}')

    offs = np.linspace(-window, window, n_azim)
    rows = []

    for x in np.linspace(box_x_range[0], box_x_range[1], nx):
        for y in np.linspace(box_y_range[0], box_y_range[1], ny):
            frozen = _reach_vantage_and_freeze(setup, x, y)
            if frozen is None:
                if verbose:
                    print(f'  {x:>8.3f}{y:>8.3f}   never reached the vantage')
                continue
            hold, dof_a, dof_e = frozen
            azim0 = float(hold[dof_a])
            elev0 = float(hold[dof_e])
            box_pos = data.joint('middle_box_joint').qpos[:3].copy()

            hits = []
            hint = elev0
            for off in offs:
                q = hold.copy()
                q[dof_a] = azim0 + off
                elev, err = aim_elevation_at_point(
                    model, data, wrist_cam_id, q, dof_e, box_pos, hint)
                hint = elev
                if err <= SCAN_DETECT_CONE:
                    q[dof_e] = elev
                    data.qpos[:len(q)] = q
                    mujoco.mj_forward(model, data)
                    if box_is_found(model, data, wrist_cam_id, box_pos):
                        hits.append((off, elev, err))

            if hits:
                o = [h[0] for h in hits]
                best = min(hits, key=lambda h: h[2])
                rows.append((x, y, min(o), max(o), best[0], best[1]))
                if verbose:
                    print(f'  {x:>8.3f}{y:>8.3f}{len(hits):>7d}{min(o):>10.3f}'
                          f'{max(o):>10.3f}{best[0]:>10.3f}{best[1]:>9.3f}')
            elif verbose:
                print(f'  {x:>8.3f}{y:>8.3f}   NEVER VISIBLE at any azimuth in window')

    if verbose:
        print()
        if not rows:
            print('  No spawn is visible from this vantage at any azimuth in the window,')
            print('  even with elevation aimed straight at the box. The vantage itself is')
            print('  wrong: move SCAN_VANTAGE back and up so the whole spawn strip falls')
            print('  in front of the camera.')
        else:
            lo = min(r[2] for r in rows)
            hi = max(r[3] for r in rows)
            pad = 0.10
            print('  SUGGESTED CONSTANTS (padded by 0.10 rad):')
            print(f'    SCAN_AZIM_REL_START = {lo - pad:.3f}')
            print(f'    SCAN_AZIM_REL_END   = {hi + pad:.3f}')
            print(f'  spawns visible: {len(rows)}/{nx * ny}')
            elevs = [r[5] for r in rows]
            print(f'  elevation used spans [{min(elevs):.3f}, {max(elevs):.3f}] rad -- '
                  f'the table-aimed solve must reach this.')
            if len(rows) < nx * ny:
                print('  Some spawns are invisible at every azimuth. Widening will not')
                print('  help those -- move SCAN_VANTAGE.')
    return rows


def scan_coverage_check(setup, nx=5, ny=5, verbose=True):
    """Does the CONFIGURED sweep find each spawn, and how far in?

    Runs the actual schedule -- azimuth from the clock, elevation solved for the
    table -- so it measures what a collection run would experience. Any spawn
    marked MISSED will time out, be discarded by collect_demos.py, and bias the
    kept dataset toward the spawns the sweep happens to cover.
    """
    model = setup["model"]
    data = setup["data"]
    wrist_cam_id, _ = _wrist_cam_frame(setup)

    results = {}
    for x in np.linspace(box_x_range[0], box_x_range[1], nx):
        for y in np.linspace(box_y_range[0], box_y_range[1], ny):
            frozen = _reach_vantage_and_freeze(setup, x, y)
            found_at = None
            if frozen is not None:
                hold, dof_a, dof_e = frozen
                azim0 = float(hold[dof_a])
                hint = float(hold[dof_e])
                box_pos = data.joint('middle_box_joint').qpos[:3].copy()
                dwell = 0
                for s in range(SCAN_STEPS + 1):
                    q = hold.copy()
                    q[dof_a] = scan_azimuth(s, azim0)
                    hint = solve_elevation_for_table(
                        model, data, wrist_cam_id, q, dof_e, hint)
                    q[dof_e] = hint
                    data.qpos[:len(q)] = q
                    mujoco.mj_forward(model, data)
                    dwell = dwell + 1 if box_is_found(
                        model, data, wrist_cam_id, box_pos) else 0
                    if dwell >= SCAN_DWELL_STEPS:
                        found_at = (s, scan_azimuth(s, azim0) - azim0, hint)
                        break
            results[(round(float(x), 3), round(float(y), 3))] = found_at

    if verbose:
        n_found = sum(v is not None for v in results.values())
        print(f'SWEEP COVERAGE: {n_found}/{len(results)} spawns detected')
        print(f'  azim {SCAN_AZIM_JOINT}, offsets '
              f'[{SCAN_AZIM_REL_START:.2f}, {SCAN_AZIM_REL_END:.2f}] over {SCAN_STEPS} steps')
        print(f'  elev {SCAN_ELEV_JOINT}, solved for the table at z={SCAN_TABLE_Z}')
        print(f'  cone {np.rad2deg(SCAN_DETECT_CONE):.0f} deg, dwell {SCAN_DWELL_STEPS}\n')
        print(f'  {"x":>8}{"y":>8}{"step":>8}{"azim_off":>11}{"elev":>9}')
        for (x, y), v in sorted(results.items()):
            if v is None:
                print(f'  {x:>8.3f}{y:>8.3f}{"MISSED":>8}')
            else:
                s, off, e = v
                print(f'  {x:>8.3f}{y:>8.3f}{s:>8d}{off:>11.3f}{e:>9.3f}')

        steps = [v[0] for v in results.values() if v is not None]
        if steps:
            print(f'\n  detection step: min {min(steps)}, median '
                  f'{int(np.median(steps))}, max {max(steps)} of {SCAN_STEPS}')
            if min(steps) < SCAN_DWELL_STEPS + 5:
                print('  WARNING: some spawns are detected almost immediately -- they were')
                print('  already in frame before any searching happened. Start the sweep')
                print('  further from the spawn region (more negative SCAN_AZIM_REL_START).')
        if n_found < len(results):
            print('\n  MISSED spawns will time out and be discarded. Run --probe: if they')
            print('  are visible at some azimuth, widen SCAN_AZIM_REL_*; if not, the')
            print('  vantage is wrong.')
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Scan diagnostics for the sweeping expert.')
    ap.add_argument('--probe', action='store_true',
                    help='per spawn, the azimuth offsets that can see it (elevation '
                         'aimed at the box). Run this FIRST.')
    ap.add_argument('--nx', type=int, default=None)
    ap.add_argument('--ny', type=int, default=None)
    args = ap.parse_args()

    if args.probe:
        scan_probe_grid(setup, nx=args.nx or 3, ny=args.ny or 3)
    else:
        scan_coverage_check(setup, nx=args.nx or 5, ny=args.ny or 5)