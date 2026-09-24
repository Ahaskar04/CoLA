"""
COLA rollout evaluation on the ALOHA handover task.

Success is scored in two phases, with three selectable end conditions:

  1. TRANSFER (always) — B alone holds the lifted box and A has actually opened
     its gripper. Absence of contact alone is not enough: a finger pad grazing
     the box while A is still closed reads as "not holding".

  2. End condition, --criterion:
       transfer  score at the transfer. The criterion used in the CoRL
                 submission; transfer_rate == success_rate.
       hold      (default) B must keep the box for SETTLE_STEPS after the
                 transfer. Catches the episode where B takes the box and
                 immediately drops it, which `transfer` counts as a success.
       return    B must additionally bring its arm back within HOME_TOL_RAD of
                 the neutral pose while still holding.

transfer_rate is reported under every criterion, so numbers stay comparable
with earlier runs and with the paper.

The drop test keys on CONTACT, not height: B may legitimately carry the box low
on its way back, and box_z < LIFT_Z there is not a drop.

IMPORTANT: expert_replay.py mirrors this file's control loop and criterion. If
you change one, change the other, or the replay stops being the same test.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

from cola_architecture import COLAModel, CHUNK_SIZE

SCENE_XML = '/home/users/ntu/ahaskar0/CoLA/environments/handover_2arm/scene.xml'
CACHE_DIR = '/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_aloha_handover'

# Demos stored every 10th sim step (collect_demos.py record_every_n_steps=10).
CONTROL_DECIMATION = 10

# Box spawn ranges used by scripted_policy.py — NOT the wider ones in env.py.
BOX_X_RANGE = (-0.12, 0.12)
BOX_Y_RANGE = (-0.10, 0.10)
BOX_SPAWN_HEIGHT = 0.03
# Handover-only start offsets, matching scripted_policy.run_episode's
# handover_only branch. Evaluating on a different spread than training saw
# would measure distribution shift rather than policy quality.
HO_Y_RANGE = 0.10
HO_Z_RANGE = 0.06

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Camera used for the saved rollout videos. A side-on view shows lift height and
# the A->B transfer, which the top-down overhead_cam flattens away.
# Options: teleoperator_pov (side, front), collaborator_pov (side, rear),
#          overhead_cam (top-down), worms_eye_cam (low angle).
VIDEO_CAMERA = 'teleoperator_pov'

# Gripper actuator endpoints, matching utils.py in the collection code. Used
# when the checkpoint emits a gripper logit rather than a position.
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002

# --------------------------------------------------------------------------
# Criterion constants. expert_replay.py carries identical values.
# --------------------------------------------------------------------------
# Box counts as lifted well clear of the table (it rests at z ~ 0.03).
LIFT_Z = 0.10
# Consecutive control steps the transfer condition must hold, so a one-frame
# contact glitch cannot score.
HOLD_STEPS = 5
# --criterion hold: control steps B must keep the box after the transfer.
# The demos leave only ~25-35 steps after the transfer completes, so a window
# much larger than this is unpassable by the demonstrations themselves. Verify
# with expert_replay.py --success-criterion hold before raising it.
SETTLE_STEPS = 20
# Consecutive steps with no B-box contact that count as a drop. Contact
# flickers, so one missing frame is not a drop.
DROP_STEPS = 3
# --criterion return: the returning arm is home when every one of its joints is
# within this many radians of the neutral keyframe.
HOME_TOL_RAD = 0.10
# Consecutive steps it must hold that pose while still holding the box.
HOME_HOLD_STEPS = 5
# A's gripper counts as open past this fraction of the way to GRIPPER_OPEN.
GRIPPER_OPEN_FRAC = 0.6

CRITERIA = ('transfer', 'hold', 'return')


def build_scene(xml_path: str) -> Dict:
    """Load the scene and cache every id the rollout needs.

    Deliberately does not import CoLA's utils.setup_dual_arm_ik: that pulls in
    mink/loop_rate_limiters for the scripted policy's IK, which a learned
    policy has no use for (it emits joint targets directly) and which are not
    installed in this venv.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def arm(prefix):
        return {
            'actuators': np.array([model.actuator(f'{prefix}/{n}').id for n in ARM_JOINTS]),
            'gripper_actuator': model.actuator(f'{prefix}/gripper').id,
            'subtree': model.body(f'{prefix}/base_link').id,
            # Same layout collect_demos.py recorded as state_a/state_b:
            # six arm joint positions, then the left finger position.
            'qadr': np.array([model.joint(f'{prefix}/{n}').qposadr[0] for n in ARM_JOINTS]),
            'finger_qadr': model.joint(f'{prefix}/left_finger').qposadr[0],
            # Both finger pads, left and right. CoLA's
            # check_gripper_box_contact_right lists right/right_g* twice and
            # omits right/left_g*, so it misses contacts on one pad.
            'finger_geoms': {
                model.geom(f'{prefix}/{side}_g{i}').id
                for side in ('left', 'right') for i in range(3)
            },
        }

    a, b = arm('left'), arm('right')
    neutral_key = model.key('neutral_pose').id

    # Home configurations, read from the same keyframe reset_episode restores at
    # the start of every episode. Both are kept so --return-arm can pick either.
    mujoco.mj_resetDataKeyframe(model, data, neutral_key)
    home_qpos_a = data.qpos[a['qadr']].copy()
    home_qpos_b = data.qpos[b['qadr']].copy()

    return {
        'model': model,
        'data': data,
        'a': a,
        'b': b,
        'box_qposadr': model.joint('middle_box_joint').qposadr[0],
        'box_geom': model.geom('middle_box_geom').id,
        'neutral_key': neutral_key,
        'home_qpos_a': home_qpos_a,
        'home_qpos_b': home_qpos_b,
    }


def load_action_stats(cache_dir: str):
    """Return a callable mapping tanh outputs back to joint radians."""
    with open(Path(cache_dir) / 'action_stats.json') as f:
        stats = json.load(f)

    margin = stats['margin']
    bounds = {}
    for arm in ('action_a', 'action_b'):
        lo = np.array(stats[arm]['min'], dtype=np.float32)
        hi = np.array(stats[arm]['max'], dtype=np.float32)
        bounds[arm] = (lo, hi - lo)

    def denormalise(actions, arm):
        lo, span = bounds[arm]
        return (actions + margin) / (2 * margin) * span + lo

    return denormalise


def load_state_normaliser(cache_dir: str):
    """Return a callable that z-scores a raw joint state, or None if unavailable.

    Must use the same stats prepare_states_h5.py fitted on the train split;
    normalising a rollout with anything else silently shifts the input
    distribution the policy was trained on.
    """
    path = Path(cache_dir) / 'state_stats.json'
    if not path.exists():
        return None
    with open(path) as f:
        stats = json.load(f)
    bounds = {arm: (np.array(stats[arm]['mean'], dtype=np.float32),
                    np.array(stats[arm]['std'], dtype=np.float32))
              for arm in ('state_a', 'state_b')}

    def normalise(state, arm):
        mean, std = bounds[arm]
        return ((state - mean) / std).astype(np.float32)[None, :]

    return normalise


def read_state(data, arm):
    """Live proprioception in the layout the demos recorded."""
    return np.concatenate([
        data.qpos[arm['qadr']],
        [data.qpos[arm['finger_qadr']]],
    ]).astype(np.float32)


def compensate_gravity(model, data, subtree_ids):
    """Cancel gravity on the arm subtrees, as collect_demos.py did."""
    qfrc_applied = data.qfrc_applied
    qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


def touching_box(data, box_geom: int, finger_geoms: set) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == box_geom and c.geom2 in finger_geoms) or \
           (c.geom2 == box_geom and c.geom1 in finger_geoms):
            return True
    return False


_HO_SETUP = None


def _handover_setup(scene_xml: str):
    """Lazily build the scripted expert's IK rig, reused across episodes.

    build_scene() deliberately skips mink because a learned policy emits joint
    targets directly. The handover-only START STATE, though, is produced by IK:
    the box is held by contact, so it cannot be repositioned by writing qpos --
    the closed fingers drag it back to equilibrium between the pads. The arm has
    to move and carry it. Importing the expert's own setup here means eval and
    collect_demos.py construct that state with the same code rather than two
    implementations that can drift apart.
    """
    global _HO_SETUP
    if _HO_SETUP is None:
        import sys
        sys.path.insert(0, '/home/users/ntu/ahaskar0/CoLA/data_collection/handover_2arm')
        from utils import setup_dual_arm_ik
        _HO_SETUP = setup_dual_arm_ik(scene_xml)
    return _HO_SETUP


def reset_episode_handover(scene, seed: int, scene_xml: str):
    """Start with arm A already holding the box, as the training demos do.

    Mirrors the handover_only branch of scripted_policy.run_episode(): load the
    handover_start keyframe, offset A's gripper by the SAME y/z ranges the
    dataset was collected with, settle, and verify the grasp survived. Returns
    False if it did not, so the caller can skip rather than score an episode
    that began with an empty gripper.
    """
    import mink
    # _handover_setup puts the expert's directory on sys.path, so `utils` is
    # only importable after it has run.
    su = _handover_setup(scene_xml)
    from utils import check_gripper_box_contact
    m, d = su['model'], su['data']
    cfg = su['configuration']
    dt = 1.0 / 200.0
    rng = np.random.default_rng(seed)

    mujoco.mj_resetDataKeyframe(m, d, m.key('handover_start').id)
    mujoco.mj_forward(m, d)
    grip0 = d.site('left/gripper').xpos.copy()
    # Same ranges as scripted_policy.run_episode: x fixed (the arms always meet
    # at the same point along the axis between them), y/z randomised.
    offset = np.array([0.0,
                       rng.uniform(-HO_Y_RANGE, HO_Y_RANGE),
                       rng.uniform(-HO_Z_RANGE, HO_Z_RANGE)])
    goal = grip0 + offset

    cfg.update(d.qpos)
    su['posture_task'].set_target_from_configuration(cfg)
    d.mocap_pos[su['left_mocap_id']] = goal
    d.mocap_quat[su['left_mocap_id']] = su['GRASP_ORIENTATION'].wxyz
    su['left_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "left/target"))
    mink.move_mocap_to_frame(m, d, "right/target", "right/gripper", "site")
    su['right_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "right/target"))

    gc = su['GRIPPER_CLOSED']
    for _ in range(1500):
        vel = mink.solve_ik(cfg, su['tasks'], dt, limits=su['limits'],
                            solver=su['solver'], damping=1e-5)
        cfg.integrate_inplace(vel, dt)
        d.ctrl[su['left_actuator_ids']] = cfg.q[su['left_dof_ids']]
        d.ctrl[su['right_actuator_ids']] = cfg.q[su['right_dof_ids']]
        d.ctrl[su['left_gripper_actuator_id']] = gc
        d.ctrl[su['right_gripper_actuator_id']] = gc
        mujoco.mj_step(m, d)
        if np.linalg.norm(d.site('left/gripper').xpos - goal) < 0.005:
            break

    if not check_gripper_box_contact(m, d):
        return False

    # Copy the constructed state into the scene the policy actually runs in.
    scene['data'].qpos[:] = d.qpos
    scene['data'].qvel[:] = 0.0
    scene['data'].ctrl[:] = d.ctrl
    mujoco.mj_forward(scene['model'], scene['data'])
    return True


def reset_episode(scene, seed: int):
    model, data = scene['model'], scene['data']
    rng = np.random.default_rng(seed)

    mujoco.mj_resetDataKeyframe(model, data, scene['neutral_key'])
    addr = scene['box_qposadr']
    data.qpos[addr:addr + 3] = [
        rng.uniform(*BOX_X_RANGE),
        rng.uniform(*BOX_Y_RANGE),
        BOX_SPAWN_HEIGHT,
    ]
    mujoco.mj_forward(model, data)


def apply_action(data, arm, action):
    """Six joint position targets plus one gripper command."""
    data.ctrl[arm['actuators']] = action[:6]
    data.ctrl[arm['gripper_actuator']] = action[6]


def evaluate_cola(
    model_path: str,
    n_episodes: int = 50,
    scene_xml: str = SCENE_XML,
    cache_dir: str = CACHE_DIR,
    save_videos: bool = True,
    video_dir: str = 'evaluation_videos',
    results_path: str = 'logs/cola_eval_aloha_results.json',
    max_control_steps: int = 350,
    use_messages: bool = True,
    video_camera: str = VIDEO_CAMERA,
    criterion: str = 'hold',
    return_arm: str = 'b',
    handover_only: bool = False,
    seed_offset: int = 0,
) -> Dict:
    assert criterion in CRITERIA, f'criterion must be one of {CRITERIA}'
    assert return_arm in ('a', 'b')

    crit_desc = {
        'transfer': 'transfer only (legacy)',
        'hold': f'transfer + B keeps the box {SETTLE_STEPS} steps',
        'return': f'transfer + arm {return_arm.upper()} returns home, still holding',
    }[criterion]

    print('=' * 60)
    print('COLA ALOHA HANDOVER EVALUATION')
    print('=' * 60)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}')
    print(f'Messages: {"on" if use_messages else "OFF (no-coordination baseline)"}')
    print(f'Success: {crit_desc}')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading COLA model...')
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    # Build the model the way the checkpoint was trained. Getting either flag
    # wrong fails silently rather than loudly: the wrong proprio setting is a
    # parameter-shape error, but a tanh-vs-logit mismatch on the gripper just
    # produces a policy that never opens its hand.
    use_proprio = bool(ckpt.get('use_proprio', False))
    # Must match how the checkpoint was trained: overhead adds a second
    # FEATURE_DIM block to self_dim (768 -> 1536), so rebuilding without it
    # produces params of the wrong shape and the weights will not load.
    use_overhead = bool(ckpt.get('use_overhead', False))
    split_gripper = bool(ckpt.get('split_gripper', False))
    # Diffusion checkpoints carry a denoiser whose params have a different shape
    # AND a different call signature, so the flag has to come from the file
    # rather than a CLI switch -- building the wrong head fails to load.
    use_diffusion = bool(ckpt.get('config', {}).get('use_diffusion', False))
    # Set by prepare_h5_handover.py --velocity. Getting this wrong is silent:
    # deltas written as absolute targets send the arm to a pose near the
    # origin on step 1, and absolute targets integrated as deltas run away.
    with open(Path(cache_dir) / 'action_stats.json') as _f:
        velocity_actions = bool(json.load(_f).get('velocity', False))
    if velocity_actions:
        print('   action space: VELOCITY (deltas integrated onto the current pose)')
    diffusion_unet = bool(ckpt.get('config', {}).get('diffusion_unet', False))
    # Capacity ablation: a checkpoint trained at (64,128) rebuilt into the
    # default (128,256) head fails to load with a shape error. Absent on
    # pre-ablation checkpoints, where None restores the (128,256) default.
    unet_dims = ckpt.get('config', {}).get('unet_dims', None)
    cola = COLAModel(use_proprio=use_proprio, split_gripper=split_gripper,
                     use_overhead=use_overhead, use_diffusion=use_diffusion,
                     diffusion_unet=diffusion_unet, unet_dims=unet_dims)
    if diffusion_unet:
        print(f"   unet down_dims: {tuple(unet_dims) if unet_dims else '(128, 256) [default]'}")
    cola.params = ckpt['params']
    print(f"   loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.6f})")
    print(f"   overhead cam: {'on' if use_overhead else 'off'}")
    print(f"   proprioception: {'on' if use_proprio else 'off'} | "
          f"gripper: {'logit' if split_gripper else 'tanh'}")

    # A checkpoint trained with the channel severed is meaningless to evaluate
    # with messages on. Older checkpoints predate the flag, so default to True.
    trained_with_messages = bool(ckpt.get('use_messages', True))
    if use_messages and not trained_with_messages:
        raise SystemExit(
            'This checkpoint was TRAINED with the message channel severed. '
            'Evaluating it with messages on feeds the head a signal it never '
            'saw. Pass --no-messages.'
        )

    denormalise = load_action_stats(cache_dir)
    print(f'   action stats from {cache_dir}/action_stats.json')

    normalise_state = load_state_normaliser(cache_dir)
    if use_proprio and normalise_state is None:
        raise SystemExit(
            f'Checkpoint expects proprioception but {cache_dir}/state_stats.json '
            f'is missing. Run prepare_states_h5.py.'
        )

    print('\n2. Building ALOHA scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['a']['subtree'], scene['b']['subtree']]
    ret_arm = scene[return_arm]
    home_qpos = scene[f'home_qpos_{return_arm}']
    print('   scene ready')

    results = {
        'model': str(model_path),
        'use_messages': use_messages,
        'criterion': {
            'name': criterion,
            'return_arm': return_arm,
            'lift_z': LIFT_Z,
            'hold_steps': HOLD_STEPS,
            'settle_steps': SETTLE_STEPS,
            'drop_steps': DROP_STEPS,
            'home_tol_rad': HOME_TOL_RAD,
            'home_hold_steps': HOME_HOLD_STEPS,
            'gripper_open_frac': GRIPPER_OPEN_FRAC,
            'max_control_steps': max_control_steps,
        },
        'episodes': [],
        'task_successes': 0,
        'transfer_count': 0,
        'drop_count': 0,
        'a_lifted_count': 0,
        'b_touched_count': 0,
    }

    print(f'\n3. Running {n_episodes} episodes...')
    for ep_idx in tqdm(range(n_episodes), desc='Evaluating'):
        if handover_only:
            # A start state the grasp did not survive is not a policy failure;
            # skip rather than score an episode that began with an empty hand.
            if not reset_episode_handover(scene, seed_offset + ep_idx, scene_xml):
                print(f'  ep {ep_idx}: skipped (grasp lost building start state)')
                continue
        else:
            reset_episode(scene, seed=seed_offset + ep_idx)

        frames = []
        a_lifted = False        # A picked the box up off the table
        b_touched = False       # B made contact with the box at all
        handover_run = 0        # consecutive steps the transfer condition holds
        transfer_done = False   # phase 1 complete (the legacy success criterion)
        transfer_step = None    # control step at which phase 1 completed
        settle_run = 0          # steps B has kept the box since the transfer
        no_contact_run = 0      # consecutive steps B is not touching the box
        home_run = 0            # steps the returning arm has been home, holding
        dropped = False         # B lost the box after taking it
        success = False
        control_step = 0

        # Cheap diagnostics — make a failure attributable without a re-run.
        max_box_z = 0.0
        max_handover_run = 0
        best_home_err = float('inf')   # smallest post-transfer home error seen

        while control_step < max_control_steps and not success and not dropped:
            # Each arm sees only its own wrist camera — the partial
            # observability the message channel is meant to bridge.
            renderer.update_scene(data, camera='wrist_cam_left')
            image_a = renderer.render()[np.newaxis, ...]
            renderer.update_scene(data, camera='wrist_cam_right')
            image_b = renderer.render()[np.newaxis, ...]

            # One fixed third-person view, shared by both arms -- the same
            # image_overhead stream the features were extracted from.
            image_o = None
            if use_overhead:
                renderer.update_scene(data, camera='overhead_cam')
                image_o = renderer.render()[np.newaxis, ...]

            proprio_a = proprio_b = None
            if use_proprio:
                proprio_a = normalise_state(read_state(data, scene['a']), 'state_a')
                proprio_b = normalise_state(read_state(data, scene['b']), 'state_b')

            chunk_a, chunk_b = cola.forward(
                image_a, image_b, use_messages=use_messages,
                proprio_a=proprio_a, proprio_b=proprio_b,
                image_o=image_o,
            )
            chunk_a = np.array(chunk_a[0])   # (CHUNK_SIZE, 7)
            chunk_b = np.array(chunk_b[0])

            if split_gripper:
                # Column 6 is a logit, not a normalised position: threshold it
                # and command the actuator's open/closed endpoints. Running it
                # through denormalise() would map a logit of, say, 3.0 far
                # outside the ctrl range.
                grip_a = np.where(chunk_a[:, 6] > 0.0, GRIPPER_OPEN, GRIPPER_CLOSED)
                grip_b = np.where(chunk_b[:, 6] > 0.0, GRIPPER_OPEN, GRIPPER_CLOSED)
                chunk_a = denormalise(chunk_a, 'action_a')
                chunk_b = denormalise(chunk_b, 'action_b')
                chunk_a[:, 6] = grip_a
                chunk_b[:, 6] = grip_b
            else:
                chunk_a = denormalise(chunk_a, 'action_a')
                chunk_b = denormalise(chunk_b, 'action_b')

            for k in range(CHUNK_SIZE):
                if control_step >= max_control_steps:
                    break

                if velocity_actions:
                    # A velocity cache stores per-step joint DELTAS. Written
                    # straight to ctrl they would command "go to 0.03 rad from
                    # the origin" instead of "move 0.03 from here", so integrate
                    # onto the arm's CURRENT pose. Read qpos fresh each step:
                    # accumulating onto the previous command instead would let
                    # the target drift away from where the arm actually is
                    # whenever the position servo lags.
                    step_a = chunk_a[k].copy()
                    step_b = chunk_b[k].copy()
                    step_a[:6] = data.qpos[scene['a']['qadr']] + step_a[:6]
                    step_b[:6] = data.qpos[scene['b']['qadr']] + step_b[:6]
                    # Column 6 is the gripper and was never differenced.
                    apply_action(data, scene['a'], step_a)
                    apply_action(data, scene['b'], step_b)
                else:
                    apply_action(data, scene['a'], chunk_a[k])
                    apply_action(data, scene['b'], chunk_b[k])

                for _ in range(CONTROL_DECIMATION):
                    compensate_gravity(model, data, subtrees)
                    mujoco.mj_step(model, data)

                control_step += 1

                box_z = float(data.qpos[scene['box_qposadr'] + 2])
                a_holds = touching_box(data, scene['box_geom'], scene['a']['finger_geoms'])
                b_holds = touching_box(data, scene['box_geom'], scene['b']['finger_geoms'])

                a_lifted |= a_holds and box_z > LIFT_Z
                b_touched |= b_holds
                max_box_z = max(max_box_z, box_z)

                # A has genuinely let go, not merely lost contact for a frame.
                a_open = (float(data.qpos[scene['a']['finger_qadr']])
                          > GRIPPER_OPEN * GRIPPER_OPEN_FRAC)
                b_has_box = b_holds and box_z > LIFT_Z

                if not transfer_done:
                    # ---- Phase 1: transfer -------------------------------
                    if b_has_box and not a_holds and a_open:
                        handover_run += 1
                    else:
                        handover_run = 0
                    max_handover_run = max(max_handover_run, handover_run)

                    if handover_run >= HOLD_STEPS:
                        transfer_done = True
                        transfer_step = control_step
                        if criterion == 'transfer':
                            success = True
                else:
                    # ---- Phase 2: keep it (and optionally go home) --------
                    # Drop test keys on CONTACT, not height: B may carry the box
                    # low on the way back, and box_z < LIFT_Z there is not a drop.
                    no_contact_run = 0 if b_holds else no_contact_run + 1
                    if no_contact_run >= DROP_STEPS:
                        dropped = True
                        break

                    if criterion == 'return':
                        home_err = float(np.max(np.abs(
                            data.qpos[ret_arm['qadr']] - home_qpos)))
                        best_home_err = min(best_home_err, home_err)
                        home_run = home_run + 1 if home_err < HOME_TOL_RAD else 0
                        if home_run >= HOME_HOLD_STEPS:
                            success = True
                    else:   # 'hold'
                        settle_run += 1
                        if settle_run >= SETTLE_STEPS:
                            success = True

                if save_videos:
                    renderer.update_scene(data, camera=video_camera)
                    frames.append(renderer.render())

                if success or dropped:
                    break

        if save_videos and frames:
            if success:
                label = 'SUCCESS'
            elif dropped:
                label = 'FAIL_DROPPED'
            elif transfer_done:
                label = 'FAIL_NORETURN'
            elif b_touched:
                label = 'FAIL_BTOUCH_NOXFER'
            elif a_lifted:
                label = 'FAIL_NOHANDOVER'
            else:
                label = 'FAIL_NOLIFT'
            imageio.mimsave(str(videos_dir / f'episode_{ep_idx:03d}_{label}.mp4'), frames, fps=20)

        results['episodes'].append({
            'episode': ep_idx,
            'control_steps': control_step,
            'success': bool(success),
            'transfer_done': bool(transfer_done),
            'transfer_step': transfer_step,
            'dropped': bool(dropped),
            'settle_run': settle_run,
            'a_lifted': bool(a_lifted),
            'b_touched': bool(b_touched),
            'final_box_z': float(data.qpos[scene['box_qposadr'] + 2]),
            'max_box_z': max_box_z,
            'max_handover_run': max_handover_run,
            'best_home_err': None if best_home_err == float('inf') else best_home_err,
        })
        results['task_successes'] += int(success)
        results['transfer_count'] += int(transfer_done)
        results['drop_count'] += int(dropped)
        results['a_lifted_count'] += int(a_lifted)
        results['b_touched_count'] += int(b_touched)

    renderer.close()

    transferred = [e for e in results['episodes'] if e['transfer_step'] is not None]
    results['summary'] = {
        'n_episodes': n_episodes,
        'success_rate': 100.0 * results['task_successes'] / n_episodes,
        # Phase 1 only == the criterion used before the return phase existed.
        'transfer_rate': 100.0 * results['transfer_count'] / n_episodes,
        'drop_rate': 100.0 * results['drop_count'] / n_episodes,
        # Of the episodes that transferred, how many then lost the box.
        'drop_given_transfer': (100.0 * results['drop_count'] / len(transferred)
                                if transferred else None),
        'a_lift_rate': 100.0 * results['a_lifted_count'] / n_episodes,
        'b_touch_rate': 100.0 * results['b_touched_count'] / n_episodes,
        'mean_control_steps': float(np.mean([e['control_steps'] for e in results['episodes']])),
        'mean_transfer_step': (float(np.mean([e['transfer_step'] for e in transferred]))
                               if transferred else None),
    }

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    s = results['summary']
    print(f"Success:           {results['task_successes']}/{n_episodes} = "
          f"{s['success_rate']:.1f}%   ({crit_desc})")
    print(f"  Transfer only:   {s['transfer_rate']:.1f}%   (legacy criterion)")
    print(f"  Dropped after:   {s['drop_rate']:.1f}%"
          + (f"   ({s['drop_given_transfer']:.1f}% of transfers)"
             if s['drop_given_transfer'] is not None else ''))
    print(f"  A lifted the box:  {s['a_lift_rate']:.1f}%   (partial credit)")
    print(f"  B touched the box: {s['b_touch_rate']:.1f}%   (partial credit)")
    print(f"Mean control steps: {s['mean_control_steps']:.1f}")
    if s['mean_transfer_step'] is not None:
        # If this sits near max_control_steps there is no room left for phase 2
        # — raise --max-control-steps rather than reading the success rate as a
        # policy failure.
        print(f"Mean transfer step: {s['mean_transfer_step']:.1f} / {max_control_steps}")

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out_path}')
    if save_videos:
        print(f'Videos saved to: {videos_dir}/')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str,
                        default='experiments/run_aloha_handover_01/checkpoints/best_model.pkl')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--scene-xml', type=str, default=SCENE_XML)
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='shift the per-episode seeds by this amount. Seeds are '
                             'deterministic, so the default 0 always evaluates the '
                             'same box positions -- useful for matched comparisons '
                             'between models, but it means a single run samples one '
                             'fixed set. Offsetting draws a different set.')
    parser.add_argument('--handover-only', action='store_true',
                        help='start each episode with arm A already holding the box, '
                             'as the handover-only demos do, instead of spawning it '
                             'on the table. Required for checkpoints trained on '
                             'aloha-handover-only-*: evaluating one of those against '
                             'a table spawn measures distribution mismatch, not skill.')
    parser.add_argument('--video-dir', type=str,
                        default='experiments/run_aloha_handover_01/evaluation_videos')
    parser.add_argument('--results-path', type=str,
                        default='experiments/run_aloha_handover_01/logs/cola_eval_aloha_results.json')
    # Raised from 200: the episode no longer stops at the transfer, so phase 2
    # needs room. Watch mean_transfer_step to size this.
    parser.add_argument('--max-control-steps', type=int, default=350)
    parser.add_argument('--no-messages', action='store_true',
                        help='zero the message channel (L0 no-coordination baseline)')
    parser.add_argument('--criterion', choices=CRITERIA, default='hold',
                        help="transfer: score at the handover (the CoRL criterion). "
                             "hold: B must keep the box afterwards (default). "
                             "return: B must also bring its arm home while holding.")
    parser.add_argument('--return-arm', choices=('a', 'b'), default='b',
                        help='which arm must return home under --criterion return')
    parser.add_argument('--camera', type=str, default=VIDEO_CAMERA,
                        help='camera for saved videos: teleoperator_pov (side), '
                             'collaborator_pov (side, rear), overhead_cam (top-down), '
                             'worms_eye_cam')
    args = parser.parse_args()

    evaluate_cola(
        model_path=args.model,
        n_episodes=args.episodes,
        scene_xml=args.scene_xml,
        cache_dir=args.cache_dir,
        save_videos=not args.no_videos,
        video_dir=args.video_dir,
        results_path=args.results_path,
        max_control_steps=args.max_control_steps,
        use_messages=not args.no_messages,
        video_camera=args.camera,
        handover_only=args.handover_only,
        seed_offset=args.seed_offset,
        criterion=args.criterion,
        return_arm=args.return_arm,
    )