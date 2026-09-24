"""
COLA rollout evaluation on the THREE-ARM ALOHA handover task.

The chain is A -> B -> C: A starts holding the box, hands it to B, B turns 180
degrees and hands it on to C. That is TWO handovers, so unlike the 2-arm eval
there are two transfers to score and a failure can happen at either one.

Scored signals per episode:

  transfer_ab : B alone holds the lifted box and A has actually opened. Absence
                of contact is not enough -- a finger pad grazing the box while A
                is still closed reads as "not holding".
  transfer_bc : same test one link down the chain, C holding and B opened.
  success     : both transfers, and (criterion 'hold') C keeps the box for
                SETTLE_STEPS afterwards.

--criterion:
    transfer  score at transfer_bc. success_rate == bc_rate.
    hold      (default) C must keep the box SETTLE_STEPS after transfer_bc.
              Catches the episode where C takes the box and drops it.
    ab_only   score at transfer_ab and stop. Isolates the first link so it can
              be compared against the 2-arm handover-only number (88%).

Both transfer rates are always reported, so a broken chain is attributable to a
link without re-running. The drop test keys on CONTACT, not height: an arm may
legitimately carry the box low, and box_z < LIFT_Z there is not a drop.

Ported from handover/cola/cola_eval_aloha.py (the script that measured the 88%
2-arm result). The old 3arm-handover/cola/cola_eval_3arm.py is NOT the ancestor
-- it targets a deleted ThreeArmHandoverEnv with a tray-placement task.

Usage:
    python3 cola_eval_3arm_handover.py \
        --model /home/users/ntu/ahaskar0/CoLA/experiments/run_3arm_honly_unet/checkpoints/best_model.pkl \
        --episodes 50
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola_architecture_3arm import COLAModel3Arm, CHUNK_SIZE, ARMS  # noqa: E402

SCENE_XML = '/home/users/ntu/ahaskar0/CoLA/environments/handover_3arm/scene.xml'
CACHE_DIR = ('/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/'
             'cache_3arm_honly_v1')
# The expert's IK rig, used only to build the handover-only start state.
EXPERT_DIR = '/home/users/ntu/ahaskar0/CoLA/data_collection/handover_3arm'

# Demos stored every 10th sim step (collect_demos.py record_every_n_steps=10).
CONTROL_DECIMATION = 10

# Handover-only start offsets. These MUST match scripted_policy.run_episode's
# handover_only branch (y +-0.10, z +-0.06); evaluating on a different spread
# measures distribution shift rather than policy quality. x is held fixed: it
# is the axis separating the arms, so the rendezvous point stays put and the
# variation is in where A presents the box within its own workspace.
HO_Y_RANGE = 0.10
HO_Z_RANGE = 0.06

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Arm key -> XML body prefix. The third arm was duplicated from the LEFT block,
# so its joints are third/* but its structure mirrors A's.
PREFIX = {'a': 'left', 'b': 'right', 'c': 'third'}
# Each arm sees only its own wrist camera -- the partial observability the
# message channel exists to bridge.
WRIST_CAM = {'a': 'wrist_cam_left', 'b': 'wrist_cam_right', 'c': 'wrist_cam_third'}

# Rollout video camera. teleoperator_pov is framed on the 2-arm pair and cuts
# off arm C at x=1.65, so the default here is the top-down view that contains
# all three bases.
VIDEO_CAMERA = 'overhead_cam'

# Gripper actuator endpoints, matching utils.py in the collection code.
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002

# --------------------------------------------------------------------------
# Criterion constants, carried over from the 2-arm eval.
# --------------------------------------------------------------------------
# Box counts as lifted well clear of the table (it rests at z ~ 0.03).
LIFT_Z = 0.10
# Consecutive control steps a transfer condition must hold, so a one-frame
# contact glitch cannot score.
HOLD_STEPS = 5
# --criterion hold: control steps C must keep the box after transfer_bc.
# NOTE: the 2-arm file warns its demos leave only ~25-35 steps after the
# transfer, making a larger window unpassable by the demonstrations
# themselves. The B->C transfer sits at the END of a 13-phase chain, so the
# margin here may be thinner still. Verify with --expert-replay before
# trusting a failure at this threshold.
SETTLE_STEPS = 20
# Consecutive steps with no contact that count as a drop. Contact flickers, so
# one missing frame is not a drop.
DROP_STEPS = 3
# An arm's gripper counts as open past this fraction of the way to GRIPPER_OPEN.
GRIPPER_OPEN_FRAC = 0.6

CRITERIA = ('transfer', 'hold', 'ab_only')


def build_scene(xml_path: str) -> Dict:
    """Load the scene and cache every id the rollout needs.

    Deliberately does not import the collection utils' setup_dual_arm_ik: that
    pulls in mink for the scripted policy's IK, which a learned policy has no
    use for (it emits joint targets directly).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def arm(prefix):
        return {
            'actuators': np.array([model.actuator(f'{prefix}/{n}').id for n in ARM_JOINTS]),
            'gripper_actuator': model.actuator(f'{prefix}/gripper').id,
            'subtree': model.body(f'{prefix}/base_link').id,
            # Same layout collect_demos.py recorded as state_a/state_b/state_c:
            # six arm joint positions, then the left finger position.
            'qadr': np.array([model.joint(f'{prefix}/{n}').qposadr[0] for n in ARM_JOINTS]),
            'finger_qadr': model.joint(f'{prefix}/left_finger').qposadr[0],
            # Both finger pads. check_gripper_box_contact_right in the 2-arm
            # utils listed one pad twice and missed the other; list both here.
            'finger_geoms': {
                model.geom(f'{prefix}/{side}_g{i}').id
                for side in ('left', 'right') for i in range(3)
            },
        }

    arms = {k: arm(p) for k, p in PREFIX.items()}
    neutral_key = model.key('neutral_pose').id

    mujoco.mj_resetDataKeyframe(model, data, neutral_key)
    home_qpos = {k: data.qpos[arms[k]['qadr']].copy() for k in ARMS}

    return {
        'model': model,
        'data': data,
        'arms': arms,
        'box_qposadr': model.joint('middle_box_joint').qposadr[0],
        'box_geom': model.geom('middle_box_geom').id,
        'neutral_key': neutral_key,
        'home_qpos': home_qpos,
    }


def load_action_stats(cache_dir: str):
    """Return a callable mapping normalised outputs back to joint radians."""
    with open(Path(cache_dir) / 'action_stats.json') as f:
        stats = json.load(f)

    margin = stats['margin']
    bounds = {}
    for a in ARMS:
        key = f'action_{a}'
        lo = np.array(stats[key]['min'], dtype=np.float32)
        hi = np.array(stats[key]['max'], dtype=np.float32)
        bounds[a] = (lo, hi - lo)

    def denormalise(actions, a):
        lo, span = bounds[a]
        return (actions + margin) / (2 * margin) * span + lo

    return denormalise


def load_state_normaliser(cache_dir: str):
    """Return a callable that z-scores a raw joint state, or None if absent.

    Must use the stats prepare_h5_3arm.py fitted on the TRAIN split; normalising
    a rollout with anything else silently shifts the input distribution.
    """
    path = Path(cache_dir) / 'state_stats.json'
    if not path.exists():
        return None
    with open(path) as f:
        stats = json.load(f)
    # NOTE the asymmetry: prepare_h5_3arm.py keys state_stats.json by the bare
    # arm ('a'), but action_stats.json by 'action_a'. The 2-arm cache used
    # 'state_a' for both, so this is not a copy of that convention.
    bounds = {a: (np.array(stats[a]['mean'], dtype=np.float32),
                  np.array(stats[a]['std'], dtype=np.float32))
              for a in ARMS}

    def normalise(state, a):
        mean, std = bounds[a]
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
    """Lazily build the expert's IK rig, reused across episodes.

    build_scene() skips mink because a learned policy emits joint targets
    directly. The handover-only START STATE, though, is produced by IK: the box
    is held by contact, so it cannot be repositioned by writing qpos -- the
    closed fingers drag it back to equilibrium between the pads. The arm has to
    move and carry it. Importing the expert's own setup means eval and
    collect_demos.py build that state with the same code rather than two
    implementations that drift apart.
    """
    global _HO_SETUP
    if _HO_SETUP is None:
        sys.path.insert(0, EXPERT_DIR)
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
    offset = np.array([0.0,
                       rng.uniform(-HO_Y_RANGE, HO_Y_RANGE),
                       rng.uniform(-HO_Z_RANGE, HO_Z_RANGE)])
    goal = grip0 + offset

    cfg.update(d.qpos)
    su['posture_task'].set_target_from_configuration(cfg)
    d.mocap_pos[su['left_mocap_id']] = goal
    d.mocap_quat[su['left_mocap_id']] = su['GRASP_ORIENTATION'].wxyz
    su['left_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "left/target"))
    # B and C stay where the keyframe put them: park their tasks on their own
    # current pose so the solver holds them instead of driving them somewhere.
    mink.move_mocap_to_frame(m, d, "right/target", "right/gripper", "site")
    su['right_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "right/target"))
    mink.move_mocap_to_frame(m, d, "third/target", "third/gripper", "site")
    su['third_ee_task'].set_target(mink.SE3.from_mocap_name(m, d, "third/target"))

    gc = su['GRIPPER_CLOSED']
    for _ in range(1500):
        vel = mink.solve_ik(cfg, su['tasks'], dt, limits=su['limits'],
                            solver=su['solver'], damping=1e-5)
        cfg.integrate_inplace(vel, dt)
        d.ctrl[su['left_actuator_ids']] = cfg.q[su['left_dof_ids']]
        d.ctrl[su['right_actuator_ids']] = cfg.q[su['right_dof_ids']]
        d.ctrl[su['third_actuator_ids']] = cfg.q[su['third_dof_ids']]
        d.ctrl[su['left_gripper_actuator_id']] = gc
        d.ctrl[su['right_gripper_actuator_id']] = gc
        d.ctrl[su['third_gripper_actuator_id']] = gc
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
    video_dir: str = 'evaluation_videos_3arm',
    results_path: str = 'logs/cola_eval_3arm_results.json',
    max_control_steps: int = 600,
    use_messages: bool = True,
    video_camera: str = VIDEO_CAMERA,
    criterion: str = 'hold',
    seed_offset: int = 0,
) -> Dict:
    assert criterion in CRITERIA, f'criterion must be one of {CRITERIA}'

    crit_desc = {
        'transfer': 'both transfers (A->B then B->C)',
        'hold': f'both transfers + C keeps the box {SETTLE_STEPS} steps',
        'ab_only': 'A->B transfer only (comparable to the 2-arm number)',
    }[criterion]

    print('=' * 60)
    print('COLA 3-ARM HANDOVER EVALUATION')
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

    # Build the model the way the checkpoint was trained. Getting a flag wrong
    # fails silently rather than loudly: the wrong proprio setting is a
    # parameter-shape error, but a tanh-vs-logit gripper mismatch just produces
    # a policy that never opens its hand.
    cfg = ckpt.get('config', {})
    use_proprio = bool(ckpt.get('use_proprio', cfg.get('use_proprio', False)))
    use_overhead = bool(ckpt.get('use_overhead', cfg.get('use_overhead', False)))
    split_gripper = bool(ckpt.get('split_gripper', True))
    use_diffusion = bool(cfg.get('use_diffusion', False))
    diffusion_unet = bool(cfg.get('diffusion_unet', False))

    with open(Path(cache_dir) / 'action_stats.json') as _f:
        velocity_actions = bool(json.load(_f).get('velocity', False))
    if velocity_actions:
        print('   action space: VELOCITY (deltas integrated onto the current pose)')

    cola = COLAModel3Arm(use_proprio=use_proprio, split_gripper=split_gripper,
                         use_overhead=use_overhead, use_diffusion=use_diffusion,
                         diffusion_unet=diffusion_unet)
    cola.params = ckpt['params']
    print(f"   loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.6f})")
    print(f"   overhead cam: {'on' if use_overhead else 'off'} | "
          f"proprioception: {'on' if use_proprio else 'off'} | "
          f"gripper: {'logit' if split_gripper else 'tanh'} | "
          f"head: {'diffusion+unet' if diffusion_unet else ('diffusion' if use_diffusion else 'mlp')}")

    # A checkpoint trained with the channel severed is meaningless to evaluate
    # with messages on. Older checkpoints predate the flag, so default True.
    trained_with_messages = bool(ckpt.get('use_messages', True))
    if use_messages and not trained_with_messages:
        raise SystemExit(
            'This checkpoint was TRAINED with the message channel severed. '
            'Evaluating it with messages on feeds the head a signal it never '
            'saw. Pass --no-messages.'
        )

    denormalise = load_action_stats(cache_dir)
    normalise_state = load_state_normaliser(cache_dir)
    if use_proprio and normalise_state is None:
        raise SystemExit(
            f'Checkpoint expects proprioception but {cache_dir}/state_stats.json '
            f'is missing.'
        )

    print('\n2. Building 3-arm ALOHA scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['arms'][k]['subtree'] for k in ARMS]
    print('   scene ready')

    results = {
        'model': str(model_path),
        'use_messages': use_messages,
        'criterion': {
            'name': criterion,
            'lift_z': LIFT_Z,
            'hold_steps': HOLD_STEPS,
            'settle_steps': SETTLE_STEPS,
            'drop_steps': DROP_STEPS,
            'gripper_open_frac': GRIPPER_OPEN_FRAC,
            'max_control_steps': max_control_steps,
        },
        'episodes': [],
        'task_successes': 0,
        'ab_count': 0,
        'bc_count': 0,
        'drop_count': 0,
        'b_touched_count': 0,
        'c_touched_count': 0,
        'skipped': 0,
    }

    print(f'\n3. Running {n_episodes} episodes...')
    for ep_idx in tqdm(range(n_episodes), desc='Evaluating'):
        # A start state the grasp did not survive is not a policy failure; skip
        # rather than score an episode that began with an empty hand.
        if not reset_episode_handover(scene, seed_offset + ep_idx, scene_xml):
            print(f'  ep {ep_idx}: skipped (grasp lost building start state)')
            results['skipped'] += 1
            continue

        frames = []
        b_touched = False
        c_touched = False
        ab_run = 0              # consecutive steps the A->B condition holds
        bc_run = 0              # consecutive steps the B->C condition holds
        ab_done = False
        bc_done = False
        ab_step = None
        bc_step = None
        settle_run = 0          # steps C has kept the box since transfer_bc
        no_contact_run = 0      # consecutive steps the holder is not touching
        dropped = False
        drop_after = None       # which link the drop followed: 'ab' or 'bc'
        success = False
        control_step = 0

        # Cheap diagnostics -- make a failure attributable without a re-run.
        max_box_z = 0.0
        max_ab_run = 0
        max_bc_run = 0

        while control_step < max_control_steps and not success and not dropped:
            images = {}
            for a in ARMS:
                renderer.update_scene(data, camera=WRIST_CAM[a])
                images[a] = renderer.render()[np.newaxis, ...]

            # One fixed third-person view shared by all arms -- the same
            # image_overhead stream the features were extracted from.
            image_o = None
            if use_overhead:
                renderer.update_scene(data, camera='overhead_cam')
                image_o = renderer.render()[np.newaxis, ...]

            proprio = None
            if use_proprio:
                proprio = {a: normalise_state(read_state(data, scene['arms'][a]), a)
                           for a in ARMS}

            chunks = cola.forward(images, use_messages=use_messages,
                                  proprio=proprio, image_o=image_o)
            chunks = {a: np.array(chunks[a][0]) for a in ARMS}   # (CHUNK, 7)

            for a in ARMS:
                ch = chunks[a]
                if split_gripper:
                    # Column 6 is a logit, not a normalised position: threshold
                    # it and command the actuator endpoints. Running a logit of
                    # e.g. 3.0 through denormalise() lands far outside ctrl range.
                    grip = np.where(ch[:, 6] > 0.0, GRIPPER_OPEN, GRIPPER_CLOSED)
                    ch = denormalise(ch, a)
                    ch[:, 6] = grip
                else:
                    ch = denormalise(ch, a)
                chunks[a] = ch

            for k in range(CHUNK_SIZE):
                if control_step >= max_control_steps:
                    break

                for a in ARMS:
                    step = chunks[a][k].copy()
                    if velocity_actions:
                        # A velocity cache stores per-step joint DELTAS. Written
                        # straight to ctrl they command "go to 0.03 rad from the
                        # origin" instead of "move 0.03 from here". Read qpos
                        # fresh each step rather than accumulating onto the
                        # previous command, which would let the target drift
                        # away from the arm whenever the servo lags.
                        step[:6] = data.qpos[scene['arms'][a]['qadr']] + step[:6]
                    apply_action(data, scene['arms'][a], step)

                for _ in range(CONTROL_DECIMATION):
                    compensate_gravity(model, data, subtrees)
                    mujoco.mj_step(model, data)

                control_step += 1

                box_z = float(data.qpos[scene['box_qposadr'] + 2])
                holds = {a: touching_box(data, scene['box_geom'],
                                         scene['arms'][a]['finger_geoms'])
                         for a in ARMS}
                is_open = {a: (float(data.qpos[scene['arms'][a]['finger_qadr']])
                               > GRIPPER_OPEN * GRIPPER_OPEN_FRAC)
                           for a in ARMS}

                b_touched |= holds['b']
                c_touched |= holds['c']
                max_box_z = max(max_box_z, box_z)

                if not ab_done:
                    # ---- Link 1: A -> B ----------------------------------
                    if holds['b'] and box_z > LIFT_Z and not holds['a'] and is_open['a']:
                        ab_run += 1
                    else:
                        ab_run = 0
                    max_ab_run = max(max_ab_run, ab_run)

                    if ab_run >= HOLD_STEPS:
                        ab_done = True
                        ab_step = control_step
                        if criterion == 'ab_only':
                            success = True

                elif not bc_done:
                    # ---- Link 2: B -> C ----------------------------------
                    # B must keep the box until C takes it. Key the drop test on
                    # CONTACT, not height: B turns 180 degrees carrying the box
                    # and may dip below LIFT_Z on the way round.
                    if not holds['b'] and not holds['c']:
                        no_contact_run += 1
                        if no_contact_run >= DROP_STEPS:
                            dropped = True
                            drop_after = 'ab'
                            break
                    else:
                        no_contact_run = 0

                    if holds['c'] and box_z > LIFT_Z and not holds['b'] and is_open['b']:
                        bc_run += 1
                    else:
                        bc_run = 0
                    max_bc_run = max(max_bc_run, bc_run)

                    if bc_run >= HOLD_STEPS:
                        bc_done = True
                        bc_step = control_step
                        if criterion == 'transfer':
                            success = True
                else:
                    # ---- Phase 3: C keeps it -----------------------------
                    no_contact_run = 0 if holds['c'] else no_contact_run + 1
                    if no_contact_run >= DROP_STEPS:
                        dropped = True
                        drop_after = 'bc'
                        break

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
                label = f'FAIL_DROPPED_AFTER_{drop_after.upper()}'
            elif bc_done:
                label = 'FAIL_NOSETTLE'
            elif ab_done:
                label = 'FAIL_NO_BC'
            elif b_touched:
                label = 'FAIL_BTOUCH_NO_AB'
            else:
                label = 'FAIL_NO_AB'
            imageio.mimsave(str(videos_dir / f'episode_{ep_idx:03d}_{label}.mp4'),
                            frames, fps=20)

        results['episodes'].append({
            'episode': ep_idx,
            'control_steps': control_step,
            'success': bool(success),
            'ab_done': bool(ab_done),
            'bc_done': bool(bc_done),
            'ab_step': ab_step,
            'bc_step': bc_step,
            'dropped': bool(dropped),
            'drop_after': drop_after,
            'settle_run': settle_run,
            'b_touched': bool(b_touched),
            'c_touched': bool(c_touched),
            'final_box_z': float(data.qpos[scene['box_qposadr'] + 2]),
            'max_box_z': max_box_z,
            'max_ab_run': max_ab_run,
            'max_bc_run': max_bc_run,
        })
        results['task_successes'] += int(success)
        results['ab_count'] += int(ab_done)
        results['bc_count'] += int(bc_done)
        results['drop_count'] += int(dropped)
        results['b_touched_count'] += int(b_touched)
        results['c_touched_count'] += int(c_touched)

    renderer.close()

    scored = len(results['episodes'])
    if scored == 0:
        raise SystemExit('No episodes were scored -- every start state lost the grasp.')

    ab_eps = [e for e in results['episodes'] if e['ab_step'] is not None]
    bc_eps = [e for e in results['episodes'] if e['bc_step'] is not None]
    results['summary'] = {
        'n_episodes': n_episodes,
        'n_scored': scored,
        'n_skipped': results['skipped'],
        'success_rate': 100.0 * results['task_successes'] / scored,
        'ab_rate': 100.0 * results['ab_count'] / scored,
        'bc_rate': 100.0 * results['bc_count'] / scored,
        # Of the episodes that cleared A->B, how many went on to clear B->C.
        # This is the number that says whether the SECOND link is the bottleneck.
        'bc_given_ab': (100.0 * results['bc_count'] / len(ab_eps)) if ab_eps else None,
        'drop_rate': 100.0 * results['drop_count'] / scored,
        'b_touch_rate': 100.0 * results['b_touched_count'] / scored,
        'c_touch_rate': 100.0 * results['c_touched_count'] / scored,
        'mean_control_steps': float(np.mean([e['control_steps'] for e in results['episodes']])),
        'mean_ab_step': float(np.mean([e['ab_step'] for e in ab_eps])) if ab_eps else None,
        'mean_bc_step': float(np.mean([e['bc_step'] for e in bc_eps])) if bc_eps else None,
    }

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    s = results['summary']
    print(f"Success:            {results['task_successes']}/{scored} = "
          f"{s['success_rate']:.1f}%   ({crit_desc})")
    print(f"  A->B transfer:    {s['ab_rate']:.1f}%   (compare: 2-arm handover-only 88.0%)")
    print(f"  B->C transfer:    {s['bc_rate']:.1f}%"
          + (f"   ({s['bc_given_ab']:.1f}% of episodes that cleared A->B)"
             if s['bc_given_ab'] is not None else ''))
    print(f"  Dropped:          {s['drop_rate']:.1f}%")
    print(f"  B touched box:    {s['b_touch_rate']:.1f}%   (partial credit)")
    print(f"  C touched box:    {s['c_touch_rate']:.1f}%   (partial credit)")
    if results['skipped']:
        print(f"  Skipped starts:   {results['skipped']} (grasp lost at reset, not scored)")
    print(f"Mean control steps: {s['mean_control_steps']:.1f}")
    # If either of these sits near max_control_steps there is no room left for
    # the next phase -- raise --max-control-steps rather than reading the
    # success rate as a policy failure.
    if s['mean_ab_step'] is not None:
        print(f"Mean A->B step:     {s['mean_ab_step']:.1f} / {max_control_steps}")
    if s['mean_bc_step'] is not None:
        print(f"Mean B->C step:     {s['mean_bc_step']:.1f} / {max_control_steps}")

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
                        default='/home/users/ntu/ahaskar0/CoLA/experiments/'
                                'run_3arm_honly_unet/checkpoints/best_model.pkl')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--scene-xml', type=str, default=SCENE_XML)
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='shift the per-episode seeds. Seeds are the episode '
                             'index, so a nonzero offset gives a disjoint set of '
                             'start states from a previous run.')
    parser.add_argument('--video-dir', type=str, default='evaluation_videos_3arm')
    parser.add_argument('--results-path', type=str,
                        default='logs/cola_eval_3arm_results.json')
    parser.add_argument('--max-control-steps', type=int, default=600,
                        help='the demos run ~233 recorded steps over 13 phases, '
                             'so 350 (the 2-arm default) truncates the chain.')
    parser.add_argument('--no-messages', action='store_true',
                        help='sever the message channel: the no-coordination baseline.')
    parser.add_argument('--criterion', choices=CRITERIA, default='hold')
    parser.add_argument('--camera', type=str, default=VIDEO_CAMERA,
                        help='rollout video camera. teleoperator_pov cuts off '
                             'arm C at x=1.65; overhead_cam shows all three.')
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
        criterion=args.criterion,
        seed_offset=args.seed_offset,
    )
