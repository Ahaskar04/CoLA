"""Evaluate a three-arm CoLA checkpoint on the A -> B -> C handover.

A starts holding the box and hands it to B; B turns 180 degrees and hands it
to C. Scored per episode:
  transfer_ab  B alone holds the lifted box and A's gripper is open.
  transfer_bc  the same for C taking the box from B.
  success      both transfers; with --criterion hold (default), C also keeps
               the box for SETTLE_STEPS.

--criterion transfer scores at transfer_bc; ab_only scores the first link only.
Drops are detected by loss of contact, not by height.

Usage:
    python eval_3arm.py --model RUN/checkpoints/best_model.pkl --episodes 50
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cola.model_3arm import COLAModel3Arm, CHUNK_SIZE, ARMS         # noqa: E402

SCENE_XML = str(REPO / 'envs' / 'handover_3arm' / 'scene.xml')
CACHE_DIR = str(REPO / 'data' / 'cache' / 'handover_3arm')
# The expert's IK rig, used only to build the handover-only start state.
EXPERT_DIR = str(REPO / 'data_collection' / 'handover_3arm')
RESULTS_DIR = REPO / 'results' / 'eval_3arm'

# Demos were recorded every 10th sim step.
CONTROL_DECIMATION = 10

# Handover-only start offsets; must match scripted_policy.py's handover_only
# branch. x stays fixed (the axis between the arms).
HO_Y_RANGE = 0.10
HO_Z_RANGE = 0.06

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Arm key -> XML body prefix (the third arm is a copy of the left one).
PREFIX = {'a': 'left', 'b': 'right', 'c': 'third'}
# Each arm sees only its own wrist camera.
WRIST_CAM = {'a': 'wrist_cam_left', 'b': 'wrist_cam_right', 'c': 'wrist_cam_third'}

# Rollout video camera: the top-down view shows all three arms.
VIDEO_CAMERA = 'overhead_cam'

# Gripper actuator endpoints, matching utils.py in the collection code.
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002

# Success-criterion constants (as in the two-arm eval)
# Box counts as lifted well clear of the table (it rests at z ~ 0.03).
LIFT_Z = 0.10
# Consecutive control steps a transfer condition must hold.
HOLD_STEPS = 5
# --criterion hold: control steps C must keep the box after transfer_bc.
SETTLE_STEPS = 20
# Consecutive no-contact steps that count as a drop (contact flickers).
DROP_STEPS = 3
# An arm's gripper counts as open past this fraction of the way to GRIPPER_OPEN.
GRIPPER_OPEN_FRAC = 0.6

CRITERIA = ('transfer', 'hold', 'ab_only')


def build_scene(xml_path: str) -> Dict:
    """Load the scene and cache every id the rollout needs (no IK)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def arm(prefix):
        return {
            'actuators': np.array([model.actuator(f'{prefix}/{n}').id for n in ARM_JOINTS]),
            'gripper_actuator': model.actuator(f'{prefix}/gripper').id,
            'subtree': model.body(f'{prefix}/base_link').id,
            # Same layout as the recorded states: six joints, then the left finger.
            'qadr': np.array([model.joint(f'{prefix}/{n}').qposadr[0] for n in ARM_JOINTS]),
            'finger_qadr': model.joint(f'{prefix}/left_finger').qposadr[0],
            # Both finger pads.
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
    """Return a callable that z-scores a joint state (train-split stats), or None."""
    path = Path(cache_dir) / 'state_stats.json'
    if not path.exists():
        return None
    with open(path) as f:
        stats = json.load(f)
    # state_stats.json is keyed by arm ('a'); action_stats.json by 'action_a'.
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
    """Build the expert's IK rig once (used only to construct the start state).

    The held box can't be placed by writing qpos, so the start state is reached
    with the expert's own IK, as in collect_demos.py.
    """
    global _HO_SETUP
    if _HO_SETUP is None:
        sys.path.insert(0, EXPERT_DIR)
        from utils import setup_dual_arm_ik
        _HO_SETUP = setup_dual_arm_ik(scene_xml)
    return _HO_SETUP


def reset_episode_handover(scene, seed: int, scene_xml: str):
    """Start with arm A holding the box, as in the training demos.

    Mirrors scripted_policy.py's handover_only branch (same y/z offsets).
    Returns False if the grasp didn't survive, so the episode can be skipped.
    """
    import mink
    # utils is importable only after _handover_setup adds the expert dir to sys.path.
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
    # Hold B and C at their keyframe poses while A moves.
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

    # Copy the start state into the scene used for the rollout.
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
    video_dir: str = str(RESULTS_DIR / 'videos'),
    results_path: str = str(RESULTS_DIR / 'results.json'),
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

    # Rebuild the model with the flags it was trained with.
    cfg = ckpt.get('config', {})
    use_proprio = bool(ckpt.get('use_proprio', cfg.get('use_proprio', False)))
    use_overhead = bool(ckpt.get('use_overhead', cfg.get('use_overhead', False)))
    use_wrist = bool(ckpt.get('use_wrist', cfg.get('use_wrist', True)))
    split_gripper = bool(ckpt.get('split_gripper', True))
    use_diffusion = bool(cfg.get('use_diffusion', False))
    diffusion_unet = bool(cfg.get('diffusion_unet', False))

    with open(Path(cache_dir) / 'action_stats.json') as _f:
        velocity_actions = bool(json.load(_f).get('velocity', False))
    if velocity_actions:
        print('   action space: VELOCITY (deltas integrated onto the current pose)')

    cola = COLAModel3Arm(use_proprio=use_proprio, split_gripper=split_gripper,
                         use_overhead=use_overhead, use_wrist=use_wrist,
                         use_diffusion=use_diffusion, diffusion_unet=diffusion_unet)
    cola.params = ckpt['params']
    print(f"   loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.6f})")
    print(f"   cameras: {'+'.join(c for c, on in (('wrist', use_wrist), ('overhead', use_overhead)) if on)} | "
          f"proprioception: {'on' if use_proprio else 'off'} | "
          f"gripper: {'logit' if split_gripper else 'tanh'} | "
          f"head: {'diffusion+unet' if diffusion_unet else ('diffusion' if use_diffusion else 'mlp')}")

    # A checkpoint trained without messages must be evaluated without them.
    # Older checkpoints predate the flag, so default to True.
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
        # Skip episodes whose start-state grasp failed.
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

        # Extra diagnostics, to tell failure modes apart.
        max_box_z = 0.0
        max_ab_run = 0
        max_bc_run = 0

        while control_step < max_control_steps and not success and not dropped:
            images = {}
            for a in ARMS:
                renderer.update_scene(data, camera=WRIST_CAM[a])
                images[a] = renderer.render()[np.newaxis, ...]

            # Shared overhead view, as in the extracted features.
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
                    # Column 6 is a logit: threshold it to the gripper's open/closed targets.
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
                        # Velocity actions are deltas on the current joint positions.
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
                    # Link 1: A -> B
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
                    # Link 2: B -> C
                    # Drops are detected by contact, since B can dip below LIFT_Z
                    # while turning.
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
                    # Phase 3: C keeps it
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
        # Fraction of A->B successes that also cleared B->C.
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
    # If these are close to max_control_steps, episodes are timing out; try a
    # larger --max-control-steps.
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
                        default=str(REPO / 'runs' / 'handover_3arm' / 'checkpoints' / 'best_model.pkl'))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--scene-xml', type=str, default=SCENE_XML)
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='shift the per-episode seeds. Seeds are the episode '
                             'index, so a nonzero offset gives a disjoint set of '
                             'start states from a previous run.')
    parser.add_argument('--video-dir', type=str, default=str(RESULTS_DIR / 'videos'))
    parser.add_argument('--results-path', type=str, default=str(RESULTS_DIR / 'results.json'))
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
