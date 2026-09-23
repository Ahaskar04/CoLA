"""
COLA rollout evaluation on the HIDDEN-MARKER handover task.

The box carries a coloured marker on its negative-x face: visible to arm A's
wrist camera, physically occluded from B. After the handover, B must drop the
box in the matching tray. B never sees the marker, so the colour can only reach
it through A -- via the learned message channel, or via A's behaviour.

WHY THE SCORING IS DIFFERENT FROM cola_eval_aloha.py. That script scores "B
holds the box after the handover", which here would measure the handover and
ignore the entire point of the task. Success here is CORRECT TRAY of three, so
chance is 33.3%, not ~50%. The three-way split matters:

    correct_tray : box settled in the tray matching the marker
    wrong_tray   : box settled in one of the other two trays
    no_tray      : box never reached any tray (handover failed, or dropped short)

A policy that transfers no colour information can still reach a tray every time
-- it just picks at random, giving ~33%. So `tray_rate` (reached ANY tray) and
`correct_given_tray` separate "can it place?" from "does it know where?". Only
the second is about communication.

EVERY EPISODE STARTS FROM AN IDENTICAL POSE. scripted_policy.py sets the
handover-only reset offset to exactly zeros (line ~425), so unlike the plain
handover task there is no y/z randomisation. The marker colour is the ONLY
thing that varies between episodes. That is what makes this a clean test: there
is no geometric cue B could exploit instead of the colour.

Usage:
    python3 cola_eval_marker.py \
        --model .../checkpoints/best_model.pkl \
        --episodes 150
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

_COLA = '/scratch/users/ntu/ahaskar0/v1/cola-research-code/handover/cola'
if _COLA not in sys.path:
    sys.path.insert(0, _COLA)
from cola_architecture import COLAModel, CHUNK_SIZE   # noqa: E402

SCENE_XML = ('/home/users/ntu/ahaskar0/CoLA/environments/handover_2arm_marker/'
             'scene.xml')
CACHE_DIR = ('/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/'
             'cache_marker_v1')
EXPERT_DIR = '/home/users/ntu/ahaskar0/CoLA/data_collection/handover_2arm_marker'

# Demos stored every 10th sim step (collect_demos.py record_every_n_steps=10).
CONTROL_DECIMATION = 10

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle',
              'wrist_rotate']

# Must match scripted_policy.py: MARKER_COLORS order sets which colour each
# episode index gets, and TRAY_POSITIONS sets where each tray sits.
MARKER_COLORS = ["blue", "green", "yellow"]
TRAY_POSITIONS = {
    "blue":   np.array([0.36, -0.247, 0.20]),
    "green":  np.array([0.36,  0.0,   0.20]),
    "yellow": np.array([0.36,  0.247, 0.20]),
}
# A box counts as "in" a tray when it settles within this radius of the tray
# centre in the x-y plane. The trays are 0.247 apart in y, so 0.12 cannot
# claim two trays at once; the expert landed 0.010 from centre.
TRAY_RADIUS = 0.12
# ...and below this height, i.e. resting in the tray rather than carried over
# it. The expert's box settled at z=0.069 after the drop.
TRAY_Z_MAX = 0.15

GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002
GRIPPER_OPEN_FRAC = 0.6

# Box counts as lifted clear of the table.
LIFT_Z = 0.10
# Consecutive control steps the transfer condition must hold.
HOLD_STEPS = 5
# Consecutive steps the box must sit in a tray before it counts as placed.
SETTLE_STEPS = 5


def build_scene(xml_path: str) -> Dict:
    """Load the scene and cache every id the rollout needs."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def arm(prefix):
        return {
            'actuators': np.array([model.actuator(f'{prefix}/{n}').id
                                   for n in ARM_JOINTS]),
            'gripper_actuator': model.actuator(f'{prefix}/gripper').id,
            'subtree': model.body(f'{prefix}/base_link').id,
            'qadr': np.array([model.joint(f'{prefix}/{n}').qposadr[0]
                              for n in ARM_JOINTS]),
            'finger_qadr': model.joint(f'{prefix}/left_finger').qposadr[0],
            'finger_geoms': {
                model.geom(f'{prefix}/{side}_g{i}').id
                for side in ('left', 'right') for i in range(3)
            },
        }

    return {
        'model': model,
        'data': data,
        'a': arm('left'),
        'b': arm('right'),
        'box_qposadr': model.joint('middle_box_joint').qposadr[0],
        'box_geom': model.geom('middle_box_geom').id,
        'marker_geoms': {c: model.geom(f'marker_{c}').id for c in MARKER_COLORS},
    }


def set_marker(scene, color: str):
    """Make one marker visible and the other two invisible.

    Mirrors scripted_policy.py: alpha 1.0 on the chosen colour, 0.0 on the
    rest. This writes model.geom_rgba, which persists across resets, so it must
    be set every episode rather than once.
    """
    model = scene['model']
    for c, gid in scene['marker_geoms'].items():
        model.geom_rgba[gid, 3] = 1.0 if c == color else 0.0
    mujoco.mj_forward(model, scene['data'])


def load_action_stats(cache_dir: str):
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
    return np.concatenate([
        data.qpos[arm['qadr']],
        [data.qpos[arm['finger_qadr']]],
    ]).astype(np.float32)


def compensate_gravity(model, data, subtree_ids):
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


def which_tray(box_pos) -> str:
    """Return the tray the box is resting in, or None.

    x-y distance to the tray centre plus a height ceiling: a box being CARRIED
    over a tray at z=0.20 is not in it, only one that has settled.
    """
    if box_pos[2] > TRAY_Z_MAX:
        return None
    for color, centre in TRAY_POSITIONS.items():
        if np.linalg.norm(box_pos[:2] - centre[:2]) < TRAY_RADIUS:
            return color
    return None


_HO_SETUP = None


def _handover_setup(scene_xml: str):
    global _HO_SETUP
    if _HO_SETUP is None:
        sys.path.insert(0, EXPERT_DIR)
        from utils import setup_dual_arm_ik
        _HO_SETUP = setup_dual_arm_ik(scene_xml)
    return _HO_SETUP


def reset_episode(scene, scene_xml: str, color: str):
    """Start with arm A holding the box, marker set to `color`.

    The collection's handover-only branch uses a ZERO offset -- every episode
    starts from the handover_start keyframe unchanged -- so unlike the plain
    handover eval there is no IK settling step to mirror and no grasp to
    re-verify. Reproducing that exactly matters: adding randomisation the
    training data never had would measure distribution shift, not policy skill.
    """
    model, data = scene['model'], scene['data']
    mujoco.mj_resetDataKeyframe(model, data, model.key('handover_start').id)
    set_marker(scene, color)
    mujoco.mj_forward(model, data)
    return True


def apply_action(data, arm, action):
    data.ctrl[arm['actuators']] = action[:6]
    data.ctrl[arm['gripper_actuator']] = action[6]


def load_swap_bank(path):
    """Donor msg_a trajectories per colour, from build_message_bank.py."""
    z = np.load(path, allow_pickle=False)
    colors = [str(c) for c in z["colors"]]
    bank = {c: [z[f"{c}_{i}"] for i in range(int(z[f"{c}_n"]))] for c in colors}
    return bank, int(z["message_dim"])


def evaluate(model_path, n_episodes=150, scene_xml=SCENE_XML,
             cache_dir=CACHE_DIR, save_videos=True,
             video_dir='evaluation_videos_marker',
             results_path='logs/cola_eval_marker_results.json',
             max_control_steps=400, use_messages=True,
             video_camera='side_cam', seed_offset=0,
             swap_bank=None, swap_seed=0):

    print('=' * 64)
    print('COLA HIDDEN-MARKER HANDOVER EVALUATION')
    print('=' * 64)
    print(f'Model: {model_path}')
    print(f'Episodes: {n_episodes}  |  chance = 33.3% (three trays)')
    print(f'Messages: {"on" if use_messages else "OFF (no-coordination baseline)"}')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading COLA model...')
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    cfg = ckpt.get('config', {})
    use_proprio = bool(ckpt.get('use_proprio', cfg.get('use_proprio', False)))
    use_overhead = bool(ckpt.get('use_overhead', cfg.get('use_overhead', False)))
    # Rebuild the channel at the width it was TRAINED at. Default 64 keeps
    # every pre-sweep checkpoint loading unchanged.
    message_dim = int(ckpt.get('message_dim', cfg.get('message_dim', 64)))
    split_gripper = bool(ckpt.get('split_gripper', True))
    use_diffusion = bool(cfg.get('use_diffusion', False))
    diffusion_unet = bool(cfg.get('diffusion_unet', False))
    unet_dims = cfg.get('unet_dims', None)

    cola = COLAModel(use_proprio=use_proprio, split_gripper=split_gripper,
                     use_overhead=use_overhead, use_diffusion=use_diffusion,
                     diffusion_unet=diffusion_unet, unet_dims=unet_dims,
                     message_dim=message_dim)
    cola.params = ckpt['params']
    print(f"   loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.6f})")
    print(f"   overhead: {'on' if use_overhead else 'off'} | "
          f"proprio: {'on' if use_proprio else 'off'} | "
          f"head: {'diffusion+unet' if diffusion_unet else 'other'}")

    # A checkpoint trained with the channel severed is meaningless to evaluate
    # with messages on: it would see a signal it never saw in training.
    trained_with_messages = bool(ckpt.get('use_messages', True))
    if use_messages and not trained_with_messages:
        raise SystemExit(
            'This checkpoint was TRAINED with messages severed. Pass '
            '--no-messages.')
    if not use_messages and trained_with_messages:
        print('   NOTE: severing messages on a checkpoint TRAINED with them. '
              'This is the out-of-distribution ablation, not the matched control.')

    # An overhead-trained checkpoint can see both arms and the box, but NOT the
    # marker -- it faces A and is occluded from any third-person view too. Say
    # so rather than let a reader assume overhead leaks the colour.
    if use_overhead:
        print('   NOTE: overhead is ON. The marker faces arm A and is not '
              'legible from the overhead camera, so the colour still has to '
              'travel through A.')

    denormalise = load_action_stats(cache_dir)
    normalise_state = load_state_normaliser(cache_dir)
    if use_proprio and normalise_state is None:
        raise SystemExit(f'Checkpoint expects proprio but {cache_dir}/'
                         f'state_stats.json is missing.')

    print('\n2. Building scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['a']['subtree'], scene['b']['subtree']]
    print('   scene ready')

    results = {
        'model': str(model_path),
        'use_messages': use_messages,
        'use_overhead': use_overhead,
        'criterion': {
            'tray_radius': TRAY_RADIUS, 'tray_z_max': TRAY_Z_MAX,
            'lift_z': LIFT_Z, 'hold_steps': HOLD_STEPS,
            'settle_steps': SETTLE_STEPS,
            'max_control_steps': max_control_steps,
            'chance_rate': 100.0 / len(MARKER_COLORS),
        },
        'episodes': [],
    }

    swap = None
    if swap_bank:
        bank, bank_dm = load_swap_bank(swap_bank)
        if bank_dm != message_dim:
            raise SystemExit(
                f'donor bank has message_dim {bank_dm} but this checkpoint '
                f'uses {message_dim}; the override would be the wrong shape')
        swap = {'bank': bank, 'rng': np.random.default_rng(swap_seed)}
        print(f'\n   MESSAGE SWAP: B receives a donor msg_a from a DIFFERENT '
              f'colour ({ {c: len(v) for c, v in bank.items()} } donors)')
        print('   Tests positive LISTENING: the probe already shows the colour '
              'is readable in msg_a (73.3% vs a 33.3% control), but readable is '
              'not used. If B follows the SWAPPED colour, the channel causally '
              'drives the tray choice.')

    print(f'\n3. Running {n_episodes} episodes...')
    for ep in tqdm(range(n_episodes), desc='Evaluating'):
        # Same cycling the collection used, so the eval is colour-balanced by
        # construction: 150 episodes is exactly 50/50/50.
        color = MARKER_COLORS[(seed_offset + ep) % len(MARKER_COLORS)]
        reset_episode(scene, scene_xml, color)

        # Draw a donor whose marker was a DIFFERENT colour. The tray B picks is
        # then the whole measurement: the true colour's tray means B is not
        # using the message, the donor's tray means it is.
        donor = donor_color = None
        if swap is not None:
            others = [c for c in MARKER_COLORS if c != color]
            donor_color = str(swap['rng'].choice(others))
            pool = swap['bank'][donor_color]
            donor = pool[int(swap['rng'].integers(len(pool)))]

        frames = []
        transfer_done = False
        transfer_run = 0
        settle_run = 0
        landed = None          # tray the box settled in
        control_step = 0

        while control_step < max_control_steps and landed is None:
            renderer.update_scene(data, camera='wrist_cam_left')
            image_a = renderer.render()[np.newaxis, ...]
            renderer.update_scene(data, camera='wrist_cam_right')
            image_b = renderer.render()[np.newaxis, ...]

            image_o = None
            if use_overhead:
                renderer.update_scene(data, camera='overhead_cam')
                image_o = renderer.render()[np.newaxis, ...]

            proprio_a = proprio_b = None
            if use_proprio:
                proprio_a = normalise_state(read_state(data, scene['a']), 'state_a')
                proprio_b = normalise_state(read_state(data, scene['b']), 'state_b')

            # Donor episodes run ~150 steps but a rollout may reach 400. Past
            # the end we HOLD THE LAST FRAME rather than zeroing: zeros are the
            # no-message condition, so a zero tail would silently turn the back
            # half of every long episode into the L0 ablation.
            msg_override = None
            if donor is not None:
                msg_override = donor[min(control_step, len(donor) - 1)][None, :]

            chunk_a, chunk_b = cola.forward(
                image_a, image_b, use_messages=use_messages,
                proprio_a=proprio_a, proprio_b=proprio_b, image_o=image_o,
                msg_a_override=msg_override)
            chunk_a = np.array(chunk_a[0])
            chunk_b = np.array(chunk_b[0])

            if split_gripper:
                # Column 6 is a logit: threshold it and command the actuator
                # endpoints. Denormalising a logit lands far outside ctrl range.
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
                apply_action(data, scene['a'], chunk_a[k])
                apply_action(data, scene['b'], chunk_b[k])
                for _ in range(CONTROL_DECIMATION):
                    compensate_gravity(model, data, subtrees)
                    mujoco.mj_step(model, data)
                control_step += 1

                box = data.qpos[scene['box_qposadr']:scene['box_qposadr'] + 3]
                a_holds = touching_box(data, scene['box_geom'],
                                       scene['a']['finger_geoms'])
                b_holds = touching_box(data, scene['box_geom'],
                                       scene['b']['finger_geoms'])
                a_open = (float(data.qpos[scene['a']['finger_qadr']])
                          > GRIPPER_OPEN * GRIPPER_OPEN_FRAC)

                if not transfer_done:
                    if b_holds and box[2] > LIFT_Z and not a_holds and a_open:
                        transfer_run += 1
                    else:
                        transfer_run = 0
                    if transfer_run >= HOLD_STEPS:
                        transfer_done = True
                else:
                    # Require the box to REST in a tray for a few steps, so a
                    # box passing through the region on its way elsewhere does
                    # not score.
                    tray = which_tray(box)
                    if tray is not None and not b_holds:
                        settle_run += 1
                        if settle_run >= SETTLE_STEPS:
                            landed = tray
                    else:
                        settle_run = 0

                if save_videos:
                    renderer.update_scene(data, camera=video_camera)
                    frames.append(renderer.render())
                if landed is not None:
                    break

        correct = (landed == color)
        if save_videos and frames:
            if correct:
                label = f'CORRECT_{color}'
            elif landed is not None:
                label = f'WRONG_{color}_to_{landed}'
            elif transfer_done:
                label = f'NOTRAY_{color}'
            else:
                label = f'NOXFER_{color}'
            imageio.mimsave(str(videos_dir / f'episode_{ep:03d}_{label}.mp4'),
                            frames, fps=20)

        box = data.qpos[scene['box_qposadr']:scene['box_qposadr'] + 3]
        results['episodes'].append({
            'episode': ep, 'marker_color': color, 'landed_tray': landed,
            'correct': bool(correct), 'transfer_done': bool(transfer_done),
            'donor_color': donor_color,
            'control_steps': control_step,
            'final_box_pos': [float(v) for v in box],
        })

    renderer.close()

    eps = results['episodes']
    n = len(eps)
    n_correct = sum(e['correct'] for e in eps)
    n_tray = sum(e['landed_tray'] is not None for e in eps)
    n_xfer = sum(e['transfer_done'] for e in eps)

    # Per-colour, because a policy that always picks one tray scores ~33%
    # overall and looks like partial information. The per-colour split exposes
    # that: a constant policy is 100/0/0, real information is balanced.
    by_color = {}
    for c in MARKER_COLORS:
        sub = [e for e in eps if e['marker_color'] == c]
        # Conditioned on transfer for the same reason as the headline: an
        # episode that never handed the box over says nothing about routing.
        sub_x = [e for e in sub if e['transfer_done']]
        by_color[c] = {
            'n': len(sub),
            'correct': sum(e['correct'] for e in sub),
            'n_transfers': len(sub_x),
            'correct_given_transfer': sum(e['correct'] for e in sub_x),
            'chose': {t: sum(1 for e in sub_x if e['landed_tray'] == t)
                      for t in MARKER_COLORS + [None]},
        }

    # Correct answers among episodes where the handover actually completed.
    # THIS IS THE HEADLINE NUMBER. A policy cannot route a box it never
    # received, so raw correct_rate conflates two unrelated abilities:
    # manipulation (can B take the box?) and routing (does B know where it
    # goes?). Only the second is about communication. The no-messages control
    # made this concrete -- it failed the physical handover in 93% of episodes,
    # so its raw 1.5% mostly measured a broken grasp, not a missing colour.
    n_correct_xfer = sum(e['correct'] for e in eps if e['transfer_done'])
    correct_given_xfer = (100.0 * n_correct_xfer / n_xfer) if n_xfer else None

    results['summary'] = {
        'n_episodes': n,
        # Report this FIRST: correct tray among completed handovers.
        'correct_given_transfer': correct_given_xfer,
        'n_transfers': n_xfer,
        'n_correct_given_transfer': n_correct_xfer,
        'correct_rate': 100.0 * n_correct / n,
        'tray_rate': 100.0 * n_tray / n,
        'transfer_rate': 100.0 * n_xfer / n,
        # Narrower still: of the times B placed the box ANYWHERE, how often was
        # it the right tray? Differs from correct_given_transfer only by the
        # episodes that transferred but never reached a tray.
        'correct_given_tray': (100.0 * n_correct / n_tray) if n_tray else None,
        'chance_rate': 100.0 / len(MARKER_COLORS),
        'by_color': by_color,
    }

    s = results['summary']
    print('\n' + '=' * 64)
    print('RESULTS')
    print('=' * 64)
    # Headline: routing skill with manipulation factored out.
    if correct_given_xfer is not None:
        print(f"CORRECT | HANDOVER: {n_correct_xfer}/{n_xfer} = "
              f"{correct_given_xfer:.1f}%   (chance {s['chance_rate']:.1f}%)")
    else:
        print("CORRECT | HANDOVER: n/a -- no episode completed the handover")
    print("  ^ the number that matters: a policy cannot route a box it never")
    print("    received, so this separates routing from manipulation.")
    print()
    print(f"  Raw correct tray: {n_correct}/{n} = {s['correct_rate']:.1f}%"
          f"   (depressed by failed handovers)")
    print(f"  Handover done:    {s['transfer_rate']:.1f}%"
          + ("   <- LOW: the raw rate above is mostly a manipulation failure, "
             "not a routing one" if s['transfer_rate'] < 50 else ""))
    print(f"  Reached a tray:   {s['tray_rate']:.1f}%")
    print(f"  Correct | tray:   "
          + (f"{s['correct_given_tray']:.1f}%" if s['correct_given_tray'] is not None else 'n/a'))
    if n_xfer and n_xfer < 30:
        print(f"  WARNING: only {n_xfer} completed handovers -- the conditional "
              f"above is too small to read.")
    print('\nPer-colour, among COMPLETED handovers '
          '(a constant policy shows up here):')
    for c in MARKER_COLORS:
        d = by_color[c]
        chose = ', '.join(f"{t or 'none'}:{v}" for t, v in d['chose'].items() if v)
        pct = (f"{100.0 * d['correct_given_transfer'] / d['n_transfers']:5.1f}%"
               if d['n_transfers'] else "  n/a")
        print(f"  marker {c:7s} xfer={d['n_transfers']:3d}  "
              f"correct={d['correct_given_transfer']:3d} ({pct})  chose {chose}")

    out = Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out}')
    if save_videos:
        print(f'Videos saved to: {videos_dir}/')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=150)
    p.add_argument('--scene-xml', default=SCENE_XML)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--video-dir', default='evaluation_videos_marker')
    p.add_argument('--results-path', default='logs/cola_eval_marker_results.json')
    p.add_argument('--max-control-steps', type=int, default=400,
                   help='expert episodes run ~152 steps; 400 leaves room for a '
                        'policy that wanders before finding the tray.')
    p.add_argument('--no-messages', action='store_true')
    p.add_argument('--camera', default='side_cam',
                   help='side_cam frames both arms AND the three trays; '
                        'overhead_cam flattens the tray row.')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--swap-messages', default=None, metavar='BANK.npz',
                   help='MESSAGE-SWAP INTERVENTION. Feed B a donor msg_a '
                        'recorded from an episode whose marker was a DIFFERENT '
                        'colour (build_message_bank.py writes the bank). The '
                        'probe shows the colour is READABLE in msg_a; this asks '
                        'whether B ACTS on it. B landing in the donor colour\'s '
                        'tray is positive listening; landing in the true '
                        'colour\'s tray means the colour reaches B by some '
                        'other route.')
    p.add_argument('--swap-seed', type=int, default=0,
                   help='seeds which donor episode each rollout draws')
    a = p.parse_args()

    evaluate(model_path=a.model, n_episodes=a.episodes, scene_xml=a.scene_xml,
             cache_dir=a.cache_dir, save_videos=not a.no_videos,
             video_dir=a.video_dir, results_path=a.results_path,
             max_control_steps=a.max_control_steps,
             use_messages=not a.no_messages, video_camera=a.camera,
             seed_offset=a.seed_offset, swap_bank=a.swap_messages,
             swap_seed=a.swap_seed)
