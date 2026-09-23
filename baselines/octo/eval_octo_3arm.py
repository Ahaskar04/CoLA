"""Octo baseline on the 3-arm A->B->C handover, through CoLA's own harness.

Generated from /home/users/ntu/ahaskar0/CoLA/training/handover_3arm/cola_eval_3arm_handover.py
by replacing only the two model-specific blocks -- loading the policy and the
per-chunk forward pass. The handover-only reset, the A->B / B->C / hold
criterion, drop detection, video labels and summary are CoLA's code unchanged,
so an Octo number and a CoLA number are scored identically. check_criterion()
re-verifies that at start-up and refuses to run if CoLA's evaluator has since
changed.

Three decentralised policies, one per arm, each seeing the overhead camera and
its own proprioception, with no channel between them.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

COLA_EVAL_DIR = '/home/users/ntu/ahaskar0/CoLA/training/handover_3arm'
sys.path.insert(0, COLA_EVAL_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cola_eval_3arm_handover as H                       # noqa: E402
from cola_eval_3arm_handover import (                     # noqa: E402
    ARMS, CHUNK_SIZE, CONTROL_DECIMATION, CRITERIA, DROP_STEPS, GRIPPER_OPEN,
    GRIPPER_OPEN_FRAC, HOLD_STEPS, LIFT_Z, SCENE_XML, CACHE_DIR, SETTLE_STEPS,
    VIDEO_CAMERA, apply_action, build_scene, compensate_gravity,
    reset_episode_handover, touching_box)
import eval_octo as E                                     # noqa: E402
from eval_octo_finetuned import FinetunedOctoPolicy       # noqa: E402

COLA_SOURCE = '/home/users/ntu/ahaskar0/CoLA/training/handover_3arm/cola_eval_3arm_handover.py'


# Markers are assembled rather than written out: spelled literally they would
# appear in this function's own source and the search would start here.
_START = ' ' * 12 + 'for k in range(' + 'CHUNK_SIZE):'
_END = ' ' * 4 + 'renderer.' + 'close()\n\n' + ' ' * 4 + "scored = len(results['episodes'])"


def _criterion_block(text):
    i = text.index(_START)
    j = text.index(_END, i)
    return text[i:j]


def check_criterion():
    """Refuse to run if the scoring code no longer matches CoLA's."""
    mine = _criterion_block(open(__file__).read())
    theirs = _criterion_block(open(COLA_SOURCE).read())
    if mine != theirs:
        raise SystemExit(
            'The criterion block differs from ' + COLA_SOURCE + '. CoLA\'s '
            'evaluator has changed since this file was generated; regenerate '
            'it so Octo and CoLA are scored by the same rules.')
    print('   criterion: identical to CoLA\'s evaluator')


def evaluate_octo_3arm(
    checkpoints: Dict[str, str],
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
    seed: int = 0,
) -> Dict:
    assert criterion in CRITERIA, f'criterion must be one of {CRITERIA}'

    crit_desc = {
        'transfer': 'both transfers (A->B then B->C)',
        'hold': f'both transfers + C keeps the box {SETTLE_STEPS} steps',
        'ab_only': 'A->B transfer only (comparable to the 2-arm number)',
    }[criterion]

    print('=' * 60)
    print('OCTO 3-ARM HANDOVER EVALUATION (decentralised)')
    print('=' * 60)
    for _a in ARMS:
        print(f'Arm {_a}: {checkpoints[_a]}')
    print(f'Episodes: {n_episodes}')
    print(f'Messages: {"on" if use_messages else "OFF (no-coordination baseline)"}')
    print(f'Success: {crit_desc}')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading Octo policies (one per arm, no channel)...')
    # Same convention CoLA reads: absolute joint targets unless the cache says
    # velocity. Octo learned whatever the cache holds, so this stays in sync.
    with open(Path(cache_dir) / 'action_stats.json') as _f:
        velocity_actions = bool(json.load(_f).get('velocity', False))
    if velocity_actions:
        print('   action space: VELOCITY (deltas integrated onto the current pose)')
    policies = {a: FinetunedOctoPolicy(checkpoints[a], seed=seed + i)
                for i, a in enumerate(ARMS)}
    assert E.CHUNK_SIZE == CHUNK_SIZE, (
        f'Octo executes {E.CHUNK_SIZE} steps per query but the CoLA harness '
        f'chunks {CHUNK_SIZE}; the control cadence would differ.')

    print('\n2. Building 3-arm ALOHA scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['arms'][k]['subtree'] for k in ARMS]
    print('   scene ready')

    results = {
        'model': 'octo-small-1.5 finetuned, decentralised',
        'checkpoints': {a: str(checkpoints[a]) for a in ARMS},
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
            # Three independent Octo policies. Each renders the shared
            # overhead view and reads only its own proprio; nothing passes
            # between them. act() returns CHUNK_SIZE absolute joint targets
            # plus the gripper command in actuator units -- exactly what
            # apply_action writes to ctrl, so no denormalise or logit step.
            chunks = {a: policies[a].act(scene, renderer, scene['arms'][a])
                      for a in ARMS}

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
    p = argparse.ArgumentParser(description=__doc__)
    for a in ARMS:
        p.add_argument(f'--checkpoint-{a}', required=True)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--scene-xml', default=SCENE_XML)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video-dir', default='evaluation_videos_octo_3arm')
    p.add_argument('--results-path', required=True)
    p.add_argument('--max-control-steps', type=int, default=600)
    p.add_argument('--criterion', choices=CRITERIA, default='hold')
    p.add_argument('--camera', default=VIDEO_CAMERA)
    args = p.parse_args()

    check_criterion()
    evaluate_octo_3arm(
        checkpoints={a: getattr(args, f'checkpoint_{a}') for a in ARMS},
        n_episodes=args.episodes, scene_xml=args.scene_xml,
        cache_dir=args.cache_dir, save_videos=not args.no_videos,
        video_dir=args.video_dir, results_path=args.results_path,
        max_control_steps=args.max_control_steps, use_messages=False,
        video_camera=args.camera, criterion=args.criterion,
        seed_offset=args.seed_offset, seed=args.seed)
