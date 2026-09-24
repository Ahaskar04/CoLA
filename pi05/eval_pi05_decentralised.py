"""Decentralised finetuned pi0.5 on the HIDDEN-MARKER handover, CoLA's harness.

The baseline row for CoLA-on-pi0.5: TWO independently LoRA-finetuned pi0.5
policies, one per arm, each trained on its own arm only (train_cola_pi05.py
--arm a / --arm b) and each acting from its own wrist camera and joint state.
Nothing passes between them, so arm B never learns the marker colour and
correct-tray given a completed handover should sit near chance (33.3%) unless
A's behaviour leaks it.

Generated from eval_cola_pi05_marker.py, which came from eval_octo_marker.py
and CoLA's own cola_eval_marker.py: the reset, criterion, scoring loop,
per-colour breakdown and summary are CoLA's code, verified by check_criterion().

Runs in venv-openpi with MuJoCo 3.12.0 from /scratch/users/ntu/ahaskar0/pi05_eval_site.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import imageio
import mujoco
import numpy as np
from tqdm import tqdm

COLA_EVAL_DIR = '/home/users/ntu/ahaskar0/CoLA/training/handover_2arm_marker'
sys.path.insert(0, COLA_EVAL_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# cola_eval_marker imports cola_architecture, which imports Octo (not in this
# venv). The harness only takes CHUNK_SIZE from it; stub that, verified against
# CoLA's own source so the control cadence cannot silently differ.
import re                                                  # noqa: E402
import types                                               # noqa: E402
_COLA_ARCH = '/scratch/users/ntu/ahaskar0/v1/cola-research-code/handover/cola/cola_architecture.py'
_stub = types.ModuleType('cola_architecture')
_stub.CHUNK_SIZE = int(re.search(r'^CHUNK_SIZE = (\d+)', Path(_COLA_ARCH).read_text(), re.M).group(1))
_stub.COLAModel = None
sys.modules['cola_architecture'] = _stub

from cola_eval_marker import (                             # noqa: E402
    CHUNK_SIZE, CONTROL_DECIMATION, GRIPPER_OPEN, GRIPPER_OPEN_FRAC, HOLD_STEPS,
    LIFT_Z, MARKER_COLORS, SCENE_XML, SETTLE_STEPS, TRAY_RADIUS, TRAY_Z_MAX,
    apply_action, build_scene, compensate_gravity, reset_episode, touching_box,
    which_tray)
from cola_eval_marker import CACHE_DIR, GRIPPER_CLOSED, read_state   # noqa: E402

COLA_SOURCE = '/home/users/ntu/ahaskar0/CoLA/training/handover_2arm_marker/cola_eval_marker.py'
# Markers are assembled so this file's own copies of them do not match first.
_SCORE_START = ' ' * 12 + 'for k in range(' + 'CHUNK_SIZE):'
_SCORE_END = ' ' * 4 + 'renderer.' + 'close()'
_SUM_START = ' ' * 4 + "eps = results[" + "'episodes']"
_SUM_END = ' ' * 4 + 'return ' + 'results\n'


def _block(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def check_criterion():
    """Refuse to run if the scoring or summary code no longer matches CoLA's."""
    cola = Path(COLA_SOURCE).read_text()
    mine = Path(__file__).read_text()
    for name, s, e in (('scoring loop', _SCORE_START, _SCORE_END),
                       ('summary', _SUM_START, _SUM_END)):
        if _block(cola, s, e) != _block(mine, s, e):
            raise SystemExit(f'{name} differs from {COLA_SOURCE} -- regenerate '
                             f'eval_cola_pi05_marker.py before trusting a number from it')
    print("   criterion: identical to CoLA's evaluator (scoring loop + summary)")


def merge(shards, results_path):
    """Pool sharded runs and rerun the summary over all their episodes."""
    # Keep each shard WITH its own path. Sorting the paths separately and
    # zipping them against offset-sorted data pairs lexicographic order against
    # numeric order, so results_shard100.json gets recorded as seed_offset 25
    # once there are more than two shards. The pooled episodes were always
    # right -- they come from the sorted data -- but the provenance list lied,
    # and with per-seed offsets there are far more shards to mislabel.
    parts = sorted(((p, json.load(open(p))) for p in shards),
                   key=lambda pr: pr[1]['seed_offset'])
    merged = {k: v for k, v in parts[0][1].items()
              if k not in ('episodes', 'summary', 'seed_offset')}
    merged['shards'] = [{'path': str(p), 'seed_offset': r['seed_offset'],
                         'n': len(r['episodes'])} for p, r in parts]
    merged['episodes'] = []
    for _p, r in parts:
        assert r['checkpoints'] == parts[0][1]['checkpoints'], 'shards from different checkpoints'
        for e in r['episodes']:
            e = dict(e, episode=r['seed_offset'] + e['episode'])
            assert e['marker_color'] == MARKER_COLORS[e['episode'] % len(MARKER_COLORS)]
            merged['episodes'].append(e)
    ids = [e['episode'] for e in merged['episodes']]
    assert len(ids) == len(set(ids)), 'overlapping shards'
    return _summarise(merged, results_path)


class DecentPi05Policy:
    """One arm's independently finetuned pi0.5 policy. No channel, no partner input."""

    def __init__(self, run_dir, arm, seed=0):
        os.environ.setdefault('OPENPI_DATA_HOME', '/scratch/users/ntu/ahaskar0/openpi_cache')
        import flax.nnx as nnx
        import jax
        import openpi.models.model as _model
        import openpi.models.tokenizer as _tokenizer
        import cola_pi05_model as C
        import pi05_marker_data as D
        import train_cola_pi05 as T

        run = Path(run_dir)
        if (run / 'resume_state').exists() or not (run / 'final_params').exists():
            raise SystemExit(f'{run} has not finished training (no final_params, or resume_state present)')
        self.meta = json.loads((run / 'final_meta.json').read_text())
        assert self.meta.get('arm') == arm, (
            f"{run} was trained with arm={self.meta.get('arm')}, expected {arm}")
        assert not self.meta['use_messages'], f'{run} has a message channel; this is the no-channel baseline'
        self.arm, self.D, self._model, self.jax = arm, D, _model, jax
        self.stats = _normalize_load(run / 'norm_stats.json')
        assert self.meta['prompts'] == D.PROMPT, 'prompts differ from training'
        self.tok = _tokenizer.PaligemmaTokenizer(self.meta['config']['max_token_len'])

        config = C.ColaPi05Config()
        model = T.build_model(config)
        graphdef, trainable, frozen = T.split_model(model, config)
        trainable.replace_by_pure_dict(T.restore_tree(run / 'final_params', T.pure(trainable)))

        @jax.jit
        def sample(tr, fr, o, r):
            return nnx.merge(graphdef, tr, fr).sample_arm_actions(r, o)
        self._sample = lambda o, r: sample(trainable, frozen, o, r)
        self.rng = jax.random.key(seed)
        print(f"   arm {arm}: {run} (step {self.meta['step']}, no channel)")

    def act(self, scene, renderer, camera):
        data = scene['data']
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        state = read_state(data, scene[self.arm])
        s = self.D.quantile_norm(state[None], self.stats[f'state_{self.arm}'])
        tok, mask = self.tok.tokenize(self.D.PROMPT[self.arm], s[0])
        obs = self._model.Observation.from_dict({
            'image': {'wrist_0_rgb': self.D.resize_224(image[None])},
            'image_mask': {'wrist_0_rgb': np.ones(1, bool)},
            'state': np.pad(s, ((0, 0), (0, self.D.ACTION_DIM - self.D.STATE_RAW))).astype(np.float32),
            'tokenized_prompt': tok[None].astype(np.int32),
            'tokenized_prompt_mask': mask[None].astype(bool)})
        self.rng, sub = self.jax.random.split(self.rng)
        x = self._sample(obs, sub)
        chunk = self.D.quantile_unnorm(np.asarray(x[0, :CHUNK_SIZE, :7], np.float64),
                                       self.stats[f'act_{self.arm}'])
        mid = (GRIPPER_OPEN + GRIPPER_CLOSED) / 2
        chunk[:, 6] = np.where(chunk[:, 6] > mid, GRIPPER_OPEN, GRIPPER_CLOSED)
        return chunk


def _normalize_load(path):
    import openpi.shared.normalize as _normalize
    return _normalize.deserialize_json(Path(path).read_text())


def check_cameras(scene_xml=SCENE_XML):
    """baselines/octo_small/scripts/check_marker_cameras.py, for the wrist views this
    policy reads, under this process's MuJoCo."""
    import h5py
    scene = build_scene(scene_xml)
    r = mujoco.Renderer(scene['model'], 256, 256)
    eps = json.load(open(f'{CACHE_DIR}/split_manifest.json'))['splits']['val']
    worst = 0.0
    for color in MARKER_COLORS:
        path = None
        for p in eps:
            with h5py.File(p, 'r') as f:
                c = f.attrs['marker_color']
                c = c.decode() if isinstance(c, bytes) else str(c)
            if c == color:
                path = p
                break
        reset_episode(scene, scene_xml, color)
        with h5py.File(path, 'r') as f:
            for key, cam in (('image_wrist_a', 'wrist_cam_left'), ('image_wrist_b', 'wrist_cam_right')):
                r.update_scene(scene['data'], camera=cam)
                worst = max(worst, float(np.abs(r.render().astype(float) - f[key][0].astype(float)).mean()))
    r.close()
    print(f'   cameras: worst MAE vs cached frame 0 = {worst:.2f} (MuJoCo {mujoco.__version__})')
    if worst > 10:
        raise SystemExit('!! marker scene cameras do NOT match the dataset -- do not evaluate')


def evaluate_pi05_decent_marker(run_a, run_b, n_episodes=150,
                         scene_xml=SCENE_XML, save_videos=True,
                         video_dir='evaluation_videos_cola_pi05_marker',
                         results_path='logs/cola_pi05_eval_marker_results.json',
                         max_control_steps=400, video_camera='side_cam',
                         seed_offset=0, seed=0):

    print('=' * 64)
    print('DECENTRALISED FINETUNED pi0.5 -- HIDDEN-MARKER HANDOVER (no channel)')
    print('=' * 64)
    print(f'Arm a: {run_a}\nArm b: {run_b}')
    print(f'Episodes: {n_episodes} (seed offset {seed_offset})  |  chance = 33.3% (three trays)')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading two independent pi0.5 policies (no channel)...')
    policy_a = DecentPi05Policy(run_a, 'a', seed=seed + seed_offset)
    policy_b = DecentPi05Policy(run_b, 'b', seed=seed + seed_offset + 1)
    # Names the results dict below records, kept so that block stays CoLA's.
    use_messages = False
    use_overhead = False
    print('   messages: none -- two independent policies | cameras: wrist only')

    print('\n2. Building scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['a']['subtree'], scene['b']['subtree']]
    print('   scene ready')

    results = {
        'model': 'decentralised LoRA-finetuned pi0.5, two policies, no channel',
        'checkpoints': {'a': str(run_a), 'b': str(run_b)},
        'seed_offset': seed_offset,
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

    print(f'\n3. Running {n_episodes} episodes...')
    for ep in tqdm(range(n_episodes), desc='Evaluating'):
        # Same cycling the collection used, so the eval is colour-balanced by
        # construction: 150 episodes is exactly 50/50/50.
        color = MARKER_COLORS[(seed_offset + ep) % len(MARKER_COLORS)]
        reset_episode(scene, scene_xml, color)

        frames = []
        transfer_done = False
        transfer_run = 0
        settle_run = 0
        landed = None          # tray the box settled in
        control_step = 0

        while control_step < max_control_steps and landed is None:
            # Each arm renders its own wrist camera and reads its own state;
            # the 64-d message is the only thing that crosses. act_both returns
            # CHUNK_SIZE absolute joint targets plus the gripper endpoint.
            chunk_a = policy_a.act(scene, renderer, 'wrist_cam_left')
            chunk_b = policy_b.act(scene, renderer, 'wrist_cam_right')

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
            'control_steps': control_step,
            'final_box_pos': [float(v) for v in box],
        })

    renderer.close()

    return _summarise(results, results_path, save_videos, videos_dir)


def _summarise(results, results_path, save_videos=False, videos_dir=None):
    """CoLA's summary and printout, verbatim (check_criterion verifies)."""
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
    p.add_argument('--run-a')
    p.add_argument('--run-b')
    p.add_argument('--episodes', type=int, default=150)
    p.add_argument('--scene-xml', default=SCENE_XML)
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--video-dir', default='evaluation_videos_pi05_decent_marker')
    p.add_argument('--results-path', required=True)
    p.add_argument('--max-control-steps', type=int, default=400)
    p.add_argument('--camera', default='side_cam')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--merge', nargs='+', help='shard results.json files to pool')
    a = p.parse_args()

    check_criterion()
    if a.merge:
        merge(a.merge, a.results_path)
    else:
        assert a.run_a and a.run_b, '--run-a and --run-b required'
        check_cameras(a.scene_xml)
        evaluate_pi05_decent_marker(a.run_a, a.run_b, n_episodes=a.episodes,
                             scene_xml=a.scene_xml, save_videos=not a.no_videos,
                             video_dir=a.video_dir, results_path=a.results_path,
                             max_control_steps=a.max_control_steps,
                             video_camera=a.camera, seed_offset=a.seed_offset,
                             seed=a.seed)
