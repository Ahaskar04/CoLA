"""Evaluate CoLA on pi0.5 (LoRA-finetuned) on the hidden-marker handover.

One pi0.5 model drives both arms, each from its own wrist camera and joint
state; the 64-d message is the only thing shared. Reset, success criterion
and summary are CoLA's evaluator code (checked by check_criterion()), and
check_cameras() confirms the scene renders the training views.

Runs in the openpi environment with MuJoCo 3.12.0 (as used for CoLA's evals).
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'eval'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# eval_marker imports cola.model, which needs Octo (absent here).
# Only CHUNK_SIZE is used, so stub it, reading the value from CoLA's source.
import re                                                  # noqa: E402
import types                                               # noqa: E402
_COLA_ARCH = REPO / 'cola' / 'model.py'
_stub = types.ModuleType('cola.model')
_stub.CHUNK_SIZE = int(re.search(r'^CHUNK_SIZE = (\d+)', Path(_COLA_ARCH).read_text(), re.M).group(1))
_stub.COLAModel = None
sys.modules['cola.model'] = _stub

from eval_marker import (                                  # noqa: E402
    CHUNK_SIZE, CONTROL_DECIMATION, GRIPPER_OPEN, GRIPPER_OPEN_FRAC, HOLD_STEPS,
    LIFT_Z, MARKER_COLORS, SCENE_XML, SETTLE_STEPS, TRAY_RADIUS, TRAY_Z_MAX,
    apply_action, build_scene, compensate_gravity, reset_episode, touching_box,
    which_tray)
from eval_marker import CACHE_DIR, GRIPPER_CLOSED, read_state        # noqa: E402

COLA_SOURCE = str(REPO / 'eval' / 'eval_marker.py')
RESULTS_DIR = REPO / 'results' / 'cola_pi05_lora'
# Built from pieces so the search doesn't match these lines themselves.
_SCORE_START = ' ' * 12 + 'for k in range(' + 'CHUNK_SIZE):'
_SCORE_END = ' ' * 4 + 'renderer.' + 'close()'
_SUM_START = ' ' * 4 + "eps = results[" + "'episodes']"
_SUM_END = ' ' * 4 + 'return ' + 'results\n'


def _block(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def check_criterion():
    """Exit if the scoring or summary code differs from CoLA's."""
    cola = Path(COLA_SOURCE).read_text()
    mine = Path(__file__).read_text()
    for name, s, e in (('scoring loop', _SCORE_START, _SCORE_END),
                       ('summary', _SUM_START, _SUM_END)):
        if _block(cola, s, e) != _block(mine, s, e):
            raise SystemExit(f'{name} differs from {COLA_SOURCE} -- regenerate '
                             f'eval_cola_pi05_lora.py before trusting a number from it')
    print("   criterion: identical to CoLA's evaluator (scoring loop + summary)")


def merge(shards, results_path):
    """Merge sharded runs and recompute the summary."""
    # Keep each shard with its own path, sorted by seed offset.
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


class ColaPi05Policy:
    """Both arms' chunks for one control step, from a finished CoLA-pi0.5 run."""

    def __init__(self, run_dir, seed=0):
        import jax
        import openpi.models.model as _model
        import openpi.models.tokenizer as _tokenizer
        import cola_pi05_lora as C
        import pi05_data as D
        import train_cola_pi05_lora as T

        run = Path(run_dir)
        if (run / 'resume_state').exists() or not (run / 'final_params').exists():
            raise SystemExit(f'{run} has not finished training (no final_params, or resume_state present)')
        self.meta = json.loads((run / 'final_meta.json').read_text())
        self.use_messages = bool(self.meta['use_messages'])
        self.D, self._model, self.jax = D, _model, jax
        self.stats = _normalize_load(run / 'norm_stats.json')
        assert self.meta['prompts'] == D.PROMPT, 'prompts differ from training'
        self.tok = _tokenizer.PaligemmaTokenizer(self.meta['config']['max_token_len'])

        config = C.ColaPi05Config()
        model = T.build_model(config)
        graphdef, trainable, frozen = T.split_model(model, config)
        trainable.replace_by_pure_dict(T.restore_tree(run / 'final_params', T.pure(trainable)))
        import flax.nnx as nnx
        use_messages = self.use_messages

        # Pass weights as jit arguments so they aren't baked in as constants.
        @jax.jit
        def sample(tr, fr, oa, ob, r):
            return nnx.merge(graphdef, tr, fr).sample_cola_actions(r, oa, ob, use_messages=use_messages)
        self._sample = lambda oa, ob, r: sample(trainable, frozen, oa, ob, r)
        self.rng = jax.random.key(seed)
        print(f'   loaded {run} (step {self.meta["step"]}, messages {"ON" if self.use_messages else "OFF"})')

    def _obs(self, image, state, arm):
        s = self.D.quantile_norm(state[None], self.stats[f'state_{arm}'])
        tok, mask = self.tok.tokenize(self.D.PROMPT[arm], s[0])
        return self._model.Observation.from_dict({
            'image': {'wrist_0_rgb': self.D.resize_224(image[None])},
            'image_mask': {'wrist_0_rgb': np.ones(1, bool)},
            'state': np.pad(s, ((0, 0), (0, self.D.ACTION_DIM - self.D.STATE_RAW))).astype(np.float32),
            'tokenized_prompt': tok[None].astype(np.int32),
            'tokenized_prompt_mask': mask[None].astype(bool)})

    def act_both(self, scene, renderer):
        data = scene['data']
        renderer.update_scene(data, camera='wrist_cam_left')
        image_a = renderer.render()
        renderer.update_scene(data, camera='wrist_cam_right')
        image_b = renderer.render()
        obs_a = self._obs(image_a, read_state(data, scene['a']), 'a')
        obs_b = self._obs(image_b, read_state(data, scene['b']), 'b')
        self.rng, sub = self.jax.random.split(self.rng)
        xa, xb = self._sample(obs_a, obs_b, sub)
        mid = (GRIPPER_OPEN + GRIPPER_CLOSED) / 2
        out = []
        for x, arm in ((xa, 'a'), (xb, 'b')):
            chunk = self.D.quantile_unnorm(np.asarray(x[0, :CHUNK_SIZE, :7], np.float64),
                                           self.stats[f'act_{arm}'])
            chunk[:, 6] = np.where(chunk[:, 6] > mid, GRIPPER_OPEN, GRIPPER_CLOSED)
            out.append(chunk)
        return out[0], out[1]


def _normalize_load(path):
    import openpi.shared.normalize as _normalize
    return _normalize.deserialize_json(Path(path).read_text())


def check_cameras(scene_xml=SCENE_XML):
    """Check that the scene's wrist cameras reproduce the cached training frames."""
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


def evaluate_cola_pi05_marker(run_dir, n_episodes=150,
                         scene_xml=SCENE_XML, save_videos=True,
                         video_dir=str(RESULTS_DIR / 'videos'),
                         results_path=str(RESULTS_DIR / 'results.json'),
                         max_control_steps=400, video_camera='side_cam',
                         seed_offset=0, seed=0):

    print('=' * 64)
    print('CoLA-on-pi0.5 HIDDEN-MARKER HANDOVER EVALUATION')
    print('=' * 64)
    print(f'Run: {run_dir}')
    print(f'Episodes: {n_episodes} (seed offset {seed_offset})  |  chance = 33.3% (three trays)')

    videos_dir = Path(video_dir)
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    print('\n1. Loading CoLA-pi0.5 (one model, two decentralised arms, 64-d channel)...')
    policy = ColaPi05Policy(run_dir, seed=seed + seed_offset)
    # Recorded in results, as in CoLA's evaluator.
    use_messages = policy.use_messages
    use_overhead = False
    print(f"   messages: {'ON' if use_messages else 'OFF (L0 control)'} | cameras: wrist only")

    print('\n2. Building scene...')
    scene = build_scene(scene_xml)
    model, data = scene['model'], scene['data']
    renderer = mujoco.Renderer(model, 256, 256)
    subtrees = [scene['a']['subtree'], scene['b']['subtree']]
    print('   scene ready')

    results = {
        'model': 'CoLA on pi0.5 (LoRA-finetuned), decentralised with 64-d channel',
        'checkpoints': {'run': str(run_dir)},
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
        # Colours cycle with the episode index, so the eval is balanced.
        color = MARKER_COLORS[(seed_offset + ep) % len(MARKER_COLORS)]
        reset_episode(scene, scene_xml, color)
        donor_color = None  # no message swap here; kept so results match eval_marker.py

        frames = []
        transfer_done = False
        transfer_run = 0
        settle_run = 0
        landed = None          # tray the box settled in
        control_step = 0

        while control_step < max_control_steps and landed is None:
            # Each arm uses its own wrist camera and state; only the message is
            # shared. act_both returns CHUNK_SIZE actions in actuator units.
            chunk_a, chunk_b = policy.act_both(scene, renderer)

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
                    # The box must rest in a tray for SETTLE_STEPS to count.
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

    return _summarise(results, results_path, save_videos, videos_dir)


def _summarise(results, results_path, save_videos=False, videos_dir=None):
    """Summary and printout copied from eval_marker.py (checked by check_criterion)."""
    eps = results['episodes']
    n = len(eps)
    n_correct = sum(e['correct'] for e in eps)
    n_tray = sum(e['landed_tray'] is not None for e in eps)
    n_xfer = sum(e['transfer_done'] for e in eps)

    # Per-colour breakdown (catches a policy that always picks the same tray).
    by_color = {}
    for c in MARKER_COLORS:
        sub = [e for e in eps if e['marker_color'] == c]
        # Only episodes with a completed handover, as in the main metric.
        sub_x = [e for e in sub if e['transfer_done']]
        by_color[c] = {
            'n': len(sub),
            'correct': sum(e['correct'] for e in sub),
            'n_transfers': len(sub_x),
            'correct_given_transfer': sum(e['correct'] for e in sub_x),
            'chose': {t: sum(1 for e in sub_x if e['landed_tray'] == t)
                      for t in MARKER_COLORS + [None]},
        }

    # Main metric: correct tray out of the episodes where the handover happened.
    n_correct_xfer = sum(e['correct'] for e in eps if e['transfer_done'])
    correct_given_xfer = (100.0 * n_correct_xfer / n_xfer) if n_xfer else None

    results['summary'] = {
        'n_episodes': n,
        'correct_given_transfer': correct_given_xfer,
        'n_transfers': n_xfer,
        'n_correct_given_transfer': n_correct_xfer,
        'correct_rate': 100.0 * n_correct / n,
        'tray_rate': 100.0 * n_tray / n,
        'transfer_rate': 100.0 * n_xfer / n,
        # Correct among episodes that reached any tray.
        'correct_given_tray': (100.0 * n_correct / n_tray) if n_tray else None,
        'chance_rate': 100.0 / len(MARKER_COLORS),
        'by_color': by_color,
    }

    s = results['summary']
    print('\n' + '=' * 64)
    print('RESULTS')
    print('=' * 64)
    # Main metric.
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
    p.add_argument('--run-dir')
    p.add_argument('--episodes', type=int, default=150)
    p.add_argument('--scene-xml', default=SCENE_XML)
    p.add_argument('--no-videos', action='store_true')
    p.add_argument('--video-dir', default=str(RESULTS_DIR / 'videos'))
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
        assert a.run_dir, '--run-dir required'
        check_cameras(a.scene_xml)
        evaluate_cola_pi05_marker(a.run_dir, n_episodes=a.episodes,
                             scene_xml=a.scene_xml, save_videos=not a.no_videos,
                             video_dir=a.video_dir, results_path=a.results_path,
                             max_control_steps=a.max_control_steps,
                             video_camera=a.camera, seed_offset=a.seed_offset,
                             seed=a.seed)
