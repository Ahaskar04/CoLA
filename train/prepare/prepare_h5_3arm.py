"""
Build the 3-arm training cache: normalised actions, z-scored states, splits.

Port of prepare_h5_handover.py + prepare_states_h5.py (the 2-arm pair), merged
into one script because there was no reason to walk the same 185 episodes twice.

Writes, for each of train/val/test:

    {split}_actions_{a,b,c}.npy   (N, 7) float32, min/max mapped to [-0.9, 0.9]
    {split}_states_{a,b,c}.npy    (N, 7) float32, z-scored
    {split}_episode_lens.npy      (n_episodes,) int64
    action_stats.json             per-arm min/max, for denormalising at rollout
    state_stats.json              per-arm mean/std
    split_manifest.json           episode paths in the order rows were written

Row i of every array refers to the same timestep of the same episode. That
alignment is the whole contract: extract_features_3arm.py reads the manifest and
must produce features in exactly this order.

Why z-score states but min/max the actions: the action normaliser exists to fit
a tanh output range. States are an input, so what matters is only that each
dimension arrives at a comparable scale.

NOTE ON FAILED EPISODES. The 2-arm script kept them and stratified the split by
the `success` attr. The 3-arm dataset at aloha-handover-3arm-v1 has already had
its 115 failures deleted, so every episode there is a success and the
stratification is a no-op. It is kept anyway: it costs nothing, and it keeps
this script correct against an un-filtered directory.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path('/scratch/users/ntu/ahaskar0/aloha-handover-3arm-v1')
CACHE_DIR = Path('/scratch/users/ntu/ahaskar0/v1/cola-3arm-scratchdata/cache_aloha_handover_3arm')

ARMS = ('a', 'b', 'c')

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = remainder = 0.10

SEED = 42

# Raw actions are mapped onto [-MARGIN, MARGIN] rather than [-1, 1]: tanh only
# reaches +/-1 asymptotically, so leaving headroom keeps the extreme timesteps
# reachable instead of demanding infinite pre-activations.
MARGIN = 0.9
# Column 6 of an action is the gripper; 0..5 are joints.
GRIPPER_COL = 6


def scan_episodes(data_dir: Path):
    """Return (paths, success_flags, lengths) for every readable episode."""
    paths = sorted(data_dir.glob('episode_*.h5'))
    if not paths:
        raise SystemExit(f'No episode_*.h5 files in {data_dir}')

    keep, successes, lengths = [], [], []
    for p in paths:
        try:
            with h5py.File(p, 'r') as f:
                # Assert the three-arm schema up front rather than failing
                # halfway through a 40 GB read.
                for arm in ARMS:
                    for key in (f'action_{arm}', f'state_{arm}'):
                        if key not in f:
                            raise KeyError(f'missing {key}')
                lengths.append(len(f['action_a']))
                successes.append(bool(f.attrs['success']))
            keep.append(p)
        except Exception as e:
            print(f'  SKIP {p.name}: {e}')
    return keep, np.array(successes), np.array(lengths, dtype=np.int64)


def stratified_split(episodes, successes, rng):
    """Split into (train, val, test), keeping the success ratio in each."""
    episodes = np.array(episodes, dtype=object)

    def _split(subset):
        subset = subset.copy()
        rng.shuffle(subset)
        n = len(subset)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        return subset[:n_train], subset[n_train:n_train + n_val], subset[n_train + n_val:]

    s_train, s_val, s_test = _split(episodes[successes])
    f_train, f_val, f_test = _split(episodes[~successes])

    train = np.concatenate([s_train, f_train])
    val = np.concatenate([s_val, f_val])
    test = np.concatenate([s_test, f_test])

    for part in (train, val, test):
        rng.shuffle(part)

    return list(train), list(val), list(test)


def load_arrays(paths, kind):
    """Concatenate {kind}_{a,b,c} over `paths`, plus per-episode lengths.

    kind is 'action' or 'state'. Returns ({arm: (N,7)}, lens).
    """
    acc = {arm: [] for arm in ARMS}
    lens = []
    for p in paths:
        with h5py.File(p, 'r') as f:
            arrs = {arm: f[f'{kind}_{arm}'][:].astype(np.float32) for arm in ARMS}
        n = len(arrs['a'])
        for arm in ARMS:
            assert len(arrs[arm]) == n, \
                f'{Path(p).name}: {kind}_a has {n} rows, {kind}_{arm} has {len(arrs[arm])}'
            acc[arm].append(arrs[arm])
        lens.append(n)
    return ({arm: np.concatenate(acc[arm]) for arm in ARMS},
            np.array(lens, dtype=np.int64))


def to_velocity(actions, lens):
    """Absolute joint targets -> per-step deltas, respecting episode boundaries.

    The gripper column is left ABSOLUTE. It is binary and handled by a BCE head;
    differencing it would produce three values (-1/0/+1 open, hold, close) and
    break that head's assumption.

    Differencing must not cross episode boundaries -- the last action of one
    episode and the first of the next are unrelated poses, and a delta between
    them is a fictional jump. The first step of each episode gets a zero delta.
    """
    out = np.zeros_like(actions)
    start = 0
    for n in lens:
        ep = actions[start:start + n]
        d = np.zeros_like(ep)
        d[1:, :GRIPPER_COL] = ep[1:, :GRIPPER_COL] - ep[:-1, :GRIPPER_COL]
        d[:, GRIPPER_COL] = ep[:, GRIPPER_COL]      # gripper stays absolute
        out[start:start + n] = d
        start += n
    assert start == len(actions), 'episode lengths do not cover the action array'
    return out


def fit_normaliser(actions):
    """Per-dimension min/max, with degenerate (constant) columns left alone."""
    lo = actions.min(axis=0)
    hi = actions.max(axis=0)
    span = hi - lo
    # A constant column would divide by zero; map it to 0 instead.
    span = np.where(span < 1e-8, 1.0, span)
    return lo.astype(np.float32), hi.astype(np.float32), span.astype(np.float32)


def normalise(actions, lo, span):
    """Map raw actions onto [-MARGIN, MARGIN], clipped to [-1, 1]."""
    scaled = (actions - lo) / span * (2 * MARGIN) - MARGIN
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', type=Path, default=DATA_DIR)
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--velocity', action='store_true',
                    help='store per-step joint DELTAS instead of absolute joint '
                         'targets (gripper column stays absolute). Rollout must '
                         'then integrate predictions onto the current pose -- a '
                         'cache built this way is NOT interchangeable with a '
                         'position one, so use a separate cache_dir.')
    ap.add_argument('--max_episodes', type=int, default=0,
                    help='use at most this many episodes, sampled randomly and '
                         'stratified by success, before the 80/10/10 split. '
                         '0 = use everything.')
    args = ap.parse_args()

    print(f'Scanning {args.data_dir} ...')
    episodes, successes, lengths = scan_episodes(args.data_dir)
    n_succ = int(successes.sum())
    print(f'  {len(episodes)} episodes, {int(lengths.sum())} timesteps')
    print(f'  success: {n_succ}, failure: {len(episodes) - n_succ} '
          f'({n_succ / len(episodes) * 100:.1f}%)')

    rng = np.random.default_rng(args.seed)

    # Optional cap on how many episodes to use at all. Subsample BEFORE the
    # split so train/val/test stay 80/10/10 of the capped set, and do it
    # stratified so the success ratio is preserved.
    if args.max_episodes and args.max_episodes < len(episodes):
        keep_idx = []
        for mask in (successes, ~successes):
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            take = int(round(args.max_episodes * len(idx) / len(episodes)))
            keep_idx.append(idx[:take])
        keep = np.sort(np.concatenate(keep_idx))
        episodes = [episodes[i] for i in keep]
        successes, lengths = successes[keep], lengths[keep]
        n_succ = int(successes.sum())
        print(f'  capped to {len(episodes)} episodes '
              f'(success {n_succ}, failure {len(episodes) - n_succ})')

    train, val, test = stratified_split(episodes, successes, rng)
    splits = {'train': train, 'val': val, 'test': test}

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- actions: fit the normaliser on train only, apply to all splits ----
    print('\nLoading train actions to fit the normaliser...')
    train_act, train_lens = load_arrays(train, 'action')
    if args.velocity:
        # Fit on deltas, not absolute targets: their ranges differ by an order
        # of magnitude, so reusing position stats would squash every delta to
        # near zero after normalisation.
        train_act = {a: to_velocity(train_act[a], train_lens) for a in ARMS}
    norm = {a: fit_normaliser(train_act[a]) for a in ARMS}

    np.set_printoptions(precision=4, suppress=True)
    for a in ARMS:
        print(f'  arm {a.upper()} min {norm[a][0]}\n  arm {a.upper()} max {norm[a][1]}')

    stats = {
        'margin': MARGIN,
        # Rollout must know: velocity actions are integrated onto the current
        # pose, position actions are written straight to ctrl.
        'velocity': bool(args.velocity),
        'source': 'train split of ' + str(args.data_dir),
        'note': 'raw = (norm + margin) / (2*margin) * (max - min) + min',
    }
    for a in ARMS:
        stats[f'action_{a}'] = {'min': norm[a][0].tolist(), 'max': norm[a][1].tolist()}
    with open(args.cache_dir / 'action_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # ---- states: z-score, fitted on train only ----
    print('\nLoading train states to fit the z-scoring...')
    train_state, state_lens = load_arrays(train, 'state')
    assert np.array_equal(train_lens, state_lens), \
        'action and state episode lengths disagree on the train split'
    smean, sstd = {}, {}
    for a in ARMS:
        smean[a] = train_state[a].mean(axis=0)
        # A constant column would divide by zero.
        sstd[a] = np.where(train_state[a].std(axis=0) < 1e-8, 1.0,
                           train_state[a].std(axis=0))
    with open(args.cache_dir / 'state_stats.json', 'w') as f:
        json.dump({a: {'mean': smean[a].tolist(), 'std': sstd[a].tolist()}
                   for a in ARMS}, f, indent=2)

    manifest = {'seed': args.seed, 'data_dir': str(args.data_dir),
                'arms': list(ARMS), 'splits': {}}

    print()
    for name, paths in splits.items():
        if name == 'train':
            act, lens = train_act, train_lens
            state = train_state
        else:
            act, lens = load_arrays(paths, 'action')
            if args.velocity:
                act = {a: to_velocity(act[a], lens) for a in ARMS}
            state, state_lens = load_arrays(paths, 'state')
            assert np.array_equal(lens, state_lens), \
                f'{name}: action and state episode lengths disagree'

        n_clipped = 0
        for a in ARMS:
            lo, hi, span = norm[a]
            normed = normalise(act[a], lo, span)
            np.save(args.cache_dir / f'{name}_actions_{a}.npy', normed)
            np.save(args.cache_dir / f'{name}_states_{a}.npy',
                    ((state[a] - smean[a]) / sstd[a]).astype(np.float32))
            n_clipped += int((np.abs(normed) >= 1.0).sum())

        np.save(args.cache_dir / f'{name}_episode_lens.npy', lens)
        manifest['splits'][name] = [str(p) for p in paths]

        print(f'  {name:5s}: {len(paths):3d} episodes, {int(lens.sum()):6d} timesteps'
              f'  ({n_clipped} values clipped at +/-1)')

    with open(args.cache_dir / 'split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nWrote actions, states, episode lengths and manifest to {args.cache_dir}')
    print('Next: extract_features_3arm.py')


if __name__ == '__main__':
    main()
