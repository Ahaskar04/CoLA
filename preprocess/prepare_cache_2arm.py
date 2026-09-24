"""Build the two-arm training cache: normalised actions and splits.

Writes to cache_dir:
    {split}_actions_{a,b}.npy   (N, 7), min/max mapped to [-0.9, 0.9]
    {split}_episode_lens.npy    (n_episodes,)
    action_stats.json           per-arm min/max, for denormalising
    split_manifest.json         episode paths in row order

The normaliser is fitted on the train split. The split is 80/10/10, stratified
by episode success. Run prepare_states_2arm.py next for the joint states.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / 'data' / 'demos' / 'handover_2arm'
CACHE_DIR = REPO / 'data' / 'cache' / 'handover_2arm'

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = remainder = 0.10

SEED = 42

# Actions map to [-MARGIN, MARGIN], leaving headroom inside tanh's range.
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
                lengths.append(len(f['action_a']))
                successes.append(bool(f.attrs['success']))
            keep.append(p)
        except Exception as e:
            print(f'  SKIP {p.name}: {e}')
    return keep, np.array(successes), np.array(lengths, dtype=np.int64)


def stratified_split(episodes, successes, rng):
    """Split into (train, val, test), keeping the success ratio in each."""
    episodes = np.array(episodes, dtype=object)

    def _split(arr):
        arr = arr.copy()
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # test gets the remainder so the three parts always sum to n
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    s_train, s_val, s_test = _split(episodes[successes])
    f_train, f_val, f_test = _split(episodes[~successes])

    train = np.concatenate([s_train, f_train])
    val = np.concatenate([s_val, f_val])
    test = np.concatenate([s_test, f_test])

    for part in (train, val, test):
        rng.shuffle(part)

    return list(train), list(val), list(test)


def load_actions(paths):
    """Concatenate action_a/action_b over `paths`, plus per-episode lengths."""
    acts_a, acts_b, lens = [], [], []
    for p in paths:
        with h5py.File(p, 'r') as f:
            a = f['action_a'][:].astype(np.float32)
            b = f['action_b'][:].astype(np.float32)
        assert len(a) == len(b), f'{p.name}: action_a/action_b length mismatch'
        acts_a.append(a)
        acts_b.append(b)
        lens.append(len(a))
    return (
        np.concatenate(acts_a),
        np.concatenate(acts_b),
        np.array(lens, dtype=np.int64),
    )


def to_velocity(actions, lens):
    """Absolute joint targets -> per-step deltas, within each episode.

    The binary gripper column stays absolute; each episode's first delta is 0.
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
    # Avoid dividing by zero on constant columns.
    span = np.where(span < 1e-8, 1.0, span)
    return lo.astype(np.float32), hi.astype(np.float32), span.astype(np.float32)


def normalise(actions, lo, span):
    """Map raw actions onto [-MARGIN, MARGIN], clipped to [-1, 1]."""
    scaled = (actions - lo) / span * (2 * MARGIN) - MARGIN
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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

    # Optional episode cap, applied before the split and stratified by success.
    if args.max_episodes and args.max_episodes < len(episodes):
        keep_idx = []
        for mask in (successes, ~successes):
            idx = np.flatnonzero(mask)
            rng.shuffle(idx)
            take = int(round(args.max_episodes * len(idx) / len(episodes)))
            keep_idx.append(idx[:take])
        keep = np.sort(np.concatenate(keep_idx))
        # episodes is a list; successes and lengths are arrays.
        episodes = [episodes[i] for i in keep]
        successes, lengths = successes[keep], lengths[keep]
        n_succ = int(successes.sum())
        print(f'  capped to {len(episodes)} episodes '
              f'(success {n_succ}, failure {len(episodes) - n_succ})')

    train, val, test = stratified_split(episodes, successes, rng)
    splits = {'train': train, 'val': val, 'test': test}

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Fit the normaliser on train only, then apply it to all three splits.
    print('\nLoading train actions to fit the normaliser...')
    train_a, train_b, train_lens = load_actions(train)
    if args.velocity:
        # Fit the normaliser on deltas (their range differs from positions).
        train_a = to_velocity(train_a, train_lens)
        train_b = to_velocity(train_b, train_lens)
    lo_a, hi_a, span_a = fit_normaliser(train_a)
    lo_b, hi_b, span_b = fit_normaliser(train_b)

    np.set_printoptions(precision=4, suppress=True)
    print(f'  arm A min {lo_a}\n  arm A max {hi_a}')
    print(f'  arm B min {lo_b}\n  arm B max {hi_b}')

    with open(args.cache_dir / 'action_stats.json', 'w') as f:
        json.dump({
            'margin': MARGIN,
            # Tells the rollout whether to integrate actions or apply them directly.
            'velocity': bool(args.velocity),
            'source': 'train split of ' + str(args.data_dir),
            'action_a': {'min': lo_a.tolist(), 'max': hi_a.tolist()},
            'action_b': {'min': lo_b.tolist(), 'max': hi_b.tolist()},
            'note': 'raw = (norm + margin) / (2*margin) * (max - min) + min',
        }, f, indent=2)

    manifest = {'seed': args.seed, 'data_dir': str(args.data_dir), 'splits': {}}

    print()
    for name, paths in splits.items():
        if name == 'train':
            acts_a, acts_b, lens = train_a, train_b, train_lens
        else:
            acts_a, acts_b, lens = load_actions(paths)
            if args.velocity:
                acts_a = to_velocity(acts_a, lens)
                acts_b = to_velocity(acts_b, lens)

        np.save(args.cache_dir / f'{name}_actions_a.npy', normalise(acts_a, lo_a, span_a))
        np.save(args.cache_dir / f'{name}_actions_b.npy', normalise(acts_b, lo_b, span_b))
        np.save(args.cache_dir / f'{name}_episode_lens.npy', lens)

        manifest['splits'][name] = [str(p) for p in paths]

        n_clipped = int(
            (np.abs(normalise(acts_a, lo_a, span_a)) >= 1.0).sum()
            + (np.abs(normalise(acts_b, lo_b, span_b)) >= 1.0).sum()
        )
        print(f'  {name:5s}: {len(paths):3d} episodes, {int(lens.sum()):6d} timesteps'
              f'  ({n_clipped} values clipped at +/-1)')

    with open(args.cache_dir / 'split_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nWrote actions, episode lengths and manifest to {args.cache_dir}')
    print('Next: extract_features_2arm.py')


if __name__ == '__main__':
    main()
