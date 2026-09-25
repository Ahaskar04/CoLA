"""Cache z-scored joint states for the two-arm datasets.

Run after prepare_cache_2arm.py. Writes, in manifest row order:
    {split}_states_{a,b}.npy   (N, 7), z-scored
    state_stats.json           per-arm mean/std, fitted on train
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'cache' / 'handover_2arm'


def load_states(paths):
    """Concatenate state_a/state_b over `paths`, in manifest order."""
    states_a, states_b, lens = [], [], []
    for p in paths:
        with h5py.File(p, 'r') as f:
            a = f['state_a'][:].astype(np.float32)
            b = f['state_b'][:].astype(np.float32)
        assert len(a) == len(b), f'{Path(p).name}: state_a/state_b length mismatch'
        states_a.append(a)
        states_b.append(b)
        lens.append(len(a))
    return np.concatenate(states_a), np.concatenate(states_b), np.array(lens, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    args = ap.parse_args()

    manifest_path = args.cache_dir / 'split_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Missing {manifest_path}. Run prepare_cache_2arm.py first.')
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Fit on train only, so val/test don't leak into the statistics.
    train_a, train_b, _ = load_states(manifest['splits']['train'])
    mean_a, std_a = train_a.mean(0), train_a.std(0)
    mean_b, std_b = train_b.mean(0), train_b.std(0)
    # A constant column would divide by zero.
    std_a = np.where(std_a < 1e-6, 1.0, std_a)
    std_b = np.where(std_b < 1e-6, 1.0, std_b)

    np.set_printoptions(precision=4, suppress=True)
    print(f'arm A state mean {mean_a}\n           std  {std_a}')
    print(f'arm B state mean {mean_b}\n           std  {std_b}')

    with open(args.cache_dir / 'state_stats.json', 'w') as f:
        json.dump({
            'source': 'train split of ' + manifest.get('data_dir', '?'),
            'state_a': {'mean': mean_a.tolist(), 'std': std_a.tolist()},
            'state_b': {'mean': mean_b.tolist(), 'std': std_b.tolist()},
            'note': 'normalised = (raw - mean) / std',
        }, f, indent=2)

    print()
    for split, paths in manifest['splits'].items():
        if split == 'train':
            states_a, states_b, lens = train_a, train_b, None
        else:
            states_a, states_b, lens = load_states(paths)

        # Check that the rows line up.
        n_actions = len(np.load(args.cache_dir / f'{split}_actions_a.npy', mmap_mode='r'))
        assert len(states_a) == n_actions, \
            f'{split}: {len(states_a)} states vs {n_actions} actions — manifest order changed'

        np.save(args.cache_dir / f'{split}_states_a.npy', (states_a - mean_a) / std_a)
        np.save(args.cache_dir / f'{split}_states_b.npy', (states_b - mean_b) / std_b)
        print(f'  {split:5s}: {len(paths):3d} episodes, {len(states_a):6d} timesteps')

    print(f'\nWrote states and state_stats.json to {args.cache_dir}')


if __name__ == '__main__':
    main()
