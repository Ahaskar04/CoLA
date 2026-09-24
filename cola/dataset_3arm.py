"""Three-arm dataset: cached Octo-Base features and action chunks.

Same format as cola/dataset.py, with per-arm dicts:
    features: {'a': (B, 768), 'b': ..., 'c': ...}
    action:   {'a': (B, chunk, 7), ...}
    state:    {'a': (B, 7), ...}   (use_states only)
"""

import numpy as np
from pathlib import Path
from typing import Dict


CHUNK_SIZE = 10  # must match cola_architecture_3arm.CHUNK_SIZE
ARMS = ('a', 'b', 'c')


def _build_valid_start_indices(episode_lens: np.ndarray, chunk_size: int) -> np.ndarray:
    """Global indices where a chunk of `chunk_size` actions stays in one episode."""
    starts = np.concatenate([[0], np.cumsum(episode_lens)[:-1]])
    counts = np.clip(episode_lens - chunk_size + 1, a_min=0, a_max=None)

    valid = np.empty(int(counts.sum()), dtype=np.int64)
    write = 0
    for s, c in zip(starts, counts):
        valid[write:write + c] = np.arange(s, s + c)
        write += int(c)
    return valid


class COLADataset3Arm:
    """Pre-extracted 768-d Octo features, actions and (optionally) states, three arms."""

    def __init__(
        self,
        cache_dir: str,
        feat_dir: str,
        split: str = 'train',
        chunk_size: int = CHUNK_SIZE,
        use_states: bool = False,
        use_overhead: bool = False,
        **kwargs,
    ):
        self.split = split
        self.chunk_size = chunk_size
        self.use_states = use_states
        self.use_overhead = use_overhead
        cache_dir = Path(cache_dir)
        feat_dir = Path(feat_dir)

        print(f'[{split}] Loading pre-extracted features...')
        self.features = {
            arm: np.load(feat_dir / f'{split}_features_{arm}.npy') for arm in ARMS
        }
        # Overhead view shared by all arms.
        self.features_o = None
        if use_overhead:
            fo = feat_dir / f'{split}_features_o.npy'
            if not fo.exists():
                raise FileNotFoundError(
                    f'{fo} not found. --overhead needs the overhead features: '
                    f're-run extract_features_3arm.py --overhead after deleting '
                    f'the feature dir (it skips when files already exist).')
            self.features_o = np.load(fo)

        self.actions = {
            arm: np.load(cache_dir / f'{split}_actions_{arm}.npy') for arm in ARMS
        }

        # States from prepare_cache_3arm.py, row-aligned with features and actions.
        self.states = None
        if use_states:
            first = cache_dir / f'{split}_states_a.npy'
            if not first.exists():
                raise FileNotFoundError(
                    f'Missing {first}. Run prepare_cache_3arm.py first.')
            self.states = {
                arm: np.load(cache_dir / f'{split}_states_{arm}.npy') for arm in ARMS
            }

        # All arrays must share one timeline.
        n = len(self.features['a'])
        for arm in ARMS:
            assert len(self.features[arm]) == n, \
                f'{split}: features_a has {n} rows, features_{arm} has {len(self.features[arm])}'
            assert len(self.actions[arm]) == n, \
                f'{split}: features has {n} rows, actions_{arm} has {len(self.actions[arm])}'
            if use_states:
                assert len(self.states[arm]) == n, \
                    f'{split}: features has {n} rows, states_{arm} has {len(self.states[arm])}'
        if use_overhead:
            assert len(self.features_o) == n, \
                f'{split}: features has {n} rows, features_o has {len(self.features_o)}'

        lens_path = cache_dir / f'{split}_episode_lens.npy'
        if not lens_path.exists():
            raise FileNotFoundError(
                f'Missing {lens_path}. Run prepare_cache_3arm.py first.')
        episode_lens = np.load(lens_path)
        assert int(episode_lens.sum()) == n, \
            f'{split}: episode_lens sum to {int(episode_lens.sum())}, data has {n} rows'

        self.valid_starts = _build_valid_start_indices(episode_lens, chunk_size)
        print(
            f'[{split}] {n} timesteps, {len(episode_lens)} episodes, '
            f'{len(self.valid_starts)} valid chunk-starts (chunk_size={chunk_size})'
        )

    def _load_batch(self, start_indices):
        offsets = np.arange(self.chunk_size, dtype=np.int64)
        idx_grid = start_indices[:, None] + offsets[None, :]

        batch = {
            'features': {arm: self.features[arm][start_indices] for arm in ARMS},
            'action': {arm: self.actions[arm][idx_grid] for arm in ARMS},
        }
        if self.use_overhead:
            batch['features_o'] = self.features_o[start_indices]
        if self.use_states:
            # State at the query timestep only, like the features.
            batch['state'] = {arm: self.states[arm][start_indices] for arm in ARMS}
        return batch

    def get_batch(self, batch_size: int) -> Dict:
        chosen = np.random.choice(len(self.valid_starts), batch_size, replace=False)
        return self._load_batch(self.valid_starts[chosen])

    def __len__(self):
        return len(self.valid_starts)


if __name__ == '__main__':
    import sys
    data = Path(__file__).resolve().parents[1] / 'data'
    cache = sys.argv[1] if len(sys.argv) > 1 else data / 'cache' / 'handover_3arm'
    feats = sys.argv[2] if len(sys.argv) > 2 else data / 'features' / 'handover_3arm'
    ds = COLADataset3Arm(cache_dir=cache, feat_dir=feats, split='train')
    print(f'\nDataset size (valid chunk-starts): {len(ds)}')
    batch = ds.get_batch(32)
    print('\nBatch shapes:')
    for k, v in batch.items():
        if isinstance(v, dict):
            for arm, arr in v.items():
                print(f'  {k}[{arm}]: {arr.shape} {arr.dtype}')
        else:
            print(f'  {k}: {v.shape} {v.dtype}')
    for arm in ARMS:
        assert batch['features'][arm].shape == (32, 768)
        assert batch['action'][arm].shape == (32, CHUNK_SIZE, 7)
    print('\nOK')
