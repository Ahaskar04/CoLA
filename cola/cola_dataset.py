"""
COLA Dataset Loader (pre-extracted features + action chunking).

Loads 768-dim Octo-Base features (pre-extracted by extract_features.py)
and actions from the cache directory. Samples CHUNK_SIZE consecutive
actions per sample while respecting episode boundaries.
"""

import numpy as np
from pathlib import Path
from typing import Dict
from threading import Thread
from queue import Queue


CHUNK_SIZE = 10  # must match cola_architecture.CHUNK_SIZE


def _build_valid_start_indices(episode_lens: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Given concatenated episode lengths, return global timestep indices where
    a chunk of `chunk_size` consecutive actions stays within one episode.
    """
    starts = np.concatenate([[0], np.cumsum(episode_lens)[:-1]])
    counts = np.clip(episode_lens - chunk_size + 1, a_min=0, a_max=None)

    valid = np.empty(int(counts.sum()), dtype=np.int64)
    write = 0
    for s, c in zip(starts, counts):
        valid[write:write + c] = np.arange(s, s + c)
        write += int(c)
    return valid


class COLADataset:
    """
    Feature-based dataset: loads pre-extracted 768-dim Octo features (not images).
    """

    def __init__(
        self,
        cache_dir: str,
        feat_dir: str,
        split: str = 'train',
        prefetch: int = 4,
        chunk_size: int = CHUNK_SIZE,
        use_states: bool = False,
        **kwargs,
    ):
        self.split = split
        self.chunk_size = chunk_size
        self.use_states = use_states
        cache_dir = Path(cache_dir)
        feat_dir = Path(feat_dir)

        # Pre-extracted features (small — fits in RAM)
        print(f'[{split}] Loading pre-extracted features...')
        self.features_a = np.load(feat_dir / f'{split}_features_a.npy')
        self.features_b = np.load(feat_dir / f'{split}_features_b.npy')

        # Proprioception, written by prepare_states_h5.py in manifest order so
        # row i matches row i of the features and actions.
        self.states_a = self.states_b = None
        if use_states:
            state_path = cache_dir / f'{split}_states_a.npy'
            if not state_path.exists():
                raise FileNotFoundError(
                    f'Missing {state_path}. Run prepare_states_h5.py first.'
                )
            self.states_a = np.load(state_path)
            self.states_b = np.load(cache_dir / f'{split}_states_b.npy')
            assert len(self.states_a) == len(self.features_a), \
                f'State/feature length mismatch: {len(self.states_a)} vs {len(self.features_a)}'

        # Actions
        self.actions_a = np.load(cache_dir / f'{split}_actions_a.npy')
        self.actions_b = np.load(cache_dir / f'{split}_actions_b.npy')

        assert len(self.features_a) == len(self.actions_a), \
            f'Feature/action length mismatch: {len(self.features_a)} vs {len(self.actions_a)}'

        # Episode boundaries
        lens_path = cache_dir / f'{split}_episode_lens.npy'
        if not lens_path.exists():
            raise FileNotFoundError(
                f'Missing {lens_path}. Run build_episode_lens.py first.'
            )
        episode_lens = np.load(lens_path)
        assert int(episode_lens.sum()) == len(self.actions_a)

        self.valid_starts = _build_valid_start_indices(episode_lens, chunk_size)
        print(
            f'[{split}] {len(self.actions_a)} timesteps, {len(episode_lens)} episodes, '
            f'{len(self.valid_starts)} valid chunk-starts (chunk_size={chunk_size})'
        )

        self._prefetch_queue = None
        self._prefetch_size = prefetch

    def _load_batch(self, start_indices):
        offsets = np.arange(self.chunk_size, dtype=np.int64)
        idx_grid = start_indices[:, None] + offsets[None, :]

        batch = {
            'features_a': self.features_a[start_indices],
            'features_b': self.features_b[start_indices],
            'action_a': self.actions_a[idx_grid],
            'action_b': self.actions_b[idx_grid],
        }
        if self.use_states:
            # State at the query timestep only, matching the features: the
            # policy conditions on where the arm is now, then predicts the whole
            # chunk from there.
            batch['state_a'] = self.states_a[start_indices]
            batch['state_b'] = self.states_b[start_indices]
        return batch

    def get_batch(self, batch_size: int) -> Dict:
        chosen = np.random.choice(len(self.valid_starts), batch_size, replace=False)
        return self._load_batch(self.valid_starts[chosen])

    def start_prefetch(self, batch_size: int, n_batches: int):
        self._prefetch_queue = Queue(maxsize=self._prefetch_size)

        def _worker():
            for _ in range(n_batches):
                chosen = np.random.choice(len(self.valid_starts), batch_size, replace=False)
                starts = self.valid_starts[chosen]
                starts.sort()
                batch = self._load_batch(starts)
                self._prefetch_queue.put(batch)
            self._prefetch_queue.put(None)

        Thread(target=_worker, daemon=True).start()

    def get_prefetched_batch(self) -> Dict:
        return self._prefetch_queue.get()

    def __len__(self):
        return len(self.valid_starts)


if __name__ == '__main__':
    ds = COLADataset(
        cache_dir='/scratch/users/ntu/ahaskar0/cola-research/cache_handover/',
        feat_dir='/scratch/users/ntu/ahaskar0/cola-research/features_handover/',
        split='train',
    )
    print(f'\nDataset size (valid chunk-starts): {len(ds)}')
    batch = ds.get_batch(32)
    print('\nBatch shapes:')
    for k, v in batch.items():
        print(f'  {k}: {v.shape} {v.dtype}')
    assert batch['features_a'].shape == (32, 768)
    assert batch['action_a'].shape == (32, CHUNK_SIZE, 7)
    print('\nOK')
