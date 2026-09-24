"""
Pre-extract Octo-Base features for the ALOHA handover HDF5 dataset.

Reads split_manifest.json written by prepare_h5_handover.py, streams the
wrist camera images out of each .h5 in manifest order, runs the frozen Octo
backbone once over every timestep and saves the 768-dim vectors:

    {split}_features_a.npy  (N, 768) float32
    {split}_features_b.npy  (N, 768) float32
    {split}_features_o.npy  (N, 768) float32   (overhead, when --overhead)

Row i here lines up with row i of {split}_actions_{a,b}.npy, so the manifest
order must not change between the two scripts.

Agent A sees image_wrist_a and agent B sees image_wrist_b, matching the
per-agent partial observability the COLA message channel is meant to bridge.

--overhead additionally extracts image_overhead into {split}_features_o.npy,
which the model concatenates onto BOTH arms' self-representation. Note this
deliberately weakens the partial-observability setup the message channel exists
to bridge: with a shared global view there is less for the arms to tell each
other. It is here because the wrist cameras move with the arm, so once the
policy drifts off the expert trajectory they show frames present in no
demonstration, whereas a fixed camera keeps reporting where the box is. Treat
it as an ablation arm, not the default configuration.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from cola_architecture import COLAModel, FEATURE_DIM


CACHE_DIR = Path('/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_aloha_handover')
FEAT_DIR = Path('/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/features_aloha_handover')
BATCH_SIZE = 64


def episode_features(model, path, batch_size, overhead=False):
    """Return (features_a, features_b, features_o) for one episode.

    features_o is None unless overhead=True.
    """
    with h5py.File(path, 'r') as f:
        images_a = f['image_wrist_a'][:]
        images_b = f['image_wrist_b'][:]
        images_o = f['image_overhead'][:] if overhead else None

    n = len(images_a)
    feat_a = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    feat_b = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    feat_o = np.zeros((n, FEATURE_DIM), dtype=np.float32) if overhead else None

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        feat_a[i:end] = np.asarray(model.extract_octo_features(images_a[i:end]))
        feat_b[i:end] = np.asarray(model.extract_octo_features(images_b[i:end]))
        if overhead:
            feat_o[i:end] = np.asarray(model.extract_octo_features(images_o[i:end]))

    return feat_a, feat_b, feat_o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--feat_dir', type=Path, default=FEAT_DIR)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--overhead', action='store_true',
                    help='also extract image_overhead into {split}_features_o.npy '
                         '(must match the model\'s use_overhead setting)')
    args = ap.parse_args()

    manifest_path = args.cache_dir / 'split_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Missing {manifest_path}. Run prepare_h5_handover.py first.')
    with open(manifest_path) as f:
        manifest = json.load(f)

    args.feat_dir.mkdir(parents=True, exist_ok=True)

    print('Loading Octo-Base...')
    model = COLAModel()
    print('Octo loaded\n')

    for split, paths in manifest['splits'].items():
        print(f'--- {split} ({len(paths)} episodes) ---')

        feats_a, feats_b, feats_o = [], [], []
        for path in tqdm(paths, desc=split):
            fa, fb, fo = episode_features(model, path, args.batch_size,
                                          overhead=args.overhead)
            feats_a.append(fa)
            feats_b.append(fb)
            if args.overhead:
                feats_o.append(fo)

        feats_a = np.concatenate(feats_a)
        feats_b = np.concatenate(feats_b)
        feats_o = np.concatenate(feats_o) if args.overhead else None

        # The actions were written from the same manifest order, so any length
        # disagreement means the two scripts saw different data.
        n_actions = len(np.load(args.cache_dir / f'{split}_actions_a.npy', mmap_mode='r'))
        assert len(feats_a) == n_actions, \
            f'{split}: {len(feats_a)} features vs {n_actions} actions — re-run prepare_h5_handover.py'

        np.save(args.feat_dir / f'{split}_features_a.npy', feats_a)
        np.save(args.feat_dir / f'{split}_features_b.npy', feats_b)
        if args.overhead:
            np.save(args.feat_dir / f'{split}_features_o.npy', feats_o)
        print(f'  saved: {feats_a.shape}'
              f"{' (+ overhead)' if args.overhead else ''}\n")

    print(f'Feature extraction complete -> {args.feat_dir}')


if __name__ == '__main__':
    main()
