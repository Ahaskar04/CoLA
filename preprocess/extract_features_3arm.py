"""Extract frozen Octo-Base features for the three-arm dataset.

Reads split_manifest.json from prepare_cache_3arm.py and writes, per split:
    {split}_features_{a,b,c}.npy   (N, 768)   each arm's wrist camera
    {split}_features_o.npy         (N, 768)   overhead camera, with --overhead

Rows follow the manifest, so they line up with {split}_actions_*.npy.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cola.model_3arm import COLAModel3Arm, FEATURE_DIM, ARMS  # noqa: E402


CACHE_DIR = REPO / 'data' / 'cache' / 'handover_3arm'
FEAT_DIR = REPO / 'data' / 'features' / 'handover_3arm'
BATCH_SIZE = 64


def episode_features(model, path, batch_size, overhead=False):
    """Return {'a', 'b', 'c'[, 'o']} -> (T, 768) features for one episode.

    Streams are read one at a time to limit host memory.
    """
    out = {}
    streams = [(arm, f'image_wrist_{arm}') for arm in ARMS]
    if overhead:
        streams.append(('o', 'image_overhead'))

    for tag, key in streams:
        with h5py.File(path, 'r') as f:
            images = f[key][:]
        n = len(images)
        feat = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            feat[i:end] = np.asarray(model.extract_octo_features(images[i:end]))
        out[tag] = feat
        del images

    lens = {t: len(v) for t, v in out.items()}
    assert len(set(lens.values())) == 1, \
        f'{Path(path).name}: camera streams disagree in length: {lens}'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--feat_dir', type=Path, default=FEAT_DIR)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--overhead', action='store_true',
                    help='also extract image_overhead into {split}_features_o.npy '
                         "(must match the model's use_overhead setting)")
    args = ap.parse_args()

    manifest_path = args.cache_dir / 'split_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Missing {manifest_path}. Run prepare_cache_3arm.py first.')
    with open(manifest_path) as f:
        manifest = json.load(f)

    args.feat_dir.mkdir(parents=True, exist_ok=True)

    print('Loading Octo-Base...')
    model = COLAModel3Arm()
    print('Octo loaded\n')

    tags = list(ARMS) + (['o'] if args.overhead else [])

    for split, paths in manifest['splits'].items():
        print(f'--- {split} ({len(paths)} episodes) ---')

        acc = {t: [] for t in tags}
        for path in tqdm(paths, desc=split):
            ep = episode_features(model, path, args.batch_size,
                                  overhead=args.overhead)
            for t in tags:
                acc[t].append(ep[t])

        feats = {t: np.concatenate(acc[t]) for t in tags}

        # Features and actions come from the same manifest, so lengths must match.
        n_actions = len(np.load(args.cache_dir / f'{split}_actions_a.npy', mmap_mode='r'))
        assert len(feats['a']) == n_actions, \
            f'{split}: {len(feats["a"])} features vs {n_actions} actions — re-run prepare_cache_3arm.py'

        for t in tags:
            np.save(args.feat_dir / f'{split}_features_{t}.npy', feats[t])
        print(f'  saved: {feats["a"].shape} x {len(tags)} streams '
              f'({", ".join(tags)})\n')

    print(f'Feature extraction complete -> {args.feat_dir}')


if __name__ == '__main__':
    main()
