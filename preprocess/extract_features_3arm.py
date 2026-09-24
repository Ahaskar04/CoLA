"""
Pre-extract Octo-Base features for the 3-arm ALOHA handover HDF5 dataset.

Port of extract_features_h5.py. Reads split_manifest.json written by
prepare_h5_3arm.py, streams the wrist camera images out of each .h5 in manifest
order, runs the frozen Octo backbone once over every timestep and saves the
768-dim vectors:

    {split}_features_a.npy  (N, 768) float32   image_wrist_a
    {split}_features_b.npy  (N, 768) float32   image_wrist_b
    {split}_features_c.npy  (N, 768) float32   image_wrist_c
    {split}_features_o.npy  (N, 768) float32   image_overhead, when --overhead

Row i here lines up with row i of {split}_actions_{a,b,c}.npy, so the manifest
order must not change between the two scripts. The assert at the end of each
split is the guard: a length disagreement means the two scripts saw different
data, and training would silently pair arm A's frames with arm B's labels.

Each agent sees only its own wrist camera, matching the per-agent partial
observability the COLA message channel is meant to bridge.

--overhead additionally extracts image_overhead into {split}_features_o.npy,
which the model concatenates onto ALL THREE arms' self-representation. Note this
deliberately weakens the partial-observability setup the message channel exists
to bridge: with a shared global view there is less for the arms to tell each
other. It is here because the wrist cameras move with the arm, so once the
policy drifts off the expert trajectory they show frames present in no
demonstration, whereas a fixed camera keeps reporting where the box is. The
2-arm run that scored 88-91% used it.

COST. This is the expensive step: 4 camera streams x 53,179 timesteps = 212,716
Octo forward passes, against 59,097 for the 2-arm v5 set. Budget roughly 3.6x
the 2-arm extraction wall-clock. It only has to run once per dataset.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from cola_architecture_3arm import COLAModel3Arm, FEATURE_DIM, ARMS


CACHE_DIR = Path('/scratch/users/ntu/ahaskar0/v1/cola-3arm-scratchdata/cache_aloha_handover_3arm')
FEAT_DIR = Path('/scratch/users/ntu/ahaskar0/v1/cola-3arm-scratchdata/features_aloha_handover_3arm')
BATCH_SIZE = 64


def episode_features(model, path, batch_size, overhead=False):
    """Return {'a','b','c'[,'o']} -> (T, 768) features for one episode.

    Images are read one stream at a time rather than all four at once: a 3-arm
    episode carries 4 x 287 x 256 x 256 x 3 bytes = 225 MB of uint8, and holding
    every stream resident triples peak host memory for no gain.
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
        raise SystemExit(f'Missing {manifest_path}. Run prepare_h5_3arm.py first.')
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

        # The actions were written from the same manifest order, so any length
        # disagreement means the two scripts saw different data.
        n_actions = len(np.load(args.cache_dir / f'{split}_actions_a.npy', mmap_mode='r'))
        assert len(feats['a']) == n_actions, \
            f'{split}: {len(feats["a"])} features vs {n_actions} actions — re-run prepare_h5_3arm.py'

        for t in tags:
            np.save(args.feat_dir / f'{split}_features_{t}.npy', feats[t])
        print(f'  saved: {feats["a"].shape} x {len(tags)} streams '
              f'({", ".join(tags)})\n')

    print(f'Feature extraction complete -> {args.feat_dir}')


if __name__ == '__main__':
    main()
