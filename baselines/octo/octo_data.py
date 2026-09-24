"""Octo-format batches from the ALOHA handover HDF5 demos, without TFDS.

Octo's own finetuning example feeds `make_single_dataset`, which wants the data
in RLDS. Our demos are HDF5. Rather than author a TFDS builder and re-materialise
118 episodes, this module produces the same batch structure directly. The schema
is copied from the checkpoint's own example_batch.msgpack and from
octo/data/traj_transforms.py, not invented:

    observation/image_primary                (B, window, 256, 256, 3) uint8
    observation/proprio                      (B, window, D_p)         float32
    observation/timestep_pad_mask            (B, window)              bool
    observation/task_completed               (B, window, H)           bool
    observation/pad_mask_dict/{image_primary,proprio}                 bool
    task/language_instruction/{input_ids,attention_mask}
    task/pad_mask_dict/language_instruction  (B,)                     bool
    action                                   (B, window, H, D_a)      float32
    action_pad_mask                          (B, window, H, D_a)      bool

Three configurations, sharing one code path:

    'a'     arm A alone: its own wrist camera, its own 7-d proprio and action.
    'b'     arm B alone: likewise.
    'both'  the centralised reference recipe: overhead camera, 14-d proprio and
            14-d action covering both arms.

'a' and 'b' are the decentralised baseline. Each sees ONLY its own wrist camera,
matching the partial observability CoLA's message channel exists to bridge, and
neither can see the other's observation or action. That is the whole point: it
isolates whether coordination survives without a communication channel.

Splits come from CoLA's split_manifest.json so the baseline trains and evaluates
on exactly the same episodes as CoLA. Statistics are computed on the train split
only.
"""

import json
from pathlib import Path

import h5py
import numpy as np

MANIFEST = ('/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/'
            'cache_aloha_handover/split_manifest.json')

# Which HDF5 keys each configuration reads. Concatenation order for 'both' is
# (a, b) and must stay fixed -- the eval has to split the 14-d output the same way.
ARM_SPEC = {
    'a':    dict(image='image_wrist_a', proprio=['state_a'],
                 action=['action_a']),
    'b':    dict(image='image_wrist_b', proprio=['state_b'],
                 action=['action_b']),
    'both': dict(image='image_overhead', proprio=['state_a', 'state_b'],
                 action=['action_a', 'action_b']),
    # Third arm, for the 3-arm A->B->C chain (cache_3arm_honly_v1). Decentralised
    # like 'a' and 'b': its own proprio and action only.
    'c':    dict(image='image_wrist_c', proprio=['state_c'],
                 action=['action_c']),
}

# Octo is language-conditioned and has no partner channel, so for the
# decentralised runs this string is the ONLY thing distinguishing the two arms'
# jobs. Overridable from the CLI.
# HDF5 image key -> the MuJoCo camera that produced it. Evaluation must render
# the SAME view the policy trained on; this mapping is what keeps them in step.
IMAGE_KEY_TO_CAMERA = {
    'image_overhead': 'overhead_cam',
    'image_wrist_a': 'wrist_cam_left',
    'image_wrist_b': 'wrist_cam_right',
    'image_wrist_c': 'wrist_cam_third',
}

DEFAULT_INSTRUCTION = {
    'a': 'pick up the red box and hand it to the other arm',
    'b': 'take the red box from the other arm',
    'both': 'pick up the red box with one arm and hand it to the other',
    'c': 'take the red box from the other arm',
}


def _stat_block(x):
    """Statistics in the checkpoint's own dataset_statistics.json layout."""
    return {
        'mean': x.mean(0).astype(np.float32),
        'std': x.std(0).astype(np.float32),
        'min': x.min(0).astype(np.float32),
        'max': x.max(0).astype(np.float32),
        'p01': np.percentile(x, 1, axis=0).astype(np.float32),
        'p99': np.percentile(x, 99, axis=0).astype(np.float32),
    }


class H5Split:
    """One split, held in RAM, with episode boundaries preserved.

    Images dominate: 94 episodes x ~156 steps x 256x256x3 is ~2.9 GB for one
    camera. That fits comfortably and makes batch assembly a pure gather, so
    the GPU never waits on HDF5.
    """

    def __init__(self, manifest_path, split, arm, verbose=True, image_key=None,
                 wrist_key=None):
        spec = dict(ARM_SPEC[arm])
        if image_key:
            # Decouples the camera from the arm: e.g. arm A's actions and
            # proprio paired with the third-person overhead view, which is the
            # single-third-person-camera configuration the Octo paper prefers.
            spec['image'] = image_key
        paths = json.load(open(manifest_path))['splits'][split]

        images, wrists, proprios, actions, lens = [], [], [], [], []
        for i, p in enumerate(paths):
            with h5py.File(p, 'r') as f:
                images.append(f[spec['image']][:])
                if wrist_key:
                    wrists.append(f[wrist_key][:])
                proprios.append(np.concatenate(
                    [f[k][:] for k in spec['proprio']], axis=-1))
                actions.append(np.concatenate(
                    [f[k][:] for k in spec['action']], axis=-1))
            lens.append(len(images[-1]))
            if verbose and (i + 1) % 25 == 0:
                print(f'   loaded {i + 1}/{len(paths)} episodes', flush=True)

        self.images = np.concatenate(images, axis=0)
        # Second camera stream, fed to octo's own pretrained `wrist` tokenizer
        # alongside image_primary. This is the paper's two-camera setup; None
        # means the single-camera configuration.
        self.images_wrist = (np.concatenate(wrists, axis=0) if wrist_key
                             else None)
        self.wrist_key = wrist_key
        self.proprio = np.concatenate(proprios, axis=0).astype(np.float32)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32)
        self.ep_len = np.asarray(lens, dtype=np.int64)
        self.ep_start = np.concatenate([[0], np.cumsum(self.ep_len)[:-1]])

        # Episode index and within-episode timestep for every row, so chunking
        # can clamp at the episode boundary instead of bleeding into the next.
        self.ep_of = np.repeat(np.arange(len(lens)), self.ep_len)
        self.t_of = np.concatenate([np.arange(n) for n in self.ep_len])

        self.image_key = spec['image']
        self.n = len(self.images)
        self.action_dim = self.actions.shape[-1]
        self.proprio_dim = self.proprio.shape[-1]
        if verbose:
            print(f'   {split}: {len(paths)} episodes, {self.n} transitions, '
                  f'action_dim {self.action_dim}, proprio_dim {self.proprio_dim}, '
                  f'image {self.image_key}')

    def statistics(self):
        """Octo dataset_statistics for this split (compute on train only)."""
        return {
            'action': _stat_block(self.actions),
            'proprio': _stat_block(self.proprio),
            'num_transitions': int(self.n),
            'num_trajectories': int(len(self.ep_len)),
        }


def make_batch(data, idx, stats, tokens, window_size, action_horizon,
               with_proprio=True):
    """Assemble one Octo batch for the given global row indices.

    Reproduces octo/data/traj_transforms.chunk_act_obs exactly, including its
    task_completed convention -- actions at or past the goal timestep are marked
    as padding, and the goal is the final timestep since we are language-only.
    """
    B, H, W = len(idx), action_horizon, window_size
    ep, t = data.ep_of[idx], data.t_of[idx]
    start, length = data.ep_start[ep], data.ep_len[ep]

    # Observation history: t-W+1 .. t, repeating the first frame rather than
    # running off the front of the episode.
    hist = t[:, None] + np.arange(-W + 1, 1)[None, :]        # (B, W)
    timestep_pad_mask = hist >= 0
    obs_rows = start[:, None] + np.maximum(hist, 0)

    # Action chunk: t .. t+H-1, repeating the last action at the episode end.
    chunk = np.minimum(t[:, None] + np.arange(H)[None, :],
                       length[:, None] - 1)                  # (B, H)
    act_rows = start[:, None] + chunk

    a_mean, a_std = stats['action']['mean'], stats['action']['std']
    p_mean, p_std = stats['proprio']['mean'], stats['proprio']['std']

    action = (data.actions[act_rows] - a_mean) / (a_std + 1e-8)
    action = np.repeat(action[:, None], W, axis=1)           # (B, W, H, D)

    proprio = (data.proprio[obs_rows] - p_mean) / (p_std + 1e-8)

    # task_completed, verbatim from chunk_act_obs so the masking matches the
    # reference rather than a reimplementation of it.
    goal = (length - 1)[:, None, None]
    tt, ww, hh = np.meshgrid(np.arange(1), np.arange(W), np.arange(H),
                             indexing='ij')
    rel = goal - (t[:, None, None] - (W + 1) + ww + hh)
    task_completed = rel <= 0                                # (B, W, H)

    action_pad_mask = np.broadcast_to(
        ~task_completed[..., None], action.shape).copy()

    true_ow = np.ones((B, W), dtype=bool)
    observation = {
        'image_primary': data.images[obs_rows],
        'timestep_pad_mask': timestep_pad_mask,
        'task_completed': task_completed,
        'pad_mask_dict': {'image_primary': true_ow},
    }
    if getattr(data, 'images_wrist', None) is not None:
        observation['image_wrist'] = data.images_wrist[obs_rows]
        observation['pad_mask_dict']['image_wrist'] = true_ow
    if with_proprio:
        observation['proprio'] = proprio.astype(np.float32)
        observation['pad_mask_dict']['proprio'] = true_ow
    return {
        'observation': observation,
        'task': {
            'language_instruction': {
                k: np.repeat(v, B, axis=0) for k, v in tokens.items()
            },
            'pad_mask_dict': {
                'language_instruction': np.ones((B,), dtype=bool),
            },
        },
        'action': action.astype(np.float32),
        'action_pad_mask': action_pad_mask,
    }


def batch_iterator(data, stats, tokens, batch_size, window_size,
                   action_horizon, seed=0, shuffle=True, with_proprio=True,
                   skip=0):
    """Infinite stream of batches.

    skip discards that many draws first, so a resumed run is fed exactly the
    batches an unbroken run would have seen from that point.
    """
    rng = np.random.default_rng(seed)
    if shuffle:
        for _ in range(skip):
            rng.integers(0, data.n, size=batch_size)
    while True:
        idx = (rng.integers(0, data.n, size=batch_size) if shuffle
               else np.arange(batch_size) % data.n)
        yield make_batch(data, idx, stats, tokens, window_size, action_horizon,
                         with_proprio)
