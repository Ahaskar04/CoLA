"""Train the three-arm CoLA adapters on cached Octo features.

Same objective and device-resident pipeline as train_2arm.py,
looped over ARMS = ('a', 'b', 'c') with all-to-all messages (see
cola/model_3arm.py). The EE metric measures one arm (--ee_arm).

Expects, per split and arm x in {a, b, c}:
    {feat_dir}/{split}_features_{x}.npy   (N, 768)
    {cache_dir}/{split}_actions_{x}.npy   (N, 7)
    {cache_dir}/{split}_states_{x}.npy    (N, 7), unless --no_proprio

Usage:
    python train_3arm.py --cache_dir CACHE --feat_dir FEATS --run_dir RUN \\
        --overhead --diffusion --diffusion_unet \\
        --lambda_gripper 0.25 --grip_transition_weight 0 --num_epochs 150
"""

import argparse
import functools
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cola.model_3arm import (COLAModel3Arm, ARMS, N_ARMS, GRIPPER_IDX,  # noqa: E402
                             CHUNK_SIZE, ACTION_DIM, DIFFUSION_STEPS)
from cola.dataset_3arm import COLADataset3Arm  # noqa: E402

CACHE_DIR = str(REPO / 'data' / 'cache' / 'handover_3arm')
FEAT_DIR = str(REPO / 'data' / 'features' / 'handover_3arm')
SCENE_XML = str(REPO / 'envs' / 'handover_3arm' / 'scene.xml')
RUN_DIR = REPO / 'runs' / 'handover_3arm'

# Per-joint loss weights, as in the two-arm trainer (same ViperX arms): gripper
# displacement per 1 std of each joint's action, normalised to mean 1. The
# wrist-rotation joints are floored at 0.25 because they still align the jaws.
JOINT_EE_WEIGHTS = np.array([0.34, 1.86, 2.79, 0.25, 0.93, 0.25], dtype=np.float32)
JOINT_EE_WEIGHTS = JOINT_EE_WEIGHTS / JOINT_EE_WEIGHTS.mean()

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Arm letter -> body prefix in envs/handover_3arm/aloha.xml
# (A picks up, B turns 180 deg, C receives).
ARM_PREFIX = {'a': 'left', 'b': 'right', 'c': 'third'}

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_gpu_info():
    """Return GPU memory usage string, or None if unavailable."""
    if not HAS_GPUTIL:
        return None
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return f"GPU {gpu.id}: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB ({gpu.memoryUtil*100:.0f}%)"
    except Exception:
        pass
    return None


def setup_logging(log_dir, timestamp):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f'training_log_{timestamp}.txt'
    logger = logging.getLogger('cola3arm')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger, log_path


def plot_loss_curves(train_losses, val_losses, best_epoch, save_path):
    if not HAS_MATPLOTLIB:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label='train')
    ax.plot(epochs, val_losses, label='val')
    if best_epoch is not None:
        ax.axvline(best_epoch, ls='--', c='k', alpha=0.4,
                   label=f'best (epoch {best_epoch})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('COLA 3-arm training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# Device-resident dataset

class DeviceDataset3Arm:
    """Whole split held in device memory; batches are on-device gathers.

    COLADataset3Arm is used only for .valid_starts (chunks never straddle episodes).
    """

    def __init__(self, cache_dir, feat_dir, split, use_proprio,
                 chunk_size=CHUNK_SIZE, logger=None, use_overhead=False):
        self.split = split
        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.chunk_size = chunk_size

        base = COLADataset3Arm(cache_dir, feat_dir, split=split,
                               use_states=use_proprio, use_overhead=use_overhead)
        starts_np = np.sort(np.asarray(base.valid_starts, dtype=np.int32))
        del base   # release its host-side copy before we upload ours

        cache_dir, feat_dir = Path(cache_dir), Path(feat_dir)
        feat = {a: np.load(feat_dir / f'{split}_features_{a}.npy') for a in ARMS}
        # Shared overhead view (from extract_features_3arm.py --overhead).
        feat_o = None
        if use_overhead:
            fo = feat_dir / f'{split}_features_o.npy'
            if not fo.exists():
                raise FileNotFoundError(
                    f'{fo} not found. --overhead needs the overhead features: '
                    f're-run extract_features_3arm.py --overhead after deleting '
                    f'the feature dir (it skips when files already exist).')
            feat_o = np.load(fo)
        act = {a: np.load(cache_dir / f'{split}_actions_{a}.npy') for a in ARMS}

        n = len(feat['a'])
        for a in ARMS:
            assert len(feat[a]) == n, \
                f'{split}: features_a has {n} rows, features_{a} has {len(feat[a])}'
            assert len(act[a]) == n, \
                f'{split}: features_a has {n} rows, actions_{a} has {len(act[a])}'
            assert act[a].shape[1] == ACTION_DIM, \
                f'{split}: actions_{a} is {act[a].shape}, expected (N, {ACTION_DIM})'
        if use_overhead:
            assert len(feat_o) == n, \
                f'{split}: features_a has {n} rows, features_o has {len(feat_o)}'
        assert starts_np.max() + chunk_size <= n, \
            f'{split}: chunk start {starts_np.max()} + {chunk_size} overruns {n} rows'

        # A gap in the valid starts marks an episode boundary. prev gives the
        # gripper-flip mask one frame of left context within the episode.
        is_ep_start = np.concatenate([[True], np.diff(starts_np) != 1])
        prev_np = np.where(is_ep_start, starts_np, starts_np - 1).astype(np.int32)
        self.n_episodes = int(is_ep_start.sum())

        self.starts = jnp.asarray(starts_np)
        self.prev = jnp.asarray(prev_np)
        self.offsets = jnp.arange(chunk_size, dtype=jnp.int32)
        self.feat = {a: jnp.asarray(feat[a]) for a in ARMS}
        self.feat_o = jnp.asarray(feat_o) if use_overhead else None
        self.act = {a: jnp.asarray(act[a]) for a in ARMS}

        if use_proprio:
            state = {}
            for a in ARMS:
                p = cache_dir / f'{split}_states_{a}.npy'
                if not p.exists():
                    raise SystemExit(
                        f'use_proprio=True but {p} is missing.\n'
                        f'  present: {sorted(q.name for q in cache_dir.glob(f"{split}_*.npy"))}\n'
                        f'Run prepare_cache_3arm.py, or pass --no_proprio.')
                state[a] = np.load(p)
                assert len(state[a]) == n, \
                    f'{split}: states_{a} has {len(state[a])} rows, features have {n}'
            self.state = {a: jnp.asarray(state[a]) for a in ARMS}
        else:
            self.state = None

        arrays = list(feat.values()) + list(act.values())
        if use_overhead:
            arrays.append(feat_o)
        mb = sum(x.size * x.itemsize for x in arrays) / 1e6
        if logger:
            logger.info(f'  [{split}] {n} timesteps, {self.n_episodes} episodes, '
                        f'{len(starts_np)} chunk-starts, {mb:.0f} MB on device')

    def __len__(self):
        return int(self.starts.shape[0])

    def batch(self, sel):
        """Gather one batch. `sel` indexes into valid_starts, not into timesteps."""
        s = self.starts[sel]
        idx = s[:, None] + self.offsets[None, :]        # (B, chunk)
        p = self.prev[sel]

        out = {
            'features': {a: self.feat[a][s] for a in ARMS},
            'action': {a: self.act[a][idx] for a in ARMS},
            'prev_grip': {a: (self.act[a][p, GRIPPER_IDX] > 0).astype(jnp.float32)
                          for a in ARMS},
        }
        if self.use_overhead:
            out['features_o'] = self.feat_o[s]
        if self.use_proprio:
            out['state'] = {a: self.state[a][s] for a in ARMS}
        return out

    def epoch_batches(self, batch_size, rng=None, drop_last=True):
        """Yield batches. Shuffled when rng is given, in order otherwise."""
        m = len(self)
        order = jax.random.permutation(rng, m) if rng is not None else jnp.arange(m)
        stop = (m // batch_size) * batch_size if drop_last else m
        for i in range(0, stop, batch_size):
            yield self.batch(order[i:i + batch_size])


# Loss

def gripper_transition_weights(target_chunk, extra_weight, prev_label=None):
    """Per-frame gripper loss weights, raised near open/close transitions.

    Frames within +/-1 step of a flip get 1 + extra_weight. prev_label
    (batch,) is the gripper label just before the chunk, so a flip on the
    chunk's first element is still caught.
    """
    g = (target_chunk[..., GRIPPER_IDX] > 0).astype(jnp.float32)   # (B, chunk)
    if prev_label is None:
        d = jnp.abs(jnp.diff(g, axis=1))                            # (B, chunk-1)
        flip = jnp.concatenate([jnp.zeros_like(g[:, :1]), d], axis=1)
    else:
        prev = prev_label[:, None]
        d = jnp.abs(jnp.diff(jnp.concatenate([prev, g], axis=1), axis=1))
        flip = d                                                    # (B, chunk)

    # Widen by one step either side, so timing is supervised, not just the frame.
    pad = jnp.pad(flip, ((0, 0), (1, 1)))
    widened = jnp.maximum(jnp.maximum(pad[:, :-2], pad[:, 1:-1]), pad[:, 2:])
    return 1.0 + extra_weight * widened


def make_loss_fn(model, cfg):
    """Build compute_loss closed over the static config."""
    w_joint = jnp.asarray(JOINT_EE_WEIGHTS) if cfg['ee_weighted'] else jnp.ones(GRIPPER_IDX)
    lam = cfg['lambda_gripper']
    trans_w = cfg['grip_transition_weight']
    use_proprio = cfg['use_proprio']
    joint_l1 = cfg['joint_l1']
    grip_prev = cfg['grip_prev_context']
    p_drop = cfg['proprio_dropout']
    use_messages = cfg.get('use_messages', True)

    def _diffusion_loss(model, params, cond, batch, rng, lam, trans_w, grip_prev):
        """DDPM training objective on the joint columns + BCE on the gripper."""
        ab = model.alpha_bars
        total = 0.0
        parts = {}
        # Validation passes rng=None; a fixed key keeps val loss comparable.
        base = rng if rng is not None else jax.random.PRNGKey(0)

        for tag in ARMS:
            target = jnp.asarray(batch['action'][tag])
            joints = target[..., :GRIPPER_IDX]
            B = joints.shape[0]

            k_t, k_n, base = jax.random.split(base, 3)
            t = jax.random.randint(k_t, (B,), 1, DIFFUSION_STEPS)
            noise = jax.random.normal(k_n, joints.shape)
            a_t = ab[t][:, None, None]
            noisy = jnp.sqrt(a_t) * joints + jnp.sqrt(1.0 - a_t) * noise

            eps_pred, grip_logit = model.denoise(cond[tag], noisy, t, params, tag)

            # Standard DDPM noise-prediction loss.
            joint_loss = jnp.mean((eps_pred - noise) ** 2)

            label = (target[..., GRIPPER_IDX] > 0).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(grip_logit[..., 0], label)
            prev = batch['prev_grip'][tag] if grip_prev else None
            gw = gripper_transition_weights(target, trans_w, prev)
            grip_loss = (bce * gw).sum() / gw.sum()

            total = total + joint_loss + lam * grip_loss
            parts[f'joint_{tag}'] = joint_loss
            parts[f'grip_{tag}'] = grip_loss
            correct = ((grip_logit[..., 0] > 0) == (label > 0.5)).astype(jnp.float32)
            flip_mask = gw > 1.0
            parts[f'grip_acc_{tag}'] = correct.mean()
            parts[f'grip_acc_flip_{tag}'] = (
                (correct * flip_mask).sum() / jnp.maximum(flip_mask.sum(), 1.0))
        return total, parts

    def compute_loss(params, batch, rng=None):
        state = {a: batch['state'][a] for a in ARMS} if use_proprio else {}

        # Proprio dropout (training only, per arm), so the policy can't ignore vision.
        if use_proprio and rng is not None and p_drop > 0:
            keys = jax.random.split(rng, N_ARMS)
            for i, a in enumerate(ARMS):
                keep = (jax.random.uniform(keys[i], (state[a].shape[0], 1)) >= p_drop)
                state[a] = state[a] * keep

        out = model.forward_from_features(
            batch['features'], params=params,
            proprio=state if use_proprio else None,
            features_o=batch.get('features_o'),
            # False zeros all six message channels; the parameter count is
            # unchanged (no-message ablation).
            use_messages=use_messages,
        )

        if model.use_diffusion:
            # Diffusion: forward_from_features returned conditioning vectors.
            # Train the denoiser on the joints; the gripper keeps its BCE logit.
            return _diffusion_loss(model, params, out, batch, rng,
                                   lam, trans_w, grip_prev)

        total = 0.0
        parts = {}
        for tag in ARMS:
            pred = out[tag]
            target = jnp.asarray(batch['action'][tag])

            # Joints: weighted by how much each moves the gripper
            err = pred[..., :GRIPPER_IDX] - target[..., :GRIPPER_IDX]
            err = jnp.abs(err) if joint_l1 else err ** 2
            joint_loss = (err * w_joint).mean()

            # Gripper: binary, so BCE on a raw logit
            label = (target[..., GRIPPER_IDX] > 0).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(pred[..., GRIPPER_IDX], label)
            prev = batch['prev_grip'][tag] if grip_prev else None
            gw = gripper_transition_weights(target, trans_w, prev)
            grip_loss = (bce * gw).sum() / gw.sum()

            total = total + joint_loss + lam * grip_loss
            parts[f'joint_{tag}'] = joint_loss
            parts[f'grip_{tag}'] = grip_loss
            # Accuracy overall and near gripper transitions.
            correct = ((pred[..., GRIPPER_IDX] > 0) == (label > 0.5)).astype(jnp.float32)
            flip_mask = gw > 1.0
            parts[f'grip_acc_{tag}'] = correct.mean()
            parts[f'grip_acc_flip_{tag}'] = (
                (correct * flip_mask).sum() / jnp.maximum(flip_mask.sum(), 1.0)
            )

        return total, parts

    return compute_loss


# Compiled train step

def make_train_step(loss_fn, optimizer):
    """Jitted train step: forward, backward and optimizer update.

    params and opt_state are donated, so the inputs are invalid after the call.
    """
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, batch, rng):
        rng, step_rng = jax.random.split(rng)
        (loss, parts), grads = grad_fn(params, batch, step_rng)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, loss, parts

    return train_step


# End-effector error, in millimetres

class EEMetric:
    """Median gripper-position error (mm) of one arm's predicted joint targets.

    Forward kinematics on denormalised actions, at chunk steps 0 and N-1
    (every step with all_steps=True). Measures arm A by default.
    """

    def __init__(self, cache_dir, scene_xml, arm='a', n_samples=256, all_steps=False):
        import mujoco
        self.mujoco = mujoco
        self.arm = arm
        self.n_samples = n_samples
        self.all_steps = all_steps
        with open(Path(cache_dir) / 'action_stats.json') as f:
            self.stats = json.load(f)
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        prefix = ARM_PREFIX[arm]
        self.qadr = [self.model.joint(f'{prefix}/{n}').qposadr[0] for n in ARM_JOINTS]
        self.site = f'{prefix}/gripper'
        self.key = self.model.key('neutral_pose').id
        # Velocity caches store per-step deltas; see _ee.
        self.velocity = bool(self.stats.get('velocity', False))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        self.home_q = self.data.qpos[self.qadr].copy()

    def _denorm(self, a):
        m = self.stats['margin']
        key = f'action_{self.arm}'
        lo = np.array(self.stats[key]['min'], np.float32)
        span = np.array(self.stats[key]['max'], np.float32) - lo
        return (a + m) / (2 * m) * span + lo

    def _ee(self, q):
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        # Deltas are applied from the home pose, the one reference both share.
        self.data.qpos[self.qadr] = (self.home_q + q) if self.velocity else q
        self.mujoco.mj_forward(self.model, self.data)
        return self.data.site(self.site).xpos.copy()

    def __call__(self, pred_arm, true_arm):
        pred = self._denorm(np.asarray(pred_arm))
        true = self._denorm(np.asarray(true_arm))
        n = min(self.n_samples, len(pred))
        idx = np.random.default_rng(0).choice(len(pred), n, replace=False)
        chunk_len = pred.shape[1]
        steps = range(chunk_len) if self.all_steps else (0, chunk_len - 1)

        errs, first, last = [], [], []
        for i in idx:
            for k in steps:
                e = np.linalg.norm(self._ee(pred[i, k, :GRIPPER_IDX])
                                   - self._ee(true[i, k, :GRIPPER_IDX]))
                errs.append(e)
                if k == 0:
                    first.append(e)
                elif k == chunk_len - 1:
                    last.append(e)

        return {
            'ee_mm_median': float(np.median(errs) * 1000.0),
            'ee_mm_p90': float(np.percentile(errs, 90) * 1000.0),
            'ee_mm_step0': float(np.median(first) * 1000.0),
            'ee_mm_stepN': float(np.median(last) * 1000.0),
        }


# Train / validate

def train_epoch(train_step, params, opt_state, dataset, rng, batch_size,
                epoch_num, num_epochs, progress_file=None):
    n_batches = len(dataset) // batch_size
    rng, shuffle_rng = jax.random.split(rng)
    sums = None
    t0 = time.time()

    batches = dataset.epoch_batches(batch_size, rng=shuffle_rng)
    for batch_idx, batch in enumerate(
            tqdm(batches, total=n_batches, desc=f'Epoch {epoch_num+1}/{num_epochs}',
                 leave=False)):
        params, opt_state, rng, loss, parts = train_step(params, opt_state, batch, rng)

        # Device-side accumulation: no host sync until the epoch ends.
        acc = {'loss': loss, **parts}
        sums = acc if sums is None else jax.tree_util.tree_map(jnp.add, sums, acc)

        # Progress file, updated every 100 steps.
        if progress_file is not None and batch_idx % 100 == 0:
            with open(progress_file, 'w') as pf:
                pf.write(f'Epoch {epoch_num+1}/{num_epochs} | Batch {batch_idx}/{n_batches} '
                         f'| Loss: {float(loss):.6f} | Time: {time.time()}\n')

    sums = jax.device_get(sums)
    out = {k: float(v) / max(n_batches, 1) for k, v in sums.items()}
    out['it_per_s'] = n_batches / max(time.time() - t0, 1e-9)
    return out, params, opt_state, rng


def validate(loss_fn_jit, params, dataset, batch_size, model=None, ee_metric=None,
             use_proprio=False, use_messages=True):
    """Deterministic pass over every valid chunk start, weighted by batch size."""
    sums, total_n = None, 0
    pred_acc, true_acc = [], []
    ee_arm = ee_metric.arm if ee_metric is not None else 'a'

    for batch in dataset.epoch_batches(batch_size, rng=None, drop_last=False):
        n = batch['features']['a'].shape[0]
        loss, parts = loss_fn_jit(params, batch, None)

        acc = jax.tree_util.tree_map(lambda x: x * n, {'loss': loss, **parts})
        sums = acc if sums is None else jax.tree_util.tree_map(jnp.add, sums, acc)
        total_n += n

        if ee_metric is not None and len(pred_acc) < 4:
            out = model.forward_from_features(
                batch['features'], params=params,
                proprio=batch.get('state') if use_proprio else None,
                features_o=batch.get('features_o'),
                # Same setting as the loss.
                use_messages=use_messages)
            if model.use_diffusion:
                # Diffusion returns conditioning: sample a chunk (fixed key).
                pa = model.sample_actions(out[ee_arm], params, ee_arm,
                                          jax.random.PRNGKey(0))
            else:
                pa = out[ee_arm]
            pred_acc.append(np.asarray(pa))
            true_acc.append(np.asarray(batch['action'][ee_arm]))

    sums = jax.device_get(sums)
    out = {k: float(v) / max(total_n, 1) for k, v in sums.items()}
    if ee_metric is not None and pred_acc:
        out.update(ee_metric(np.concatenate(pred_acc), np.concatenate(true_acc)))
    return out


def train_cola(
    cache_dir=CACHE_DIR,
    feat_dir=FEAT_DIR,
    num_epochs=150,
    learning_rate=1e-4,
    batch_size=128,
    checkpoint_dir=str(RUN_DIR / 'checkpoints'),
    log_dir=str(RUN_DIR / 'logs'),
    use_proprio=True,
    use_overhead=False,
    use_wrist=True,
    proprio_dropout=0.1,
    lambda_gripper=1.0,
    grip_transition_weight=4.0,
    grip_prev_context=True,
    ee_weighted=True,
    joint_l1=True,
    use_diffusion=False,
    diffusion_unet=False,
    use_messages=True,
    scene_xml=SCENE_XML,
    ee_metric_on=True,
    ee_all_steps=False,
    ee_arm='a',
    seed=0,
    patience=25,
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, log_path = setup_logging(log_dir, timestamp)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    cfg = {
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'seed': seed,
        'use_proprio': use_proprio,
        'use_overhead': use_overhead,
        'use_wrist': use_wrist,
        'proprio_dropout': proprio_dropout,
        'lambda_gripper': lambda_gripper,
        'grip_transition_weight': grip_transition_weight,
        'grip_prev_context': grip_prev_context,
        'ee_weighted': ee_weighted,
        'joint_l1': joint_l1,
        'use_diffusion': use_diffusion,
        'diffusion_unet': diffusion_unet,
        'use_messages': use_messages,
        'arms': list(ARMS),
        'topology': 'all_to_all',
        # Data provenance (which action_stats.json to denormalise with).
        'cache_dir': str(cache_dir),
        'feat_dir': str(feat_dir),
    }

    logger.info('=' * 70)
    logger.info('COLA 3-ARM TRAINING')
    logger.info('=' * 70)
    for k, v in cfg.items():
        logger.info(f'  {k}: {v}')
    gpu = get_gpu_info()
    if gpu:
        logger.info(f'  {gpu}')

    logger.info('\nLoading datasets...')
    splits = {}
    for split in ('train', 'val', 'test'):
        splits[split] = DeviceDataset3Arm(
            cache_dir, feat_dir, split, use_proprio,
            logger=logger, use_overhead=use_overhead)
    train_ds, val_ds, test_ds = splits['train'], splits['val'], splits['test']

    logger.info('\nBuilding model...')
    model = COLAModel3Arm(
        use_proprio=use_proprio,
        use_overhead=use_overhead,
        use_wrist=use_wrist,
        split_gripper=True,
        use_diffusion=use_diffusion,
        diffusion_unet=diffusion_unet,
    )
    params = model.params

    # Cosine decay with a short warmup.
    steps_per_epoch = max(len(train_ds) // batch_size, 1)
    total_steps = steps_per_epoch * num_epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.1,
        peak_value=learning_rate,
        warmup_steps=min(500, total_steps // 10),
        decay_steps=total_steps,
        end_value=learning_rate * 0.05,
    )
    # Plain Adam: adamw's weight decay needs params in optimizer.update(),
    # which train_step does not pass.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(model, cfg)
    train_step = make_train_step(loss_fn, optimizer)
    loss_fn_jit = jax.jit(loss_fn)

    ee_metric = None
    if ee_metric_on:
        try:
            ee_metric = EEMetric(cache_dir, scene_xml, arm=ee_arm,
                                 all_steps=ee_all_steps)
            logger.info(f'  EE metric: arm {ee_arm} ({ARM_PREFIX[ee_arm]}/gripper)')
        except Exception as e:
            logger.info(f'  EE metric unavailable ({e}); continuing without it')

    # Gripper class prior: the baseline for gripper accuracy.
    grip_train = np.asarray(jax.device_get(train_ds.act['a'][:, GRIPPER_IDX]))
    prior = float((grip_train > 0).mean())
    logger.info(f'\n  gripper prior (arm a, fraction open): {prior:.4f}')

    rng = jax.random.PRNGKey(seed)
    best_val, best_epoch, since_best = float('inf'), None, 0
    train_losses, val_losses, history = [], [], []
    progress_file = Path(log_dir) / 'progress.txt'

    logger.info(f'\nTraining for {num_epochs} epochs '
                f'({steps_per_epoch} steps/epoch)...\n')

    for epoch in range(num_epochs):
        tr, params, opt_state, rng = train_epoch(
            train_step, params, opt_state, train_ds, rng, batch_size,
            epoch, num_epochs, progress_file)

        va = validate(loss_fn_jit, params, val_ds, batch_size,
                      model=model, ee_metric=ee_metric, use_proprio=use_proprio,
                      use_messages=use_messages)

        train_losses.append(tr['loss'])
        val_losses.append(va['loss'])
        history.append({'epoch': epoch + 1, 'train': tr, 'val': va})

        joint_tot = sum(tr[f'joint_{a}'] for a in ARMS)
        grip_tot = sum(tr[f'grip_{a}'] for a in ARMS)
        msg = (f"Epoch {epoch+1:3d}/{num_epochs} | "
               f"train {tr['loss']:.6f} (joint {joint_tot:.4f}, grip {grip_tot:.4f}) | "
               f"val {va['loss']:.6f} | "
               f"grip acc " + "/".join(f"{va[f'grip_acc_{a}']:.3f}" for a in ARMS) +
               f" | {tr['it_per_s']:.1f} it/s")
        if 'ee_mm_median' in va:
            msg += f" | EE {va['ee_mm_median']:.1f}mm"
        logger.info(msg)

        if va['loss'] < best_val:
            best_val, best_epoch, since_best = va['loss'], epoch + 1, 0
            with open(Path(checkpoint_dir) / 'best_model.pkl', 'wb') as f:
                pickle.dump({'params': jax.device_get(params),
                             'config': cfg,
                             # Top level, so the evaluator can refuse a
                             # mismatched message setting.
                             'use_messages': use_messages,
                             'use_overhead': use_overhead,
                             'use_wrist': use_wrist,
                             'epoch': epoch + 1,
                             'val_loss': best_val}, f)
        else:
            since_best += 1
            if since_best >= patience:
                logger.info(f'\nEarly stopping: no improvement in {patience} epochs '
                            f'(best was epoch {best_epoch}, val {best_val:.6f})')
                break

    # Final test pass on the best checkpoint
    with open(Path(checkpoint_dir) / 'best_model.pkl', 'rb') as f:
        best = pickle.load(f)
    te = validate(loss_fn_jit, best['params'], test_ds, batch_size,
                  model=model, ee_metric=ee_metric, use_proprio=use_proprio,
                  use_messages=use_messages)

    logger.info('\n' + '=' * 70)
    logger.info(f'Best epoch {best["epoch"]}, val loss {best["val_loss"]:.6f}')
    logger.info(f'Test loss: {te["loss"]:.6f}')
    for a in ARMS:
        logger.info(f'  arm {a}: gripper acc {te[f"grip_acc_{a}"]:.3f} '
                    f'(at transitions: {te[f"grip_acc_flip_{a}"]:.3f}, prior {prior:.3f})')
    if 'ee_mm_median' in te:
        logger.info(f'  EE error (arm {ee_arm}): median {te["ee_mm_median"]:.1f}mm, '
                    f'p90 {te["ee_mm_p90"]:.1f}mm '
                    f'(step0 {te["ee_mm_step0"]:.1f} -> stepN {te["ee_mm_stepN"]:.1f})')
    logger.info('=' * 70)

    metrics_path = Path(log_dir) / f'training_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump({'timestamp': timestamp,
                   'hyperparameters': cfg,
                   'gripper_prior_open': prior,
                   'history': history,
                   'best_epoch': best['epoch'],
                   'best_val_loss': best['val_loss'],
                   'test': te}, f, indent=2)
    plot_loss_curves(train_losses, val_losses, best_epoch,
                     Path(log_dir) / f'loss_curves_{timestamp}.png')
    logger.info(f'\nMetrics: {metrics_path}')
    logger.info(f'Log:     {log_path}')
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache_dir', default=CACHE_DIR)
    ap.add_argument('--feat_dir', default=FEAT_DIR)
    ap.add_argument('--run_dir', default=str(RUN_DIR))
    ap.add_argument('--scene_xml', default=SCENE_XML)
    ap.add_argument('--num_epochs', type=int, default=150)
    ap.add_argument('--learning_rate', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--overhead', action='store_true',
                    help='concatenate the fixed overhead view onto every arm\'s '
                         'self-representation (needs {split}_features_o.npy)')
    ap.add_argument('--overhead_only', action='store_true',
                    help='use only the fixed overhead view, no wrist cameras '
                         '(needs {split}_features_o.npy)')
    ap.add_argument('--no_messages', action='store_true',
                    help='L0 / NO-COORDINATION ABLATION: zero every message '
                         'between the encoders and the decoders, so each of '
                         'the three arms acts only on its own Octo features. '
                         'All six ordered channels (a<-b, a<-c, b<-a, b<-c, '
                         'c<-a, c<-b) go dead together. Distinct from '
                         'evaluating a message-trained checkpoint with '
                         '--no-messages, which runs it out of distribution -- '
                         'this policy never learns to communicate in the first '
                         'place, which is the defensible "is coordination '
                         'necessary" test. Encoder and decoder weights are '
                         'kept and still trained, so parameter count is '
                         'unchanged and the ablation isolates communication '
                         'rather than capacity.')
    ap.add_argument('--no_proprio', action='store_true',
                    help='ablation: train without per-arm joint states')
    ap.add_argument('--proprio_dropout', type=float, default=0.1,
                    help='probability of blanking an arm\'s state (causal confusion)')
    ap.add_argument('--lambda_gripper', type=float, default=1.0,
                    help='weight on the gripper BCE term (2-arm best: 0.25)')
    ap.add_argument('--grip_transition_weight', type=float, default=4.0,
                    help='extra weight on gripper frames near a flip (2-arm best: 0)')
    ap.add_argument('--no_grip_prev', action='store_true',
                    help='ablation: do not give the flip mask one frame of left context')
    ap.add_argument('--no_ee_weights', action='store_true',
                    help='ablation: weight all joints equally in the L1 term')
    ap.add_argument('--diffusion', action='store_true',
                    help='replace the deterministic head with a denoiser')
    ap.add_argument('--diffusion_unet', action='store_true',
                    help='use the 1D temporal U-Net denoiser (needs --diffusion). '
                         'This is the 2-arm configuration that scored 88-91%%.')
    ap.add_argument('--joint_mse', action='store_true',
                    help='ablation: MSE instead of L1 on joints. NOT recommended -- '
                         'the 2-arm equivalent scored 0/50 in the v4 sweep.')
    ap.add_argument('--no_ee_metric', action='store_true')
    ap.add_argument('--ee_all_steps', action='store_true',
                    help='evaluate every chunk step in the EE metric, not just 0 and N')
    ap.add_argument('--ee_arm', default='a', choices=list(ARMS),
                    help='which arm the EE metric measures (default a, as in 2-arm)')
    args = ap.parse_args()

    if args.diffusion_unet and not args.diffusion:
        raise SystemExit('--diffusion_unet only has an effect with --diffusion. '
                         'Pass both, or neither.')

    run_dir = Path(args.run_dir)
    train_cola(
        cache_dir=args.cache_dir,
        feat_dir=args.feat_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=run_dir / 'checkpoints',
        log_dir=run_dir / 'logs',
        use_proprio=not args.no_proprio,
        use_overhead=args.overhead or args.overhead_only,
        use_wrist=not args.overhead_only,
        proprio_dropout=args.proprio_dropout,
        lambda_gripper=args.lambda_gripper,
        grip_transition_weight=args.grip_transition_weight,
        grip_prev_context=not args.no_grip_prev,
        ee_weighted=not args.no_ee_weights,
        joint_l1=not args.joint_mse,
        use_diffusion=args.diffusion,
        diffusion_unet=args.diffusion_unet,
        use_messages=not args.no_messages,
        scene_xml=args.scene_xml,
        ee_metric_on=not args.no_ee_metric,
        ee_all_steps=args.ee_all_steps,
        ee_arm=args.ee_arm,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()
