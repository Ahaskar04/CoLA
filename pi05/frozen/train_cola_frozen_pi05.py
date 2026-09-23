"""
COLA training, v3: identical objective to v2, ~100x faster per step.

v2 measured 3.8 it/s (0.36 s/step) on a 740K-parameter MLP consuming cached
(128, 768) float32 arrays. That is dispatch overhead, not arithmetic. Three
causes, all fixed here:

1. NOTHING WAS JITTED. `jax.value_and_grad(loss_fn)` was called raw, so every
   op in the forward, backward and optimizer update was dispatched individually
   from Python. train_step() below fuses all four into one compiled function.

2. NINE HOST SYNCS PER STEP. v2 did `float(loss)` plus `float(v)` for each of
   the eight aux metrics, every step, each one blocking until the device caught
   up. Metrics are now accumulated as device arrays and pulled once per epoch.

3. A PREFETCH THREAD FOR DATA THAT FITS IN VRAM. The train split is
   20170 x 768 x 4 bytes x 2 arms = 124 MB of features plus ~2 MB of actions
   and states. DeviceDataset uploads all of it once and every batch becomes an
   on-device gather -- no host transfer, no background thread, no locking.

Also changed, all of it optional and off by default except where noted:

  - EEMetric evaluates chunk steps 0 and CHUNK_SIZE-1 rather than all ten
    (5x fewer mj_forward calls). Those are the two that matter: the gap
    between them IS the chunk degradation. --ee_all_steps restores v2.
  - gripper_transition_weights now sees one frame of left context, so a flip
    landing on a chunk's first element is no longer invisible to it. With ~3
    flips per 155-step episode you cannot afford to drop any. --no_grip_prev
    restores v2 behaviour.
  - Validation averages weighted by batch size instead of per-batch, so the
    ragged final batch no longer counts as much as a full one.
  - Checkpoints record cache_dir/feat_dir, and a rollout can assert it is
    denormalising with the same action_stats.json the model was trained on.

ASSUMPTIONS TO CHECK ONCE (they are asserted at startup, so a mismatch fails
loudly rather than silently):
  - features live at {feat_dir}/{split}_features_{a,b}.npy, shape (N, 768)
  - actions  live at {cache_dir}/{split}_actions_{a,b}.npy, shape (N, 7)
  - states   live at {cache_dir}/{split}_states_{a,b}.npy,  shape (N, 7)
    (STATE_FILE_PATTERNS below tries a few spellings; add yours if it differs)
  - COLADataset exposes .valid_starts, and within one episode those starts are
    consecutive integers. Episode boundaries are recovered from the gaps and
    cross-checked against the episode count COLADataset reports.

Usage:
    python train_handover_joint_h5_v3.py --use_proprio
    python train_handover_joint_h5_v3.py --no_proprio --joint_mse --no_ee_weights
"""

import argparse
import functools
import sys
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

# CoLA's own dataset module, unmodified, from the CoLA tree; the architecture
# is the frozen-pi0.5 duplicate next to this file.
sys.path.insert(0, '/scratch/users/ntu/ahaskar0/v1/cola-research-code/handover/cola')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cola_architecture_pi05 import (COLAModel, GRIPPER_IDX, CHUNK_SIZE, ACTION_DIM,
                                    DIFFUSION_STEPS)
from cola_dataset import COLADataset

CACHE_DIR = '/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_marker_v1'
FEAT_DIR = '/scratch/users/ntu/ahaskar0/pi05_marker_features'
SCENE_XML = '/home/users/ntu/ahaskar0/CoLA/environments/handover_2arm/scene.xml'

# Millimetres of gripper travel per 1 std of each joint's action, measured with
# forward kinematics over 150 poses drawn from the v2 dataset:
#   waist 41, shoulder 223, elbow 335, forearm_roll 9, wrist_angle 111,
#   wrist_rotate 0.3
# Used as relative weights, then normalised to mean 1 so the joint term keeps
# its scale against LAMBDA_GRIPPER.
#
# forearm_roll and wrist_rotate are floored at 0.25 rather than used raw: the
# measurement tracks the gripper SITE POSITION only, so it scores wrist rotation
# at ~0, but rotation still decides whether the jaws line up with the box.
JOINT_EE_WEIGHTS = np.array([0.34, 1.86, 2.79, 0.25, 0.93, 0.25], dtype=np.float32)
JOINT_EE_WEIGHTS = JOINT_EE_WEIGHTS / JOINT_EE_WEIGHTS.mean()

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# prepare_states_h5.py's output naming was not visible when this was written.
STATE_FILE_PATTERNS = ('{split}_states_{arm}.npy',
                       '{split}_state_{arm}.npy',
                       '{split}_proprio_{arm}.npy')

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
    """Set up file + console logging."""
    log_path = Path(log_dir) / f'training_log_{timestamp}.txt'
    logger = logging.getLogger('cola_train_v3')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False   # v2 double-printed every line via the root logger

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_path


def plot_loss_curves(train_losses, val_losses, best_epoch, save_path):
    """Plot and save train vs val loss curves."""
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    ax.axvline(x=best_epoch + 1, color='r', linestyle='--', alpha=0.7,
               label=f'Best Epoch ({best_epoch + 1})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (weighted joint + BCE)')
    ax.set_title('COLA v3 Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Device-resident dataset
# --------------------------------------------------------------------------

def _find_state_files(cache_dir, split):
    """Locate the cached proprioception arrays, whatever they are called."""
    for pattern in STATE_FILE_PATTERNS:
        pa = Path(cache_dir) / pattern.format(split=split, arm='a')
        pb = Path(cache_dir) / pattern.format(split=split, arm='b')
        if pa.exists() and pb.exists():
            return pa, pb
    tried = ', '.join(p.format(split=split, arm='{a,b}') for p in STATE_FILE_PATTERNS)
    found = sorted(p.name for p in Path(cache_dir).glob(f'{split}_*.npy'))
    raise SystemExit(
        f'use_proprio=True but no state arrays found in {cache_dir} for split '
        f'{split!r}.\n  tried: {tried}\n  present: {found}\n'
        f'Run prepare_states_h5.py, or add the right name to STATE_FILE_PATTERNS.'
    )


class DeviceDataset:
    """Whole split resident in device memory; batches are on-device gathers.

    Reuses COLADataset only for .valid_starts, which encodes the episode-boundary
    logic (a chunk must not straddle two episodes). Everything else is loaded
    straight from the .npy files so no host->device copy happens per batch.
    """

    def __init__(self, cache_dir, feat_dir, split, use_proprio,
                 chunk_size=CHUNK_SIZE, logger=None, use_overhead=False):
        self.split = split
        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.chunk_size = chunk_size

        base = COLADataset(cache_dir, feat_dir, split=split, use_states=use_proprio)
        starts_np = np.sort(np.asarray(base.valid_starts, dtype=np.int32))
        del base   # release its host-side copy before we upload ours

        cache_dir, feat_dir = Path(cache_dir), Path(feat_dir)
        feat_a = np.load(feat_dir / f'{split}_features_a.npy')
        feat_b = np.load(feat_dir / f'{split}_features_b.npy')
        # One fixed third-person view, shared by both arms. Written only by
        # extract_features_h5.py --overhead.
        feat_o = None
        if use_overhead:
            fo = feat_dir / f'{split}_features_o.npy'
            if not fo.exists():
                raise FileNotFoundError(
                    f'{fo} not found. --overhead needs the overhead features: '
                    f're-run extract_features_h5.py --overhead after deleting '
                    f'the feature dir (it skips when files already exist).')
            feat_o = np.load(fo)
        act_a = np.load(cache_dir / f'{split}_actions_a.npy')
        act_b = np.load(cache_dir / f'{split}_actions_b.npy')

        n = len(feat_a)
        for name, arr in (('features_b', feat_b), ('actions_a', act_a), ('actions_b', act_b)):
            assert len(arr) == n, f'{split}: features_a has {n} rows, {name} has {len(arr)}'
        assert act_a.shape[1] == ACTION_DIM, \
            f'{split}: actions_a is {act_a.shape}, expected (N, {ACTION_DIM})'
        assert starts_np.max() + chunk_size <= n, \
            f'{split}: chunk start {starts_np.max()} + {chunk_size} overruns {n} rows'

        # Within an episode the valid starts are consecutive integers, so a gap
        # marks an episode boundary. Used to give the gripper-flip mask one frame
        # of left context without reading across an episode seam.
        is_ep_start = np.concatenate([[True], np.diff(starts_np) != 1])
        prev_np = np.where(is_ep_start, starts_np, starts_np - 1).astype(np.int32)
        self.n_episodes = int(is_ep_start.sum())

        self.starts = jnp.asarray(starts_np)
        self.prev = jnp.asarray(prev_np)
        self.offsets = jnp.arange(chunk_size, dtype=jnp.int32)
        self.feat_a = jnp.asarray(feat_a)
        self.feat_b = jnp.asarray(feat_b)
        self.feat_o = jnp.asarray(feat_o) if use_overhead else None
        self.act_a = jnp.asarray(act_a)
        self.act_b = jnp.asarray(act_b)

        if use_proprio:
            pa, pb = _find_state_files(cache_dir, split)
            state_a, state_b = np.load(pa), np.load(pb)
            assert len(state_a) == n and len(state_b) == n, \
                f'{split}: states have {len(state_a)}/{len(state_b)} rows, features have {n}'
            self.state_a = jnp.asarray(state_a)
            self.state_b = jnp.asarray(state_b)
        else:
            self.state_a = self.state_b = None

        mb = sum(x.size * x.itemsize for x in (feat_a, feat_b, act_a, act_b)) / 1e6
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
            'features_a': self.feat_a[s],
            'features_b': self.feat_b[s],
            **({'features_o': self.feat_o[s]} if self.use_overhead else {}),
            'action_a': self.act_a[idx],
            'action_b': self.act_b[idx],
            'prev_grip_a': (self.act_a[p, GRIPPER_IDX] > 0).astype(jnp.float32),
            'prev_grip_b': (self.act_b[p, GRIPPER_IDX] > 0).astype(jnp.float32),
        }
        if self.use_proprio:
            out['state_a'] = self.state_a[s]
            out['state_b'] = self.state_b[s]
        return out

    def epoch_batches(self, batch_size, rng=None, drop_last=True):
        """Yield batches. Shuffled when rng is given, in order otherwise."""
        m = len(self)
        order = jax.random.permutation(rng, m) if rng is not None else jnp.arange(m)
        stop = (m // batch_size) * batch_size if drop_last else m
        for i in range(0, stop, batch_size):
            yield self.batch(order[i:i + batch_size])


# --------------------------------------------------------------------------
# Loss  (identical objective to v2)
# --------------------------------------------------------------------------

def gripper_transition_weights(target_chunk, extra_weight, prev_label=None):
    """Weight gripper frames by proximity to a state change.

    target_chunk: (batch, chunk, action_dim), gripper column is +/-0.9.
    Only ~1.9% of timesteps flip the gripper, so an unweighted mean is dominated
    by frames where holding the previous value is already correct. Widened by
    one step either side so the TIMING is supervised, not just the exact frame.

    prev_label: (batch,) 0/1 gripper label of the frame BEFORE the chunk. v2
    used jnp.diff inside the chunk alone, which cannot see a flip landing on
    element 0 -- the comparison frame is outside the window. Passing it in
    recovers those.
    """
    label = (target_chunk[..., GRIPPER_IDX] > 0).astype(jnp.float32)   # (B, C)

    if prev_label is None:
        flip = jnp.abs(jnp.diff(label, axis=1))                        # (B, C-1)
        flip = jnp.pad(flip, ((0, 0), (1, 0)))                         # (B, C)
    else:
        extended = jnp.concatenate([prev_label[:, None], label], axis=1)   # (B, C+1)
        flip = jnp.abs(jnp.diff(extended, axis=1))                     # (B, C)

    # dilate by +/-1 step
    near = jnp.clip(
        flip
        + jnp.pad(flip[:, 1:], ((0, 0), (0, 1)))
        + jnp.pad(flip[:, :-1], ((0, 0), (1, 0))),
        0.0, 1.0,
    )
    return 1.0 + extra_weight * near


def make_loss_fn(model, cfg):
    """Build compute_loss closed over the static config."""
    w_joint = jnp.asarray(JOINT_EE_WEIGHTS) if cfg['ee_weighted'] else jnp.ones(GRIPPER_IDX)
    lam = cfg['lambda_gripper']
    trans_w = cfg['grip_transition_weight']
    use_proprio = cfg['use_proprio']
    joint_l1 = cfg['joint_l1']
    grip_prev = cfg['grip_prev_context']
    p_drop = cfg['proprio_dropout']
    # cfg, not a closure over train_cola's locals: compute_loss is nested in
    # make_loss_fn, which is a separate function and cannot see them.
    use_messages = cfg.get('use_messages', True)

    def _diffusion_loss(model, params, cond_a, cond_b, batch, rng,
                        lam, trans_w, grip_prev):
        """DDPM training objective on the joint columns + BCE on the gripper."""
        ab = model.alpha_bars
        total = 0.0
        parts = {}
        # Validation passes rng=None; use a fixed key so val loss is comparable
        # across epochs rather than varying with the noise draw.
        base = rng if rng is not None else jax.random.PRNGKey(0)

        for tag, cond, target in (('a', cond_a, batch['action_a']),
                                  ('b', cond_b, batch['action_b'])):
            target = jnp.asarray(target)
            joints = target[..., :GRIPPER_IDX]
            B = joints.shape[0]

            k_t, k_n, base = jax.random.split(base, 3)
            t = jax.random.randint(k_t, (B,), 1, DIFFUSION_STEPS)
            noise = jax.random.normal(k_n, joints.shape)
            a_t = ab[t][:, None, None]
            noisy = jnp.sqrt(a_t) * joints + jnp.sqrt(1.0 - a_t) * noise

            eps_pred, grip_logit = model.denoise(cond, noisy, t, params, tag)

            # MSE on the noise: the standard DDPM parameterisation, better
            # conditioned than predicting the action directly because the
            # target is always unit-scale.
            joint_loss = jnp.mean((eps_pred - noise) ** 2)

            label = (target[..., GRIPPER_IDX] > 0).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(grip_logit[..., 0], label)
            prev = batch[f'prev_grip_{tag}'] if grip_prev else None
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
        state_a = batch['state_a'] if use_proprio else None
        state_b = batch['state_b'] if use_proprio else None

        # Randomly blank the proprio input so the head cannot lean on it to the
        # exclusion of vision (causal confusion). Train only -- rng is None at
        # validation.
        if use_proprio and rng is not None and p_drop > 0:
            k1, k2 = jax.random.split(rng)
            keep_a = (jax.random.uniform(k1, (state_a.shape[0], 1)) >= p_drop)
            keep_b = (jax.random.uniform(k2, (state_b.shape[0], 1)) >= p_drop)
            state_a = state_a * keep_a
            state_b = state_b * keep_b

        pred_a, pred_b = model.forward_from_features(
            batch['features_a'], batch['features_b'],
            params=params, proprio_a=state_a, proprio_b=state_b,
            features_o=batch.get('features_o'),
            # L0 / no-coordination ablation. The encoders and decoders still
            # exist and still take gradients -- only the message CONTENT is
            # zeroed -- so parameter count is unchanged and this isolates
            # communication rather than capacity. The EE-metric call below
            # passes the same flag; if the two disagree the reported metric
            # describes a differently-wired model than the loss trains.
            use_messages=use_messages,
        )

        if model.use_diffusion:
            # Under diffusion, forward_from_features returns the CONDITIONING
            # vectors, not actions. Train the denoiser: corrupt the expert's
            # joint chunk to a random noise level and have the head predict the
            # noise that was added. Nothing is regressed toward an action, so
            # two valid ways round the box stay two modes instead of averaging
            # into a reach through the middle.
            #
            # The gripper column never enters the diffusion -- it is binary, and
            # regressing it is the bug that once left the hand permanently shut.
            # The head emits it as a logit and it keeps the same BCE term below.
            return _diffusion_loss(model, params, pred_a, pred_b, batch, rng,
                                   lam, trans_w, grip_prev)

        total = 0.0
        parts = {}
        for tag, pred, target in (('a', pred_a, batch['action_a']),
                                  ('b', pred_b, batch['action_b'])):
            target = jnp.asarray(target)

            # --- joints: weighted by how much each moves the gripper ---
            err = pred[..., :GRIPPER_IDX] - target[..., :GRIPPER_IDX]
            err = jnp.abs(err) if joint_l1 else err ** 2
            joint_loss = (err * w_joint).mean()

            # --- gripper: binary, so BCE on a raw logit ---
            label = (target[..., GRIPPER_IDX] > 0).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(pred[..., GRIPPER_IDX], label)
            prev = batch[f'prev_grip_{tag}'] if grip_prev else None
            gw = gripper_transition_weights(target, trans_w, prev)
            grip_loss = (bce * gw).sum() / gw.sum()

            total = total + joint_loss + lam * grip_loss
            parts[f'joint_{tag}'] = joint_loss
            parts[f'grip_{tag}'] = grip_loss
            # Accuracy on the frames that actually decide the task.
            correct = ((pred[..., GRIPPER_IDX] > 0) == (label > 0.5)).astype(jnp.float32)
            flip_mask = gw > 1.0
            parts[f'grip_acc_{tag}'] = correct.mean()
            parts[f'grip_acc_flip_{tag}'] = (
                (correct * flip_mask).sum() / jnp.maximum(flip_mask.sum(), 1.0)
            )

        return total, parts

    return compute_loss


# --------------------------------------------------------------------------
# Compiled train step
# --------------------------------------------------------------------------

def make_train_step(loss_fn, optimizer):
    """One compiled function: forward, backward, optimizer update.

    donate_argnums lets XLA write the new params and opt_state over the old
    buffers instead of allocating fresh ones. The donated inputs are invalid
    after the call, so nothing may hold another reference to them -- in
    particular model.params is only rebound between epochs, never read during.
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


# --------------------------------------------------------------------------
# End-effector error, in millimetres
# --------------------------------------------------------------------------

class EEMetric:
    """Median gripper-position error of the predicted joint targets, in mm.

    The loss is in normalised units and does not say whether the policy can hit
    a box. This does: it denormalises the predicted and true joint targets and
    measures how far apart the two put the gripper.

    v2 ran mj_forward on every one of the 10 chunk steps, for both pred and
    true, for 256 samples -- 5120 FK calls in a Python loop, every epoch. Once
    the training step is compiled that dominates the epoch. Steps 0 and
    CHUNK_SIZE-1 answer the same question: the spread between them IS the chunk
    degradation.
    """

    def __init__(self, cache_dir, scene_xml, n_samples=256, all_steps=False):
        import mujoco
        self.mujoco = mujoco
        self.n_samples = n_samples
        self.all_steps = all_steps
        with open(Path(cache_dir) / 'action_stats.json') as f:
            self.stats = json.load(f)
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(f'left/{n}').qposadr[0] for n in ARM_JOINTS]
        self.key = self.model.key('neutral_pose').id
        # A velocity cache stores per-step DELTAS. Writing those straight into
        # qpos would put the arm at "0.03 rad from the origin" rather than
        # "current pose + 0.03", making the metric meaningless. Both pred and
        # true get the same treatment, so the comparison stays fair either way.
        self.velocity = bool(self.stats.get('velocity', False))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        self.home_q = self.data.qpos[self.qadr].copy()

    def _denorm(self, a, arm):
        m = self.stats['margin']
        lo = np.array(self.stats[arm]['min'], np.float32)
        span = np.array(self.stats[arm]['max'], np.float32) - lo
        return (a + m) / (2 * m) * span + lo

    def _ee(self, q):
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        # Deltas are measured from the home pose: without the true starting
        # configuration (which the cached features do not carry) this is the
        # one reference both pred and true can share.
        self.data.qpos[self.qadr] = (self.home_q + q) if self.velocity else q
        self.mujoco.mj_forward(self.model, self.data)
        return self.data.site('left/gripper').xpos.copy()

    def __call__(self, pred_a, true_a):
        pred = self._denorm(np.asarray(pred_a), 'action_a')
        true = self._denorm(np.asarray(true_a), 'action_a')
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


# --------------------------------------------------------------------------
# Train / validate
# --------------------------------------------------------------------------

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

        # One sync per 100 steps, for the PBS progress file only.
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
    """Deterministic sweep over every valid chunk-start.

    Walks the split in order so the number is repeatable. Unlike v2 this
    weights each batch by its size, so the ragged final batch does not count
    as much as a full one.
    """
    sums, total_n = None, 0
    pred_acc, true_acc = [], []

    for batch in dataset.epoch_batches(batch_size, rng=None, drop_last=False):
        n = batch['features_a'].shape[0]
        loss, parts = loss_fn_jit(params, batch, None)

        acc = jax.tree_util.tree_map(lambda x: x * n, {'loss': loss, **parts})
        sums = acc if sums is None else jax.tree_util.tree_map(jnp.add, sums, acc)
        total_n += n

        if ee_metric is not None and len(pred_acc) < 4:
            out_a, _ = model.forward_from_features(
                batch['features_a'], batch['features_b'], params=params,
                proprio_a=batch.get('state_a') if use_proprio else None,
                proprio_b=batch.get('state_b') if use_proprio else None,
                features_o=batch.get('features_o'),
                # Must match the loss path's setting -- see the note there.
                use_messages=use_messages)
            if model.use_diffusion:
                # Under diffusion forward_from_features returns CONDITIONING,
                # not actions -- feeding that straight to the EE metric would
                # silently measure the wrong tensor. Sample an actual chunk.
                # Fixed key so the metric is comparable across epochs rather
                # than moving with the noise draw.
                pa = model.sample_actions(out_a, params, 'a',
                                          jax.random.PRNGKey(0))
            else:
                pa = out_a
            pred_acc.append(np.asarray(pa))
            true_acc.append(np.asarray(batch['action_a']))

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
    checkpoint_dir='experiments/run_aloha_handover_v3/checkpoints',
    log_dir='experiments/run_aloha_handover_v3/logs',
    use_proprio=True,
    use_overhead=False,
    proprio_dropout=0.1,
    lambda_gripper=1.0,
    grip_transition_weight=4.0,
    grip_prev_context=True,
    ee_weighted=True,
    joint_l1=True,
    use_diffusion=False,
    diffusion_unet=False,
    unet_dims=None,
    use_messages=True,
    ee_metric=True,
    ee_all_steps=False,
    scene_xml=SCENE_XML,
    patience=25,
    seed=0,
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    progress_file = Path(log_dir) / 'progress.txt'

    logger, log_path = setup_logging(log_dir, timestamp)
    metrics_path = Path(log_dir) / f'training_metrics_{timestamp}.json'
    plot_path = Path(log_dir) / f'loss_curves_{timestamp}.png'

    cfg = {
        'use_proprio': use_proprio,
        'use_overhead': use_overhead,
        'proprio_dropout': proprio_dropout,
        'lambda_gripper': lambda_gripper,
        'grip_transition_weight': grip_transition_weight,
        'grip_prev_context': grip_prev_context,
        'ee_weighted': ee_weighted,
        'joint_l1': joint_l1,
        'use_diffusion': use_diffusion,
        'use_messages': use_messages,
        'diffusion_unet': diffusion_unet,
        # Recorded even when None: the eval rebuilds the head from this, and a
        # checkpoint trained at (64,128) loaded into a (128,256) head is a
        # parameter-shape error, not a silent degradation.
        'unet_dims': list(unet_dims) if unet_dims else None,
    }

    logger.info('=' * 60)
    logger.info('COLA Training v3 (jitted step, device-resident data)')
    logger.info('=' * 60)
    logger.info(f'Timestamp: {timestamp}')
    logger.info(f'devices: {jax.devices()}')
    logger.info(f'lr={learning_rate}, epochs={num_epochs}, batch_size={batch_size}, seed={seed}')
    logger.info(f'config: {cfg}')
    if ee_weighted:
        logger.info(f'  joint weights (EE-normalised): '
                    f'{dict(zip(ARM_JOINTS, np.round(JOINT_EE_WEIGHTS, 3)))}')
    gpu_info = get_gpu_info()
    if gpu_info:
        logger.info(f'GPU: {gpu_info}')

    logger.info('\n1. Loading datasets to device...')
    train_dataset = DeviceDataset(cache_dir, feat_dir, 'train', use_proprio, logger=logger,
                    use_overhead=use_overhead)
    val_dataset = DeviceDataset(cache_dir, feat_dir, 'val', use_proprio, logger=logger,
                  use_overhead=use_overhead)
    test_dataset = DeviceDataset(cache_dir, feat_dir, 'test', use_proprio, logger=logger,
                  use_overhead=use_overhead)

    # The gripper head can hit high accuracy by learning the class prior alone.
    # Log the prior so the training numbers can be read against the right baseline.
    grip_train = np.asarray(jax.device_get(train_dataset.act_a[:, GRIPPER_IDX]))
    prior = float((grip_train > 0).mean())
    logger.info(f'  gripper class balance (arm A, train): {prior:.3f} open / '
                f'{1 - prior:.3f} closed  <- accuracy below this is worse than the prior')

    logger.info('\n2. Initializing COLA model...')
    model = COLAModel(use_proprio=use_proprio, split_gripper=True,
                      use_overhead=use_overhead, use_diffusion=use_diffusion,
                      diffusion_unet=diffusion_unet, unet_dims=unet_dims)
    if not use_messages:
        logger.info('  messages: SEVERED (L0 ablation) -- each arm sees only '
                    'its own features; evaluate this checkpoint with '
                    '--no-messages')
    if use_diffusion:
        logger.info(f'  denoiser: {"1D temporal U-Net (FiLM per block)" if diffusion_unet else "flat residual MLP"}')
        if diffusion_unet:
            logger.info(f'  unet down_dims: {unet_dims if unet_dims else "(128, 256) [default]"}')
        logger.info(f'  action head: DIFFUSION denoiser over the 6 joint columns '
                    f'({DIFFUSION_STEPS} train noise levels); gripper stays on the '
                    f'BCE logit head')
    loss_fn = make_loss_fn(model, cfg)
    loss_fn_jit = jax.jit(loss_fn)

    metric = None
    if ee_metric:
        try:
            metric = EEMetric(cache_dir, scene_xml, all_steps=ee_all_steps)
            logger.info(f'  end-effector metric: on (arm A, chunk steps '
                        f'{"all" if ee_all_steps else "0 and N-1"})')
        except Exception as e:
            logger.info(f'  end-effector metric unavailable ({e}); continuing without it')

    logger.info('\n3. Setting up optimizer...')
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(1, num_epochs * steps_per_epoch),
        alpha=0.1,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    params = model.params
    opt_state = optimizer.init(params)
    train_step = make_train_step(loss_fn, optimizer)
    logger.info(f'  {steps_per_epoch} steps/epoch, {num_epochs * steps_per_epoch} total')

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses, val_losses = [], []
    history = []
    test_loss = None

    rng = jax.random.PRNGKey(seed)
    start_time = time.time()

    logger.info('\n4. Training...')
    logger.info('=' * 60)

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()

            tr, params, opt_state, rng = train_epoch(
                train_step, params, opt_state, train_dataset, rng,
                batch_size, epoch, num_epochs, progress_file)
            model.params = params   # rebound only here, never read mid-epoch

            va = validate(loss_fn_jit, params, val_dataset, batch_size,
                          model=model, ee_metric=metric, use_proprio=use_proprio,
                          use_messages=use_messages)

            train_losses.append(tr['loss'])
            val_losses.append(va['loss'])
            history.append({'epoch': epoch + 1, 'train': tr, 'val': va})

            line = (f"Epoch {epoch+1}/{num_epochs} | "
                    f"train {tr['loss']:.6f} (joint {tr['joint_a']+tr['joint_b']:.4f}, "
                    f"grip {tr['grip_a']+tr['grip_b']:.4f}) | "
                    f"val {va['loss']:.6f} | "
                    f"grip acc {va['grip_acc_a']:.3f}/flip {va['grip_acc_flip_a']:.3f}")
            if 'ee_mm_median' in va:
                line += (f" | EE {va['ee_mm_median']:.1f}mm "
                         f"(p90 {va['ee_mm_p90']:.1f}, k0 {va['ee_mm_step0']:.1f}"
                         f"->kN {va['ee_mm_stepN']:.1f})")
            line += f" | {time.time()-epoch_start:.1f}s ({tr['it_per_s']:.0f} it/s)"
            logger.info(line)

            payload = {
                'params': params,
                'epoch': epoch,
                'train_loss': tr['loss'],
                'val_loss': va['loss'],
                # A rollout must know how the checkpoint was built: whether to
                # feed it proprio, and that column 6 is a logit not a position.
                'use_proprio': use_proprio,
                # self_dim is 768 larger when overhead is on, so a rollout that
                # rebuilds the model without this flag gets 832-dim params and
                # fails to load 1600-dim weights.
                'use_overhead': use_overhead,
                'split_gripper': True,
                # TOP LEVEL deliberately: cola_eval_aloha.py reads
                # ckpt.get('use_messages', True) from here, not from config, to
                # refuse evaluating a severed checkpoint with messages on. Left
                # only in cfg, that guard is blind.
                'use_messages': use_messages,
                'config': cfg,
                # Provenance. cola_eval_aloha.py defaults to the v1 cache; if it
                # denormalises with different action_stats.json the arm goes to a
                # systematically wrong pose and nothing errors. Assert on load.
                'cache_dir': str(cache_dir),
                'feat_dir': str(feat_dir),
            }

            if va['loss'] < best_val_loss:
                best_val_loss = va['loss']
                best_epoch = epoch
                with open(Path(checkpoint_dir) / 'best_model.pkl', 'wb') as f:
                    pickle.dump(payload, f)
                logger.info(f"  >>> New best model saved (val {va['loss']:.6f})")

            if (epoch + 1) % 10 == 0:
                p = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pkl'
                with open(p, 'wb') as f:
                    pickle.dump({**payload, 'opt_state': opt_state,
                                 'best_val_loss': best_val_loss,
                                 'best_epoch': best_epoch}, f)
                logger.info(f'  Periodic checkpoint saved: {p}')

            if epoch - best_epoch > patience:
                logger.info(f'\nEarly stopping: no improvement for {patience} epochs.')
                logger.info(f'Best epoch: {best_epoch+1}')
                break

    except KeyboardInterrupt:
        logger.info('\n*** Training interrupted ***')
    finally:
        total_time = time.time() - start_time
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'hyperparameters': {'learning_rate': learning_rate,
                                    'num_epochs': num_epochs,
                                    'batch_size': batch_size,
                                    'seed': seed, **cfg},
                'gripper_prior_open': prior,
                'history': history,
                'best_epoch': int(best_epoch),
                'best_val_loss': float(best_val_loss) if val_losses else None,
                'total_training_time_seconds': float(total_time),
            }, f, indent=2, default=float)
        if train_losses:
            plot_loss_curves(train_losses, val_losses, best_epoch, plot_path)
        logger.info(f'  Metrics saved: {metrics_path}')

    best_path = Path(checkpoint_dir) / 'best_model.pkl'
    if best_path.exists():
        logger.info('\n' + '=' * 60)
        logger.info('5. Final Evaluation')
        logger.info('=' * 60)
        with open(best_path, 'rb') as f:
            best = pickle.load(f)
        model.params = best['params']
        te = validate(loss_fn_jit, best['params'], test_dataset, batch_size,
                      model=model, ee_metric=metric, use_proprio=use_proprio,
                      use_messages=use_messages)
        test_loss = te['loss']
        logger.info(f"\nBest model (epoch {best['epoch']+1}):")
        logger.info(f"  Val loss:  {best['val_loss']:.6f}")
        logger.info(f"  Test loss: {test_loss:.6f}")
        logger.info(f"  Test gripper acc: {te['grip_acc_a']:.3f} "
                    f"(at transitions: {te['grip_acc_flip_a']:.3f}, prior {prior:.3f})")
        if 'ee_mm_median' in te:
            logger.info(f"  Test EE error: {te['ee_mm_median']:.1f} mm median, "
                        f"{te['ee_mm_p90']:.1f} mm p90")
            logger.info(f"    chunk step 0: {te['ee_mm_step0']:.1f} mm -> "
                        f"step {CHUNK_SIZE-1}: {te['ee_mm_stepN']:.1f} mm")
            logger.info('  (box needs roughly +/-15 mm; v2 measured ~25 mm at descend/grasp)')

    logger.info(f'\nTotal training time: {(time.time()-start_time)/60:.1f} minutes')
    return model, train_losses, val_losses, test_loss


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train COLA adapters, jitted.')
    ap.add_argument('--cache_dir', default=CACHE_DIR)
    ap.add_argument('--feat_dir', default=FEAT_DIR)
    ap.add_argument('--run_dir', default='experiments/run_aloha_handover_v4')
    ap.add_argument('--scene_xml', default=SCENE_XML)
    ap.add_argument('--num_epochs', type=int, default=150)
    ap.add_argument('--learning_rate', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--overhead', action='store_true',
                    help="concatenate the shared overhead-camera features onto "
                         "both arms' self-representation (self_dim 768 -> 1536). "
                         "Requires extract_features_h5.py --overhead.")
    ap.add_argument('--no_proprio', action='store_true',
                    help='ablation: train without state_a/state_b')
    ap.add_argument('--proprio_dropout', type=float, default=0.1,
                    help='probability of blanking the state input during training')
    ap.add_argument('--lambda_gripper', type=float, default=1.0,
                    help='weight on the gripper BCE term; at 1.0 it was 77%% of v2 total loss')
    ap.add_argument('--grip_transition_weight', type=float, default=4.0,
                    help='extra weight on gripper frames near a state change (0 = uniform)')
    ap.add_argument('--no_grip_prev', action='store_true',
                    help='ablation: v2 flip mask, blind to a flip on chunk element 0')
    ap.add_argument('--no_ee_weights', action='store_true',
                    help='ablation: weight all six joints equally')
    ap.add_argument('--diffusion_unet', action='store_true',
                    help='with --diffusion, use the 1D temporal U-Net denoiser '
                         'instead of the flat MLP: convolutions along the chunk '
                         'plus FiLM conditioning at every block. This is the '
                         'variant Chi et al. recommend trying first.')
    ap.add_argument('--no_messages', action='store_true',
                    help='L0 / no-coordination ABLATION: train with the '
                         'inter-agent message channel zeroed, so each arm acts '
                         'only on its own Octo features. Distinct from '
                         'evaluating a message-trained checkpoint with '
                         '--no-messages, which runs it out of distribution and '
                         'scored ~0%% -- this policy never learns to '
                         'communicate in the first place, which is the '
                         'defensible "is coordination necessary" test. Encoder '
                         'and decoder weights are kept and still trained, so '
                         'parameter count is unchanged and the ablation '
                         'isolates communication rather than capacity.')
    ap.add_argument('--unet_dims', type=int, nargs='+', default=None,
                    metavar='DIM',
                    help='CAPACITY ABLATION for --diffusion_unet: channel width '
                         'per U-Net level. Omit for the (128, 256) default that '
                         'measured 88%%. "--unet_dims 64 128" halves the channels '
                         'at both levels; "--unet_dims 128" drops to one level '
                         '(chunk 10->5, no second downsample). The U-Net beat the '
                         'flat MLP 88%% to 30.5%%, but temporal convolution, FiLM '
                         'and raw parameter count all changed together -- this '
                         'holds the mechanism fixed and varies only the capacity. '
                         'Diffusion overfits small data while the train loss keeps '
                         'falling, so compare VAL loss and rollout, not train.')
    ap.add_argument('--diffusion', action='store_true',
                    help='replace the deterministic joint head with a diffusion '
                         'denoiser. The L1 head predicts the AVERAGE of valid '
                         'actions, so two ways round the box average into a reach '
                         'through it; diffusion keeps the modes apart. The gripper '
                         'column stays on the BCE logit head either way.')
    ap.add_argument('--joint_mse', action='store_true',
                    help='use MSE on joints instead of L1')
    ap.add_argument('--no_ee_metric', action='store_true')
    ap.add_argument('--ee_all_steps', action='store_true',
                    help='v2 behaviour: FK on all 10 chunk steps instead of first/last')
    args = ap.parse_args()

    train_cola(
        cache_dir=args.cache_dir,
        feat_dir=args.feat_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=f'{args.run_dir}/checkpoints',
        log_dir=f'{args.run_dir}/logs',
        use_proprio=not args.no_proprio,
        use_overhead=args.overhead,
        proprio_dropout=args.proprio_dropout,
        lambda_gripper=args.lambda_gripper,
        grip_transition_weight=args.grip_transition_weight,
        grip_prev_context=not args.no_grip_prev,
        ee_weighted=not args.no_ee_weights,
        joint_l1=not args.joint_mse,
        use_diffusion=args.diffusion,
        diffusion_unet=args.diffusion_unet,
        unet_dims=tuple(args.unet_dims) if args.unet_dims else None,
        use_messages=not args.no_messages,
        ee_metric=not args.no_ee_metric,
        ee_all_steps=args.ee_all_steps,
        scene_xml=args.scene_xml,
        patience=args.patience,
        seed=args.seed,
    )

    print('\n' + '=' * 60)
    print('READY FOR ENVIRONMENT EVALUATION')
    print('=' * 60)
    print('Next: python cola_eval_aloha.py '
          '--model experiments/run_aloha_handover_v4/checkpoints/best_model.pkl '
          '--cache-dir <the SAME cache_dir this was trained on>')