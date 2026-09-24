"""Finetune Octo on the ALOHA handover demos, one arm at a time.

This follows octo's own examples/02_finetune_new_observation_action.py, which
finetunes on the ALOHA sim CUBE HANDOVER dataset -- i.e. this task. The recipe
there is not optional decoration: a pretrained Octo emits 7-d delta end-effector
actions for a single WidowX-like arm, and ALOHA needs absolute joint positions.
There is no correspondence between those spaces, which is why the reference
DELETES the pretrained action head and learns a new one. Everything below is
that recipe, with the data coming from HDF5 instead of RLDS.

  - wrist image tokenizer removed (the reference removes it; our demos have no
    second camera per arm anyway)
  - proprio added as a LowdimObsTokenizer over the arm's own joint state
  - action head fully replaced with L1ActionHead, action_dim 7 for one arm or
    14 for the centralised both-arm variant
  - action_horizon 50, L1 loss, following Zhao et al. (ACT) as the reference does
  - pretrained transformer weights merged in; only the head and the proprio
    position encodings start from scratch

--arm a and --arm b give the DECENTRALISED baseline: two independent policies,
each seeing only its own wrist camera, with no channel between them. That is the
comparison CoLA is actually about. --arm both gives the centralised reference
recipe as an upper bound.

    python finetune_octo_arm.py --arm a --save_dir /path/to/ckpt_a
"""

import argparse
import json
import os
import time
from pathlib import Path

_T0 = time.time()                      # process start, for --time-limit-min

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

import jax
import optax
import flax
import tqdm

try:                                   # keep TF off the GPU if octo drags it in
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from octo.model.components.action_heads import (
    DiffusionActionHead, L1ActionHead, MSEActionHead)
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import freeze_weights, merge_params, TrainState

import octo_h5_data as D

from loss_curve import write_history
from octo_unet_head import UNetActionHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('a', 'b', 'c', 'both'), required=True)
    p.add_argument('--pretrained-path', default='hf://rail-berkeley/octo-small-1.5')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--manifest', default=D.MANIFEST)
    p.add_argument('--instruction', default=None,
                   help='overrides the per-arm default; this is the only signal '
                        'telling the two decentralised policies apart')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--action-horizon', type=int, default=50,
                   help="predicted chunk length. The reference uses 50 (ACT-style). "
                        "Evaluation executes only the first --exec-horizon of them, "
                        "so this can exceed CoLA's control cadence without "
                        "changing the comparison.")
    p.add_argument('--image-key', default=None,
                   choices=(None, 'image_overhead', 'image_wrist_a', 'image_wrist_b',
                            'image_wrist_c'),
                   help='camera stream fed as image_primary. Default is the '
                        "arm's own wrist view; image_overhead gives the single "
                        'third-person camera configuration.')
    p.add_argument('--wrist-key',
                   choices=(None, 'image_wrist_a', 'image_wrist_b'), default=None,
                   help="second camera, fed to octo's pretrained `wrist` "
                        "tokenizer alongside image_primary. This is the "
                        "two-camera configuration the Octo paper compares "
                        "against a single third-person view.")
    p.add_argument('--grad-accum', type=int, default=1,
                   help='split each optimiser step into N micro-batches of '
                        'batch-size/N. Keeps the effective batch (and so the '
                        'training budget) identical while the GPU only ever '
                        'holds 1/N of it -- needed for the two-camera runs, '
                        'which double the image tokens and OOM at batch 128.')
    p.add_argument('--head', choices=('l1', 'diffusion', 'mse', 'unet'), default='l1',
                   help="l1 is what octo's ALOHA example uses (following ACT). "
                        "diffusion is what octo-small-1.5 itself ships with and "
                        "what the Octo paper's ablation prefers; it also models "
                        "the action DISTRIBUTION rather than its mean, which is "
                        "the failure mode seen on this task.")
    p.add_argument('--unet-features', default='256,512,1024',
                   help="channels per U-Net level for --head unet. The default "
                        "is Chi et al.'s (68.5M params); 128,256 is CoLA's own "
                        "head size (5.3M) and allows a 10-step chunk.")
    p.add_argument('--no-proprio', action='store_true',
                   help="drop the proprio tokenizer. The ALOHA reference adds "
                        "it, but the Octo paper reports proprio hurting.")
    p.add_argument('--window-size', type=int, default=1,
                   help='the reference ALOHA recipe uses 1')
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--freeze-transformer', action='store_true',
                   help='freeze only the transformer blocks; tokenizers and '
                        'head stay trainable. Ignored if --tune is set.')
    p.add_argument('--tune', choices=('full', 'head_only', 'head_mlp_only'),
                   default='full',
                   help="which parameters to update. 'head_only' freezes the "
                        "whole octo_transformer, matching octo's own "
                        "finetuning config of the same name.")
    p.add_argument('--val-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stop-at', type=int, default=None,
                   help='end this job after this optimiser step, writing '
                        'resume_state.npz; a later --resume job finishes the '
                        'run. Splits a run across queue walltime caps.')
    p.add_argument('--resume', action='store_true',
                   help='continue from save-dir/resume_state.npz if present '
                        '(params, optimiser state, dropout rng, data stream '
                        'position), otherwise start from step 0.')
    p.add_argument('--time-limit-min', type=float, default=None,
                   help='end this job at the first optimiser step after this '
                        'many minutes since process start, writing '
                        'resume_state.npz. For chaining segments under a '
                        'walltime cap when the throughput is not known.')
    args = p.parse_args()

    # A chained segment queued after the run already finished: nothing to do.
    _sd = Path(args.save_dir)
    if (args.resume and (_sd / str(args.steps - 1)).is_dir()
            and not (_sd / 'resume_state.npz').exists()):
        print(f'{_sd} already complete (checkpoint {args.steps - 1}, no resume '
              f'state) -- nothing to do')
        return

    assert args.batch_size % jax.device_count() == 0
    initialize_compilation_cache()
    instruction = args.instruction or D.DEFAULT_INSTRUCTION[args.arm]

    print('=' * 66)
    print(f'FINETUNE OCTO  |  arm {args.arm}')
    print('=' * 66)
    print(f'devices: {jax.devices()}')
    print(f'instruction: {instruction!r}')

    print('\n1. Loading pretrained model...')
    pretrained = OctoModel.load_pretrained(args.pretrained_path)
    text_processor = pretrained.text_processor
    tokens = text_processor.encode([instruction])
    tokens = {k: np.asarray(v) for k, v in tokens.items()}

    print('\n2. Loading demos...')
    train = D.H5Split(args.manifest, 'train', args.arm, image_key=args.image_key,
                       wrist_key=args.wrist_key)
    val = D.H5Split(args.manifest, 'val', args.arm, image_key=args.image_key,
                     wrist_key=args.wrist_key)
    stats = train.statistics()          # train split only, never val/test

    assert args.batch_size % args.grad_accum == 0, (
        f'--batch-size {args.batch_size} must divide by --grad-accum '
        f'{args.grad_accum}')
    micro_batch = args.batch_size // args.grad_accum
    if args.grad_accum > 1:
        print(f'   grad accumulation: {args.grad_accum} x {micro_batch} '
              f'= effective batch {args.batch_size}')

    use_proprio = not args.no_proprio
    it = D.batch_iterator(train, stats, tokens, micro_batch,
                          args.window_size, args.action_horizon, seed=args.seed,
                          with_proprio=use_proprio)
    example_batch = next(it)
    print('   example batch:')
    shown = [('image_primary', example_batch['observation']['image_primary'])]
    if 'image_wrist' in example_batch['observation']:
        shown.append(('image_wrist',
                      example_batch['observation']['image_wrist']))
    if 'proprio' in example_batch['observation']:
        shown.append(('proprio', example_batch['observation']['proprio']))
    shown.append(('action', example_batch['action']))
    for k, v in shown:
        print(f'     {k:16s} {v.shape} {v.dtype}')

    print('\n3. Rebuilding model for the new observation & action space...')
    config = pretrained.config
    # The reference drops the wrist tokenizer; our per-arm demos have one camera.
    # octo-small-1.5 ships pretrained `primary` AND `wrist` tokenizers. Drop
    # the wrist one only when there is no second camera to feed it -- an
    # unfed tokenizer would be asked for an observation key that never arrives.
    if not args.wrist_key and 'wrist' in config['model']['observation_tokenizers']:
        del config['model']['observation_tokenizers']['wrist']
    if use_proprio:
        config['model']['observation_tokenizers']['proprio'] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256, bin_type='normal', low=-2.0, high=2.0,
            obs_keys=['proprio'],
        )
    else:
        config['model']['observation_tokenizers'].pop('proprio', None)
    # Fully replace the head: the pretrained one speaks 7-d delta EE for a
    # WidowX, this one speaks absolute ALOHA joint targets.
    if args.head == 'diffusion':
        # Same kwargs octo-small-1.5 ships with, only the horizon and dim change.
        config['model']['heads']['action'] = ModuleSpec.create(
            DiffusionActionHead,
            action_horizon=args.action_horizon,
            action_dim=train.action_dim,
            readout_key='readout_action',
            use_map=False, n_diffusion_samples=1, dropout_rate=0.0,
        )
    elif args.head == 'unet':
        # Chi et al.'s 1D temporal U-Net denoiser in a 1.5-format head (see
        # octo_unet_head.py for why octo's own UNetDDPMActionHead can't be
        # used). No pretrained weights exist for it, so it always starts cold.
        unet_features = tuple(int(x) for x in args.unet_features.split(','))
        div = 2 ** (len(unet_features) - 1)
        assert args.action_horizon % div == 0, (
            f'--head unet with {len(unet_features)} levels needs --action-horizon '
            f'divisible by {div}, got {args.action_horizon}')
        config['model']['heads']['action'] = ModuleSpec.create(
            UNetActionHead,
            action_horizon=args.action_horizon,
            action_dim=train.action_dim,
            readout_key='readout_action',
            use_map=False,
            down_features=unet_features,
        )
    else:
        config['model']['heads']['action'] = ModuleSpec.create(
            L1ActionHead if args.head == 'l1' else MSEActionHead,
            action_horizon=args.action_horizon,
            action_dim=train.action_dim,
            readout_key='readout_action',
        )
    print(f'   head: {args.head} | proprio: {use_proprio}')

    model = OctoModel.from_config(
        config, example_batch, text_processor, verbose=True,
        dataset_statistics=stats,
    )
    # Report which action-head tensors actually come from pretraining. merge_params
    # silently skips any key whose shape changed -- e.g. the diffusion head's
    # output layer at action_horizon 50 (350 wide) vs the pretrained 4 (28 wide)
    # -- and a skipped tensor starts from random init. Whether the head is warm
    # is the premise of the horizon-4 experiment, so it is measured, not assumed.
    import flax
    _new = flax.traverse_util.flatten_dict(model.params)
    _old = flax.traverse_util.flatten_dict(pretrained.params)
    _head = [k for k in _new if k[0].startswith('heads_')]
    _warm = [k for k in _head if k in _old and _old[k].shape == _new[k].shape]
    _cold = [k for k in _head if k not in _warm]
    _nw = sum(_new[k].size for k in _warm)
    _nc = sum(_new[k].size for k in _cold)
    print(f'   action head: {len(_warm)}/{len(_head)} tensors warm from pretraining '
          f'({_nw:,} params), {len(_cold)} cold ({_nc:,} params)')
    for k in _cold:
        _why = ('absent in pretrained' if k not in _old
                else f'shape {tuple(_old[k].shape)} -> {tuple(_new[k].shape)}')
        print(f'     cold: {"/".join(k)}  [{_why}]')
    model = model.replace(params=merge_params(model.params, pretrained.params))
    del pretrained

    print('\n4. Optimizer...')
    schedule = optax.join_schedules(
        [optax.linear_schedule(0, args.lr, 100), optax.constant_schedule(args.lr)],
        [100])
    tx = optax.adamw(schedule)
    # freeze_weights fnmatches the FULL dot-joined parameter path, so a bare
    # module name never matches -- every pattern here has to be anchored the
    # way octo's own configs write them.
    frozen_keys = list(model.config['optimizer']['frozen_keys'])
    if args.tune == 'head_only':
        frozen_keys.append('octo_transformer.*')
    elif args.tune == 'head_mlp_only':
        frozen_keys += ['octo_transformer.*', 'heads_*.map_head.*']
    elif args.freeze_transformer:
        frozen_keys.append('octo_transformer.BlockTransformer_0.*')
    print(f'   finetune mode: {args.tune}')
    print(f'   frozen keys: {frozen_keys}')
    tx = freeze_weights(tx, model.params, frozen_keys)
    if args.grad_accum > 1:
        # MultiSteps accumulates and applies once every k calls, so k micro
        # steps == one optimiser step at the full effective batch.
        tx = optax.MultiSteps(tx, every_k_schedule=args.grad_accum)

    # Report what is actually trainable: a pattern that silently matches
    # nothing looks identical to a full finetune in the loss curve until the
    # numbers are compared against another run.
    from fnmatch import fnmatch
    import flax
    n_all = n_train = 0
    for path, v in flax.traverse_util.flatten_dict(model.params).items():
        dotted = '.'.join(path)
        n_all += v.size
        if not any(fnmatch(dotted, k) for k in frozen_keys):
            n_train += v.size
    print(f'   trainable {n_train:,} / {n_all:,} params '
          f'({100.0 * n_train / max(n_all, 1):.2f}%)')
    train_state = TrainState.create(rng=jax.random.PRNGKey(1234), model=model, tx=tx)

    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({'params': params}, rngs={'dropout': rng})
        emb = bound.octo_transformer(
            batch['observation'], batch['task'],
            batch['observation']['timestep_pad_mask'], train=train)
        return bound.heads['action'].loss(
            emb, batch['action'], batch['observation']['timestep_pad_mask'],
            batch['action_pad_mask'], train=train)

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True)
        return state.apply_gradients(grads=grads, rng=rng), info

    @jax.jit
    def val_step(state, batch):
        _, info = loss_fn(state.model.params, batch,
                          jax.random.PRNGKey(0), train=False)
        return info

    val_it = D.batch_iterator(val, stats, tokens, micro_batch,
                              args.window_size, args.action_horizon,
                              seed=args.seed + 1, with_proprio=use_proprio)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {'arm': args.arm, 'instruction': instruction,
               'head': args.head, 'tune': args.tune,
               'unet_features': (args.unet_features if args.head == 'unet' else None),
               'use_proprio': bool(use_proprio),
               'image_key': train.image_key,
               'wrist_key': args.wrist_key,
               'wrist_camera': (D.IMAGE_KEY_TO_CAMERA[args.wrist_key]
                                if args.wrist_key else None),
               'camera': D.IMAGE_KEY_TO_CAMERA[train.image_key],
               'action_dim': int(train.action_dim),
               'proprio_dim': int(train.proprio_dim),
               'action_horizon': args.action_horizon,
               'window_size': args.window_size,
               'grad_accum': args.grad_accum,
               'manifest': args.manifest}
    json.dump(meta, open(save_dir / 'finetune_meta.json', 'w'), indent=2)
    history = []

    K = args.grad_accum
    stop = args.steps if args.stop_at is None else min(args.stop_at, args.steps)
    resume_path = save_dir / 'resume_state.npz'

    def resume_leaves(state):
        return (state.model.params, state.opt_state, state.rng)

    def save_resume(state, at_step):
        leaves = jax.tree_util.tree_leaves(resume_leaves(state))
        tmp = save_dir / 'resume_state.tmp.npz'
        np.savez(tmp, step=np.int64(at_step),
                 **{f'leaf_{j}': np.asarray(x) for j, x in enumerate(leaves)})
        os.replace(tmp, resume_path)             # never leave a half-written file

    start = 0
    if args.resume and resume_path.exists():
        leaves, treedef = jax.tree_util.tree_flatten(resume_leaves(train_state))
        with np.load(resume_path) as z:
            start = int(z['step'])
            n_saved = len(z.files) - 1
            saved = [z[f'leaf_{j}'] for j in range(min(n_saved, len(leaves)))]
        bad = [j for j, (a, b) in enumerate(zip(saved, leaves))
               if a.shape != b.shape or a.dtype != b.dtype]
        if n_saved != len(leaves) or bad:
            raise SystemExit(f'--resume: {resume_path} does not match this model/'
                             f'optimiser ({n_saved} vs {len(leaves)} tensors, '
                             f'{len(bad)} mismatched)')
        params, opt_state, rng = jax.tree_util.tree_unflatten(
            treedef, [jax.numpy.asarray(x) for x in saved])
        train_state = train_state.replace(
            model=train_state.model.replace(params=params),
            opt_state=opt_state, rng=rng, step=start * K)
        hist_file = save_dir / 'training_history.json'
        if hist_file.exists():
            history = [h for h in json.load(open(hist_file)) if h['step'] <= start]
        # +1 for the example batch drawn before the model was built; each val
        # readout draws 4 batches.
        it = D.batch_iterator(train, stats, tokens, micro_batch,
                              args.window_size, args.action_horizon,
                              seed=args.seed, with_proprio=use_proprio,
                              skip=1 + start * K)
        val_it = D.batch_iterator(val, stats, tokens, micro_batch,
                                  args.window_size, args.action_horizon,
                                  seed=args.seed + 1, with_proprio=use_proprio,
                                  skip=4 * (start // args.val_every))
        print(f'\n   RESUMED at optimiser step {start} from {resume_path} '
              f'({len(leaves)} tensors, {len(history)} history points)')
    elif args.resume and any(q.name.isdigit() for q in save_dir.iterdir()):
        raise SystemExit(f'--resume: {save_dir} already has checkpoints but no '
                         f'resume_state.npz -- refusing to restart from step 0 '
                         f'on top of them')

    print(f'\n5. Training optimiser steps {start} -> {stop} of {args.steps}'
          f'{f" ({(stop - start) * K} micro steps)" if K > 1 else ""}...')
    for i in tqdm.tqdm(range(start * K, stop * K), dynamic_ncols=True):
        train_state, info = train_step(train_state, next(it))
        step = (i + 1) // K                      # optimiser steps, not micro
        if (i + 1) % (args.val_every * K) == 0:
            tr = float(jax.device_get(info)['loss'])
            vs = [float(jax.device_get(val_step(train_state, next(val_it)))['loss'])
                  for _ in range(4)]
            history.append({'step': step, 'train_loss': tr,
                            'val_loss': float(np.mean(vs))})
            write_history(history, save_dir, meta)
            print(f'   step {step:6d}  train_loss {tr:.4f}  '
                  f'val_loss {np.mean(vs):.4f}', flush=True)
        if (i + 1) % (args.save_every * K) == 0:
            train_state.model.save_pretrained(step=step - 1,
                                              checkpoint_path=str(save_dir))
            print(f'   saved checkpoint at step {step} -> {save_dir}', flush=True)
        if (args.time_limit_min is not None and (i + 1) % K == 0 and step < stop
                and time.time() - _T0 > 60 * args.time_limit_min):
            print(f'   time limit {args.time_limit_min} min reached at step {step}')
            stop = step                          # an optimiser-step boundary
            break

    if stop < args.steps:
        if start < stop:
            save_resume(train_state, stop)
            write_history(history, save_dir, meta)
        print(f'\nSegment done at optimiser step {max(start, stop)} of '
              f'{args.steps}. Resume state -> {resume_path}')
        return

    # The periodic save only fires on multiples of save_every, so without this
    # the last (steps % save_every) optimiser steps were trained and then
    # thrown away -- a 12600-step run was evaluated at step 12000.
    if start < stop and args.steps % args.save_every != 0:
        train_state.model.save_pretrained(step=args.steps - 1,
                                          checkpoint_path=str(save_dir))
        print(f'   saved final checkpoint at step {args.steps} -> {save_dir}')
    if resume_path.exists():
        resume_path.unlink()
    write_history(history, save_dir, meta)
    if history:
        best = min(history, key=lambda h: h['val_loss'])
        print(f"\nBest val {best['val_loss']:.4f} at step {best['step']} "
              f"(final {history[-1]['val_loss']:.4f})")
        if best['step'] != history[-1]['step']:
            print('   NOTE: the last checkpoint is not the best-val one.')
    print(f'Loss curve -> {save_dir / "loss_curve.png"}')
    print(f'\nDone. Checkpoints in {save_dir}')


if __name__ == '__main__':
    main()
