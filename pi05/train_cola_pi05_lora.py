"""Finetune CoLA on pi0.5 on the hidden-marker handover.

  * Budget: CoLA's, 20,250 steps at batch 128 (150 epochs); both arms per
    example, losses summed.
  * pi0.5: openpi's LoRA recipe (LoRA on both Gemma experts, rest of the LLM
    frozen in bfloat16; AdamW(0.9, 0.95), clip 1.0, warmup-cosine
    2.5e-5 -> 2.5e-6).
  * CoLA's channel: Adam, 1e-4 cosine-decayed to 1e-5.

Batches are split into micro-batches with gradient accumulation. Training runs
in time-limited segments (--time-limit-min) that resume exactly.
"""

import argparse
import dataclasses
import functools
import json
import os
import pathlib
import shutil
import sys
import time

_T0 = time.time()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import flax.nnx as nnx          # noqa: E402
import flax.traverse_util as tu  # noqa: E402
import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
import numpy as np              # noqa: E402
import optax                    # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

import openpi.models.model as _model                  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils           # noqa: E402

import cola_pi05_lora as C      # noqa: E402
import pi05_data as D           # noqa: E402

# pi0.5 base weights in openpi's download cache (~/.cache/openpi unless OPENPI_DATA_HOME is set).
PI05_BASE = (pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
             / "openpi-assets/checkpoints/pi05_base/params")


def build_model(config: C.ColaPi05Config, seed: int = 0, params_path=PI05_BASE):
    """pi0.5 base weights + fresh LoRA and CoLA channel; frozen weights in bfloat16."""
    abstract = nnx.eval_shape(config.create, jax.random.key(seed))
    graphdef, abstract_state = nnx.split(abstract)
    ref = abstract_state.to_pure_dict()
    flat_ref = tu.flatten_dict(ref, sep="/")

    loaded = tu.flatten_dict(_model.restore_params(params_path, restore_type=np.ndarray), sep="/")
    missing = [k for k in flat_ref if k not in loaded]
    unexpected = [k for k in loaded if k not in flat_ref]
    bad = [k for k in missing if not any(s in k for s in ("lora", "encoder_", "decoder_"))]
    assert not bad, f"pretrained checkpoint lacks non-LoRA/non-CoLA params: {bad[:5]}"
    print(f"   pi05_base: {len(flat_ref) - len(missing)} tensors loaded, {len(missing)} new "
          f"(LoRA + CoLA channel), {len(unexpected)} checkpoint tensors unused")

    # Only the new tensors need a real initialiser. Everything else in this jit is
    # dead code that XLA prunes, so the full fp32 model is never materialised.
    missing_set = set(missing)

    @jax.jit
    def init_missing(rng):
        flat = tu.flatten_dict(nnx.state(config.create(rng)).to_pure_dict(), sep="/")
        return {k: v for k, v in flat.items() if k in missing_set}

    fresh = init_missing(jax.random.key(seed))
    freeze = config.get_freeze_filter()
    frozen_paths = {"/".join(str(p) for p in path) for path in _flat_paths(abstract_state.filter(freeze))}

    flat = {}
    for k, spec in flat_ref.items():
        if k in fresh:
            flat[k] = fresh[k]
        else:
            dtype = jnp.bfloat16 if k in frozen_paths else spec.dtype
            flat[k] = jnp.asarray(loaded[k], dtype=dtype)
    del loaded
    abstract_state.replace_by_pure_dict(tu.unflatten_dict(flat, sep="/"))
    return nnx.merge(graphdef, abstract_state)


def _flat_paths(state):
    fs = state.flat_state()
    return [path for path, _ in (fs.items() if hasattr(fs, "items") else fs)]


def _flat_items(state):
    fs = state.flat_state()
    return list(fs.items() if hasattr(fs, "items") else fs)


def split_model(model, config):
    trainable_filter = nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))
    graphdef, trainable, frozen = nnx.split(model, trainable_filter, ...)
    return graphdef, trainable, frozen


def param_report(trainable, frozen):
    def count(state, pred=lambda p: True):
        return int(sum(v.value.size for p, v in _flat_items(state) if pred(p)))
    cola = count(trainable, C.is_cola_param)
    lora = count(trainable, lambda p: any("lora" in str(k) for k in p))
    img = count(trainable, lambda p: "img" in [str(k) for k in p])
    total_t = count(trainable)
    print(f"   trainable {total_t:,}: CoLA channel {cola:,} | LoRA {lora:,} | SigLIP {img:,} | "
          f"other {total_t - cola - lora - img:,}")
    print(f"   frozen (bfloat16) {count(frozen):,}")
    return {"trainable": total_t, "cola_channel": cola, "lora": lora, "siglip": img, "frozen": count(frozen)}


def make_optimizer(trainable, total_steps):
    # openpi's warmup (1000 steps), shortened only for very short runs.
    warmup = min(1000, max(1, total_steps // 10))
    pi_lr = optax.warmup_cosine_decay_schedule(
        init_value=2.5e-5 / (warmup + 1), peak_value=2.5e-5, warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1), end_value=2.5e-6)
    cola_lr = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=total_steps, alpha=0.1)
    labels = jax.tree_util.tree_map_with_path(lambda path, _: "cola" if C.is_cola_param(path) else "pi", trainable)
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"pi": optax.adamw(pi_lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10),
             "cola": optax.adam(cola_lr)},
            labels,
        ),
    ), {"pi": pi_lr, "cola": cola_lr}


def to_device(np_batch, sharding=None):
    def obs(d):
        o = _model.Observation.from_dict({
            "image": {k: v for k, v in d["image"].items()},
            "image_mask": d["image_mask"], "state": d["state"],
            "tokenized_prompt": d["tokenized_prompt"], "tokenized_prompt_mask": d["tokenized_prompt_mask"]})
        return o
    b = {"obs_a": obs(np_batch["obs_a"]), "obs_b": obs(np_batch["obs_b"]),
         "actions_a": np_batch["actions_a"], "actions_b": np_batch["actions_b"]}
    put = (lambda x: jax.device_put(x, sharding)) if sharding is not None else jnp.asarray  # noqa: E731
    return jax.tree.map(put, b)


def make_steps(graphdef, use_messages, arm="both"):
    def loss_fn(trainable, frozen, batch, rng):
        model = nnx.merge(graphdef, trainable, frozen)
        if arm != "both":
            # Decentralised baseline: one independent policy for this arm, no channel.
            loss = model.compute_arm_loss(rng, batch[f"obs_{arm}"], batch[f"actions_{arm}"])
            zero = jnp.zeros(())
            return loss, {"loss": loss, "loss_a": loss if arm == "a" else zero,
                          "loss_b": loss if arm == "b" else zero,
                          "msg_norm_a": zero, "msg_norm_b": zero,
                          "cond_norm_a": zero, "cond_norm_b": zero}
        la, lb, info = model.compute_cola_loss(rng, batch["obs_a"], batch["obs_b"],
                                               batch["actions_a"], batch["actions_b"], use_messages=use_messages)
        return la + lb, {"loss": la + lb, "loss_a": la, "loss_b": lb, **info}

    @jax.jit
    def eval_loss(trainable, frozen, batch, rng):
        return loss_fn(trainable, frozen, batch, rng)

    @functools.partial(jax.jit, donate_argnums=(2,))
    def accum(trainable, frozen, grads_acc, batch, rng):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable, frozen, batch, rng)
        grads_acc = grads if grads_acc is None else jax.tree.map(jnp.add, grads_acc, grads)
        return grads_acc, aux

    return loss_fn, eval_loss, accum


def make_apply(tx):
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2))
    def apply(trainable, opt_state, grads_acc, k):
        grads = jax.tree.map(lambda g: g / k, grads_acc)
        updates, opt_state = tx.update(grads, opt_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        cola_g = [g for p, g in jax.tree_util.tree_leaves_with_path(grads) if C.is_cola_param(p)]
        return trainable, opt_state, {"grad_norm": optax.global_norm(grads),
                                      "grad_norm_cola": optax.global_norm(cola_g)}
    return apply


def save_tree(path: pathlib.Path, tree):
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    with ocp.StandardCheckpointer() as ck:
        ck.save(tmp, tree)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def restore_tree(path: pathlib.Path, target):
    with ocp.StandardCheckpointer() as ck:
        return ck.restore(path, target)


def pure(state):
    return state.to_pure_dict() if hasattr(state, "to_pure_dict") else state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=20250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--no-messages", action="store_true", help="CoLA's L0 control: channel zeroed")
    ap.add_argument("--arm", choices=("both", "a", "b"), default="both",
                    help="'a'/'b' trains ONE independent pi0.5 policy on that arm only, with no "
                         "channel: the decentralised baseline. 'both' is CoLA's two-arm model.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--save-every-min", type=float, default=45.0)
    ap.add_argument("--time-limit-min", type=float, default=None)
    ap.add_argument("--stop-at", type=int, default=None)
    ap.add_argument("--profile", type=int, default=0, help="run N steps, report s/step, save nothing")
    ap.add_argument("--cache", default=str(D.CACHE))
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    final_dir, resume_dir = run / "final_params", run / "resume_state"
    if final_dir.exists() and not resume_dir.exists() and not args.profile:
        print(f"{run} already complete (final_params present, no resume state) -- nothing to do")
        return
    run.mkdir(parents=True, exist_ok=True)
    assert args.batch_size % args.micro_batch == 0
    K = args.batch_size // args.micro_batch
    use_messages = not args.no_messages

    print(f"devices: {jax.devices()}")
    D.build_cache(cache=args.cache)
    train = D.MarkerSplit("train", args.cache)
    val = D.MarkerSplit("val", args.cache)
    spe = train.steps_per_epoch(args.batch_size)
    print(f"train {len(train)} chunk-starts, {spe} steps/epoch; run = {args.steps} steps "
          f"({args.steps / spe:.1f} epochs) at batch {args.batch_size} = {K} x {args.micro_batch}")

    config = C.ColaPi05Config()
    t = time.time()
    model = build_model(config, args.seed)
    graphdef, trainable, frozen = split_model(model, config)
    del model
    counts = param_report(trainable, frozen)
    print(f"   model built in {time.time() - t:.0f}s")

    tx, lrs = make_optimizer(trainable, args.steps)
    opt_state = tx.init(trainable)
    _, eval_loss, accum = make_steps(graphdef, use_messages, args.arm)
    apply = make_apply(tx)

    start = 0
    if resume_dir.exists() and not args.profile:
        meta = json.loads((resume_dir / "meta.json").read_text())
        assert meta["args"]["batch_size"] == args.batch_size and meta["args"]["steps"] == args.steps \
            and meta["args"]["no_messages"] == args.no_messages \
            and meta["args"].get("arm", "both") == args.arm, f"resume state from different run: {meta['args']}"
        leaves, treedef = jax.tree.flatten(opt_state)
        restored = restore_tree(resume_dir / "state", {"trainable": pure(trainable), "opt_leaves": leaves})
        trainable.replace_by_pure_dict(restored["trainable"])
        opt_state = jax.tree.unflatten(treedef, restored["opt_leaves"])
        start = meta["step"]
        print(f"   resumed at step {start} from {resume_dir}")

    meta_run = {"args": vars(args), "config": dataclasses.asdict(config), "params": counts,
                "prompts": D.PROMPT, "arm": args.arm,
                "use_messages": use_messages and args.arm == "both", "pi05_base": str(PI05_BASE),
                "steps_per_epoch": spe, "micro_batches": K}
    (run / "run_meta.json").write_text(json.dumps(meta_run, indent=2, default=str))
    shutil.copy(D.norm_stats_path(args.cache), run / "norm_stats.json")

    hist_path = run / "history.jsonl"
    base_rng = jax.random.key(args.seed)
    stop = args.stop_at or args.steps
    if args.profile:
        stop = start + args.profile
    last_save = time.time()
    step_times = []

    def save_resume(step):
        save_tree(resume_dir / "state", {"trainable": pure(trainable), "opt_leaves": jax.tree.leaves(opt_state)})
        (resume_dir / "meta.json").write_text(json.dumps({"step": step, "args": vars(args)}))
        print(f"   resume state saved at step {step}", flush=True)

    val_starts = val.starts[np.linspace(0, len(val.starts) - 1, args.val_batches * args.micro_batch).astype(int)]
    for step in range(start, stop):
        t0 = time.time()
        starts = train.indices_for_step(step, args.batch_size, args.seed)
        rng = jax.random.fold_in(base_rng, step)
        grads_acc, logs = None, []
        for k in range(K):
            mb = train.batch(starts[k * args.micro_batch:(k + 1) * args.micro_batch])
            grads_acc, aux = accum(trainable, frozen, grads_acc, to_device(mb), jax.random.fold_in(rng, k))
            logs.append(aux)
        trainable, opt_state, ginfo = apply(trainable, opt_state, grads_acc, float(K))
        del grads_acc

        if step % args.log_every == 0 or step == stop - 1 or args.profile:
            rec = {k: float(np.mean([float(l[k]) for l in logs])) for k in logs[0]}
            rec.update({k: float(v) for k, v in ginfo.items()})
            rec.update({"step": step, "lr_pi": float(lrs["pi"](step)), "lr_cola": float(lrs["cola"](step)),
                        "s_per_step": time.time() - t0})
            print(f"step {step:6d} | loss {rec['loss']:.4f} (A {rec['loss_a']:.4f} B {rec['loss_b']:.4f}) | "
                  f"grad {rec['grad_norm']:.3f} cola {rec['grad_norm_cola']:.2e} | msg {rec['msg_norm_a']:.2f}/"
                  f"{rec['msg_norm_b']:.2f} cond {rec['cond_norm_a']:.3f}/{rec['cond_norm_b']:.3f} | "
                  f"{rec['s_per_step']:.1f}s", flush=True)
            if not args.profile:
                with open(hist_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        step_times.append(time.time() - t0)

        if not args.profile and ((step + 1) % args.val_every == 0 or step + 1 == args.steps):
            vl = []
            for k in range(args.val_batches):
                mb = val.batch(val_starts[k * args.micro_batch:(k + 1) * args.micro_batch])
                _, aux = eval_loss(trainable, frozen, to_device(mb), jax.random.key(1000 + k))
                vl.append((float(aux["loss_a"]), float(aux["loss_b"])))
            va, vb = np.mean(vl, axis=0)
            print(f"   val @ {step + 1}: A {va:.4f} B {vb:.4f}", flush=True)
            with open(hist_path, "a") as f:
                f.write(json.dumps({"step": step + 1, "val_loss_a": va, "val_loss_b": vb}) + "\n")

        done = step + 1
        if args.profile:
            continue
        over = args.time_limit_min is not None and (time.time() - _T0) / 60 > args.time_limit_min
        if done < args.steps and (over or done == stop or (time.time() - last_save) / 60 > args.save_every_min):
            save_resume(done)
            last_save = time.time()
            if over or done == stop:
                print(f"segment ends at step {done} ({(time.time() - _T0) / 60:.1f} min)")
                return

    if args.profile:
        warm = step_times[2:] or step_times
        print(f"PROFILE micro-batch {args.micro_batch} x {K}: {np.mean(warm):.2f} s/step "
              f"(first {step_times[0]:.1f}s incl. compile) -> {args.steps} steps = "
              f"{np.mean(warm) * args.steps / 3600:.1f} GPU-h")
        return

    save_tree(final_dir, pure(trainable))
    (run / "final_meta.json").write_text(json.dumps({"step": args.steps, **meta_run}, indent=2, default=str))
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    print(f"finished {args.steps} steps -> {final_dir}")


if __name__ == "__main__":
    main()
