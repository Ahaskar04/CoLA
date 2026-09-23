"""Pre-extract FROZEN pi0.5 features for the hidden-marker handover.

The pi0.5 counterpart of CoLA's extract_features_h5.py. For every timestep of
every episode, in the manifest order CoLA's action/state caches were written
in, it runs pi0.5's PaliGemma prefix over that arm's own wrist image and prompt
and saves the masked mean of the final hidden states:

    {split}_features_a.npy  (N, 2048) float32
    {split}_features_b.npy  (N, 2048) float32

Row i lines up with row i of cache_marker_v1/{split}_actions_{a,b}.npy, which
the trainer asserts. pi0.5 is frozen and never runs again during training.

Prompt: pi0.5's native format, i.e. the task text plus the arm's own
discretised joint state, produced by openpi's own tokenizer. pi05 was trained
with the state in the prompt, so leaving it out would put the backbone
off-distribution. CoLA's own ProprioEncoder still receives the raw state
separately, exactly as in the Octo runs.

Images come from pi05_marker_cache (built by pi05_marker_data.py: the same
episodes, in the same order, resized to 224 with openpi's own resize_with_pad).
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENPI_DATA_HOME", "/scratch/users/ntu/ahaskar0/openpi_cache")

import flax.nnx as nnx      # noqa: E402
import jax                  # noqa: E402
import jax.numpy as jnp     # noqa: E402
import numpy as np          # noqa: E402

import openpi.models.model as _model             # noqa: E402
import openpi.models.pi0 as _pi0                 # noqa: E402
import openpi.models.pi0_config as _pi0_config   # noqa: E402
import openpi.models.tokenizer as _tokenizer     # noqa: E402

import pi05_marker_data as D                     # noqa: E402
import train_cola_pi05 as T                      # noqa: E402

# CoLA's own task prompt, used for both arms (cola_architecture.py:
# create_tasks(texts=["coordinate with partner"])). The arms are distinguished
# by which wrist camera and which state they see, as in the Octo runs.
PROMPT = "coordinate with partner"
FEAT_DIR = pathlib.Path("/scratch/users/ntu/ahaskar0/pi05_marker_features")
MAX_TOKEN_LEN = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(D.CACHE))
    ap.add_argument("--feat-dir", default=str(FEAT_DIR))
    ap.add_argument("--action-cache", default="/scratch/users/ntu/ahaskar0/v1/cola-research-scratchdata/cache_marker_v1",
                    help="CoLA's cache; row counts must match, as in extract_features_h5.py")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    feat_dir = pathlib.Path(args.feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    D.build_cache(cache=args.cache)

    # Plain pi0.5, no LoRA: every weight is the released pi05_base, frozen.
    config = _pi0_config.Pi0Config(pi05=True, action_horizon=D.CHUNK, max_token_len=MAX_TOKEN_LEN)
    t0 = time.time()
    model = T.build_model(config)
    graphdef, state = nnx.split(model)
    del model
    print(f"   frozen pi0.5 ready in {time.time() - t0:.0f}s "
          f"({sum(v.value.size for _, v in T._flat_items(state)):,} params, none trainable)")

    tok = _tokenizer.PaligemmaTokenizer(MAX_TOKEN_LEN)
    stats = D.load_norm_stats(args.cache)

    @jax.jit
    def prefix_features(st, obs):
        m = nnx.merge(graphdef, st)
        obs = _model.preprocess_observation(None, obs, train=False, image_keys=("wrist_0_rgb",))
        tokens, mask, ar_mask = m.embed_prefix(obs)
        attn = _pi0.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix_out, _), _ = m.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
        w = mask[..., None].astype(jnp.float32)
        return (prefix_out.astype(jnp.float32) * w).sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)

    cache = pathlib.Path(args.cache)
    for split in ("train", "val", "test"):
        n_actions = len(np.load(pathlib.Path(args.action_cache) / f"{split}_actions_a.npy", mmap_mode="r"))
        for arm in "ab":
            out_path = feat_dir / f"{split}_features_{arm}.npy"
            if out_path.exists():
                print(f"   [{split}/{arm}] already extracted, skipping")
                continue
            imgs = np.load(cache / f"{split}_img_{arm}.npy", mmap_mode="r")
            states = np.load(cache / f"{split}_state_{arm}.npy", mmap_mode="r")[:, :D.STATE_RAW]
            n = len(imgs)
            assert n == n_actions, f"{split}/{arm}: {n} frames vs {n_actions} actions -- order/cache mismatch"
            toks, masks = zip(*(tok.tokenize(PROMPT, s) for s in states))
            toks, masks = np.stack(toks).astype(np.int32), np.stack(masks).astype(bool)

            feats = np.zeros((n, 2048), np.float32)
            t1 = time.time()
            for i in range(0, n, args.batch_size):
                j = min(i + args.batch_size, n)
                obs = _model.Observation.from_dict({
                    "image": {"wrist_0_rgb": np.asarray(imgs[i:j])},
                    "image_mask": {"wrist_0_rgb": np.ones(j - i, bool)},
                    "state": np.zeros((j - i, config.action_dim), np.float32),
                    "tokenized_prompt": toks[i:j], "tokenized_prompt_mask": masks[i:j]})
                feats[i:j] = np.asarray(prefix_features(state, jax.tree.map(jnp.asarray, obs)))
            assert np.isfinite(feats).all(), f"{split}/{arm}: non-finite features"
            np.save(out_path, feats)
            print(f"   [{split}/{arm}] {feats.shape} in {time.time() - t1:.0f}s "
                  f"(|f| mean {np.linalg.norm(feats, axis=1).mean():.2f}) -> {out_path}")

    (feat_dir / "feature_provenance.json").write_text(json.dumps({
        "backbone": "pi05_base (frozen)", "prompt": PROMPT, "pooling": "masked mean of PaliGemma prefix",
        "feature_dim": 2048, "image_cache": str(cache), "action_cache": args.action_cache,
        "state_in_prompt": True, "max_token_len": MAX_TOKEN_LEN}, indent=2))
    print(f"done -> {feat_dir}")


if __name__ == "__main__":
    main()
