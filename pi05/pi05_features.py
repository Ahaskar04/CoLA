"""Frozen pi0.5 as a per-frame feature function for the evaluator.

Same weights, prompt and pooling as extract_pi05_features.py.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import flax.nnx as nnx      # noqa: E402
import jax                  # noqa: E402
import jax.numpy as jnp     # noqa: E402
import numpy as np          # noqa: E402

import openpi.models.model as _model            # noqa: E402
import openpi.models.pi0 as _pi0                # noqa: E402
import openpi.models.pi0_config as _pi0_config  # noqa: E402
import openpi.models.tokenizer as _tokenizer    # noqa: E402

import pi05_data as D                           # noqa: E402
import train_cola_pi05_lora as T                # noqa: E402
from extract_pi05_features import MAX_TOKEN_LEN, PROMPT   # noqa: E402


class Pi05Features:
    def __init__(self, cache=None, seed=0):
        cache = cache or D.CACHE
        config = _pi0_config.Pi0Config(pi05=True, action_horizon=D.CHUNK, max_token_len=MAX_TOKEN_LEN)
        model = T.build_model(config)
        graphdef, state = nnx.split(model)
        del model
        self.state = state
        self.tok = _tokenizer.PaligemmaTokenizer(MAX_TOKEN_LEN)
        self.stats = D.load_norm_stats(cache)
        self.action_dim = config.action_dim

        @jax.jit
        def prefix(st, obs):
            m = nnx.merge(graphdef, st)
            obs = _model.preprocess_observation(None, obs, train=False, image_keys=("wrist_0_rgb",))
            tokens, mask, ar = m.embed_prefix(obs)
            attn = _pi0.make_attn_mask(mask, ar)
            positions = jnp.cumsum(mask, axis=1) - 1
            (out, _), _ = m.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
            w = mask[..., None].astype(jnp.float32)
            return (out.astype(jnp.float32) * w).sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)

        self._prefix = prefix
        print("   frozen pi0.5 feature extractor ready (prompt: "
              f"{PROMPT!r}, state in prompt, 2048-d masked-mean prefix)")

    def features(self, image_256, state_raw, arm=None):
        """One 256x256 wrist frame + that arm's raw 7-d state -> (1, 2048)."""
        s = D.quantile_norm(np.asarray(state_raw, np.float32)[None],
                            self.stats[f"state_{arm}"] if arm else self.stats["state_a"])
        tok, mask = self.tok.tokenize(PROMPT, s[0])
        obs = _model.Observation.from_dict({
            "image": {"wrist_0_rgb": D.resize_224(np.asarray(image_256)[None])},
            "image_mask": {"wrist_0_rgb": np.ones(1, bool)},
            "state": np.zeros((1, self.action_dim), np.float32),
            "tokenized_prompt": tok[None].astype(np.int32),
            "tokenized_prompt_mask": mask[None].astype(bool)})
        return np.asarray(self._prefix(self.state, jax.tree.map(jnp.asarray, obs)))
