"""CoLA on a LoRA-finetuned pi0.5 backbone (two arms, 64-d message channel).

Same structure as the Octo-based model: each arm's encoder turns its pooled
PaliGemma prefix (2048-d masked mean) into a message, and the partner's
decoder adds it to pi0.5's action-expert conditioning (the adaRMS input,
alongside the flow-matching time). The decoder's last layer is
zero-initialised, so training starts from pretrained pi0.5.

Both arms share pi0.5's weights and differ only in their prompt; each sees
only its own wrist camera and joint state. The prefix runs first (filling the
KV cache), then the suffix. Subclasses openpi's Pi0; openpi is not modified.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
import openpi.models.gemma as _gemma

# One image slot per arm: its own wrist camera. "wrist" in the name makes
# openpi skip crop/rotate augmentation.
IMAGE_KEY = "wrist_0_rgb"
MESSAGE_DIM = 64      # CoLA's MESSAGE_DIM
MESSAGE_HIDDEN = 96   # CoLA's encoder/decoder hidden width


class MessageEncoder(nnx.Module):
    """Own prefix representation -> 64-d message. CoLA's MessageEncoder."""

    def __init__(self, in_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, MESSAGE_HIDDEN, rngs=rngs)
        self.fc2 = nnx.Linear(MESSAGE_HIDDEN, MESSAGE_DIM, rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.relu(self.fc1(x)))


class MessageDecoder(nnx.Module):
    """Partner's 64-d message -> action-expert conditioning. CoLA's MessageDecoder.

    The output layer is zero-initialised, so training starts from pretrained pi0.5.
    """

    def __init__(self, out_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(MESSAGE_DIM, MESSAGE_HIDDEN, rngs=rngs)
        self.fc2 = nnx.Linear(MESSAGE_HIDDEN, out_dim, kernel_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, m):
        return self.fc2(nnx.relu(self.fc1(m)))


@dataclasses.dataclass(frozen=True)
class ColaPi05Config(_pi0_config.Pi0Config):
    """pi0.5 with CoLA's channel. Defaults: openpi's LoRA variants, CoLA's 10-step chunk."""

    pi05: bool = True
    paligemma_variant: _gemma.Variant = "gemma_2b_lora"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"
    action_horizon: int = 10
    max_token_len: int = 64

    @override
    def create(self, rng) -> "ColaPi05":
        return ColaPi05(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        from openpi.shared import array_typing as at

        with at.disable_typechecking():
            obs = _model.Observation(
                images={IMAGE_KEY: image_spec},
                image_masks={IMAGE_KEY: mask_spec},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        act = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return obs, act


COLA_MODULES = ("encoder_a", "encoder_b", "decoder_a", "decoder_b")


def _key(k) -> str:
    return str(getattr(k, "key", getattr(k, "name", k)))


def is_cola_param(path, modules=COLA_MODULES) -> bool:
    """Parameters that belong to CoLA's channel rather than to pi0.5.

    Matched by exact module name; a prefix match would also catch SigLIP's
    `encoder_norm`.
    """
    return any(_key(k) in modules for k in path)


class ColaPi05(_pi0.Pi0):
    def __init__(self, config: ColaPi05Config, rngs: nnx.Rngs):
        assert config.pi05, "CoLA's channel enters through pi0.5's adaRMS conditioning"
        super().__init__(config, rngs)
        width_vlm = _gemma.get_config(config.paligemma_variant).width
        width_expert = _gemma.get_config(config.action_expert_variant).width
        self.encoder_a = MessageEncoder(width_vlm, rngs)
        self.encoder_b = MessageEncoder(width_vlm, rngs)
        self.decoder_a = MessageDecoder(width_expert, rngs)  # A decodes B's message
        self.decoder_b = MessageDecoder(width_expert, rngs)  # B decodes A's message

    def _prefix(self, obs: _model.Observation):
        """Run PaliGemma over [image | prompt+state]; return its KV cache and a pooled vector."""
        obs = _model.preprocess_observation(None, obs, train=False, image_keys=(IMAGE_KEY,))
        tokens, mask, ar_mask = self.embed_prefix(obs)
        attn = _pi0.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
        m = mask[..., None].astype(jnp.float32)
        pooled = (prefix_out.astype(jnp.float32) * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)
        return obs, mask, kv_cache, pooled

    def _messages(self, pooled_a, pooled_b, use_messages: bool):
        """CoLA's symmetric exchange. Returns (cond added to A's expert, cond added to B's)."""
        msg_a = self.encoder_a(pooled_a)
        msg_b = self.encoder_b(pooled_b)
        if not use_messages:
            msg_a = jnp.zeros_like(msg_a)
            msg_b = jnp.zeros_like(msg_b)
        return self.decoder_a(msg_b), self.decoder_b(msg_a), msg_a, msg_b

    def _velocity(self, obs, prefix_mask, kv_cache, x_t, time, msg_cond):
        """One action-expert pass over the noisy chunk, attending to the cached prefix."""
        suffix_tokens, suffix_mask, suffix_ar_mask, time_cond = self.embed_suffix(obs, x_t, time)
        cond = time_cond + msg_cond.astype(time_cond.dtype)
        suffix_attn = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

    @staticmethod
    def _stack(obs_a, obs_b):
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), obs_a, obs_b)

    def compute_cola_loss(self, rng, obs_a, obs_b, actions_a, actions_b, *, use_messages: bool = True):
        """pi0.5's flow-matching loss for both arms, with CoLA's channel in between.

        Both arms go through one batched prefix pass and one batched suffix pass.
        Returns (per-arm mean losses, message diagnostics).
        """
        n = actions_a.shape[0]
        obs, prefix_mask, kv_cache, pooled = self._prefix(self._stack(obs_a, obs_b))
        cond_a, cond_b, msg_a, msg_b = self._messages(pooled[:n], pooled[n:], use_messages)

        actions = jnp.concatenate([actions_a, actions_b], axis=0)
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:1]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        v_t = self._velocity(obs, prefix_mask, kv_cache, x_t, time, jnp.concatenate([cond_a, cond_b], axis=0))
        per = jnp.mean(jnp.square(v_t - u_t), axis=(1, 2))
        info = {
            "msg_norm_a": jnp.linalg.norm(msg_a, axis=-1).mean(),
            "msg_norm_b": jnp.linalg.norm(msg_b, axis=-1).mean(),
            "cond_norm_a": jnp.linalg.norm(cond_a, axis=-1).mean(),
            "cond_norm_b": jnp.linalg.norm(cond_b, axis=-1).mean(),
        }
        return per[:n].mean(), per[n:].mean(), info

    def compute_arm_loss(self, rng, obs, actions):
        """pi0.5's own flow-matching loss for one arm, with no message (decentralised baseline)."""
        obs, prefix_mask, kv_cache, _ = self._prefix(obs)
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:1]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        zero = jnp.zeros((actions.shape[0], self.action_in_proj.out_features))
        v_t = self._velocity(obs, prefix_mask, kv_cache, x_t, time, zero)
        return jnp.mean(jnp.square(v_t - u_t))

    def sample_arm_actions(self, rng, obs, *, num_steps: int = 10):
        """One arm's chunk, no message. Mirrors pi0.5's own sampler."""
        n = obs.state.shape[0]
        obs, prefix_mask, kv_cache, _ = self._prefix(obs)
        zero = jnp.zeros((n, self.action_in_proj.out_features))
        dt = -1.0 / num_steps
        x_t = jax.random.normal(rng, (n, self.action_horizon, self.action_dim))
        time = 1.0
        for _ in range(num_steps):
            x_t = x_t + dt * self._velocity(obs, prefix_mask, kv_cache, x_t, jnp.full((n,), time), zero)
            time = time + dt
        return x_t

    def sample_cola_actions(self, rng, obs_a, obs_b, *, num_steps: int = 10, use_messages: bool = True):
        """Both arms' chunks for one control step. Messages are computed once from
        this step's observations and held fixed across the denoising steps."""
        n = obs_a.state.shape[0]
        obs, prefix_mask, kv_cache, pooled = self._prefix(self._stack(obs_a, obs_b))
        cond_a, cond_b, _, _ = self._messages(pooled[:n], pooled[n:], use_messages)
        cond = jnp.concatenate([cond_a, cond_b], axis=0)

        dt = -1.0 / num_steps
        x_t = jax.random.normal(rng, (2 * n, self.action_horizon, self.action_dim))
        time = 1.0
        for _ in range(num_steps):
            v_t = self._velocity(obs, prefix_mask, kv_cache, x_t, jnp.full((2 * n,), time), cond)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t[:n], x_t[n:]

    # pi0's single-observation entry points don't apply to a two-arm model.
    @override
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("use compute_cola_loss(rng, obs_a, obs_b, actions_a, actions_b)")

    @override
    def sample_actions(self, *args, **kwargs):
        raise NotImplementedError("use sample_cola_actions(rng, obs_a, obs_b)")

    def joint_reference_loss(self, rng, obs, actions):
        """pi0.5's own single joint prefix+suffix pass (Pi0.compute_loss) on this
        model's one-image observation, with no message. The smoke test checks the
        split prefix/KV/suffix path reproduces it."""
        obs = _model.preprocess_observation(None, obs, train=False, image_keys=(IMAGE_KEY,))
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:1]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        p_tok, p_mask, p_ar = self.embed_prefix(obs)
        s_tok, s_mask, s_ar, cond = self.embed_suffix(obs, x_t, time)
        mask = jnp.concatenate([p_mask, s_mask], axis=1)
        attn = _pi0.make_attn_mask(mask, jnp.concatenate([p_ar, s_ar], axis=0))
        positions = jnp.cumsum(mask, axis=1) - 1
        (_, s_out), _ = self.PaliGemma.llm([p_tok, s_tok], mask=attn, positions=positions, adarms_cond=[None, cond])
        v_t = self.action_out_proj(s_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=(1, 2))

    def split_reference_loss(self, rng, obs, actions):
        """The same loss through this model's prefix -> KV cache -> suffix path, zero message."""
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:1]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        obs, p_mask, kv, _ = self._prefix(obs)
        zero = jnp.zeros((actions.shape[0], self.action_in_proj.out_features))
        v_t = self._velocity(obs, p_mask, kv, x_t, time, zero)
        return jnp.mean(jnp.square(v_t - u_t), axis=(1, 2))
