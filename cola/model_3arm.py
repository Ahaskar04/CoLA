"""CoLA three-arm model (A -> B -> C handover) on a frozen Octo-Base backbone.

Same adapters, diffusion U-Net head and DDIM sampler as the two-arm model,
with two changes:

- All-to-all messages: each arm sends one message and decodes both partners'.
  Head input is [self | dec(partner 1) | dec(partner 2)], partners in fixed
  ARMS order. The order is part of the checkpoint format.
- One decoder per ordered pair: decoders[x][p] is how arm x reads arm p.
"""

import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from octo.model.octo_model import OctoModel


# Octo-Base readout_action width (Octo-Small is 384).
FEATURE_DIM = 768
MESSAGE_DIM = 64
ACTION_DIM = 7   # 6 joints + gripper
CHUNK_SIZE = 10  # actions predicted per query

STATE_DIM = 7      # 6 joint positions + gripper
PROPRIO_DIM = 64   # embedded state width

# Gripper column in an action vector (columns 0-5 are joints).
GRIPPER_IDX = 6

# A picks up the box and hands it to B; B turns 180 deg and hands it to C.
# This order fixes how decoded messages are concatenated (see PARTNERS).
ARMS = ('a', 'b', 'c')
N_ARMS = len(ARMS)

# Each arm's partners in ARMS order: B reads A, then C.
PARTNERS: Dict[str, Tuple[str, ...]] = {
    arm: tuple(o for o in ARMS if o != arm) for arm in ARMS
}


class MessageEncoder(nn.Module):
    """Compress an arm's self-representation into a message."""
    message_dim: int = MESSAGE_DIM

    @nn.compact
    def __call__(self, features):
        x = nn.Dense(96)(features)
        x = nn.relu(x)
        message = nn.Dense(self.message_dim)(x)
        return message


class MessageDecoder(nn.Module):
    """Expand a partner's 64-dim message back to 768-dim feature space."""
    feature_dim: int = FEATURE_DIM

    @nn.compact
    def __call__(self, message):
        x = nn.Dense(96)(message)
        x = nn.relu(x)
        decoded = nn.Dense(self.feature_dim)(x)
        return decoded


class ProprioEncoder(nn.Module):
    """Embed the 7-d joint state so it isn't swamped by 768 vision dims."""
    proprio_dim: int = PROPRIO_DIM

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(64)(state)
        x = nn.relu(x)
        return nn.Dense(self.proprio_dim)(x)


class CoordinationHead(nn.Module):
    """MLP head: [own features | decoded partners] -> action chunk.

    With split_gripper, joints stay tanh-bounded and the gripper column is a
    raw logit (for BCE); apply a sigmoid or threshold at 0 before actuating.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    split_gripper: bool = False

    @nn.compact
    def __call__(self, combined_features):
        x = nn.Dense(128)(combined_features)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.chunk_size * self.action_dim)(x)
        x = x.reshape((-1, self.chunk_size, self.action_dim))
        if not self.split_gripper:
            return nn.tanh(x)
        joints = nn.tanh(x[..., :GRIPPER_IDX])
        gripper_logit = x[..., GRIPPER_IDX:]
        return jnp.concatenate([joints, gripper_logit], axis=-1)


# Diffusion action head
# Noise levels used in training; sampling takes fewer DDIM steps.
DIFFUSION_STEPS = 100
# Width of the sinusoidal noise-level embedding.
TIME_EMBED_DIM = 64


def cosine_alpha_bars(n_steps: int = DIFFUSION_STEPS) -> jnp.ndarray:
    """Cosine alpha-bar schedule (Nichol & Dhariwal): ~1 at t=0, ~0 at t=n_steps."""
    s = 0.008
    t = jnp.arange(n_steps + 1, dtype=jnp.float32) / n_steps
    f = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    ab = f / f[0]
    return jnp.clip(ab, 1e-4, 1.0)


def timestep_embedding(t, dim: int = TIME_EMBED_DIM):
    """Sinusoidal embedding of integer noise levels: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t.astype(jnp.float32)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def mish(x):
    """Mish activation, as in Chi et al.'s ConditionalUnet1D."""
    return x * jnp.tanh(nn.softplus(x))


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, the U-Net's basic unit."""
    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        # x: (B, T, C); the convolution runs along the chunk's time axis.
        x = nn.Conv(self.out_channels, kernel_size=(self.kernel_size,),
                    padding='SAME')(x)
        groups = min(self.n_groups, self.out_channels)
        x = nn.GroupNorm(num_groups=groups)(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning (per-channel scale and bias)."""
    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x, cond):
        residual = x
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(x)

        # cond -> (scale, bias), one pair per output channel.
        embed = nn.Dense(self.out_channels * 2)(mish(cond))
        scale = embed[:, None, :self.out_channels]
        bias = embed[:, None, self.out_channels:]
        out = scale * out + bias

        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(out)
        # 1x1 conv to match channels when the residual width changes.
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1,))(residual)
        return out + residual


class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net denoiser, after Chi et al. (Diffusion Policy).

    Convolutions run along the action chunk and FiLM conditions every block.
    down_dims is smaller than the reference (256, 512, 1024) because a
    10-step chunk can't take three downsamples.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    down_dims: Tuple[int, ...] = (128, 256)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, cond, noisy_joints, t):
        """cond: (B, D) observation; noisy_joints: (B, chunk, 6); t: (B,).

        Same body as the two-arm ConditionalUnet1D.
        """
        # Noise-level embedding and observation form one global conditioning
        # vector, as in the reference implementation.
        cond_g = jnp.concatenate([timestep_embedding(t), cond], axis=-1)
        cond_g = nn.Dense(256)(mish(nn.Dense(256)(cond_g)))

        x = noisy_joints                      # (B, chunk, 6)
        skips = []
        for dim in self.down_dims:
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            skips.append(x)
            # Stride-2 downsample in time (10 -> 5 -> 3 with two levels).
            x = nn.Conv(dim, kernel_size=(3,), strides=(2,), padding='SAME')(x)

        mid = self.down_dims[-1]
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond_g)
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond_g)

        for dim, skip in zip(reversed(self.down_dims), reversed(skips)):
            # Upsample, then trim to the skip's length (odd lengths don't double back).
            x = jnp.repeat(x, 2, axis=1)[:, :skip.shape[1]]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)

        n_joint = self.action_dim - 1
        eps = nn.Conv(n_joint, kernel_size=(1,))(x)

        # The binary gripper is predicted directly, outside the diffusion.
        g = nn.Dense(128)(cond)
        g = mish(g)
        g = nn.Dense(self.chunk_size)(g).reshape(-1, self.chunk_size, 1)
        return eps, g


class DiffusionHead(nn.Module):
    """Flat residual-MLP denoiser over the joint columns (kept for ablations).

    The binary gripper is not diffused; it gets a separate logit.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    hidden: int = 512

    @nn.compact
    def __call__(self, cond, noisy_joints, t):
        temb = timestep_embedding(t, TIME_EMBED_DIM)
        temb = nn.Dense(TIME_EMBED_DIM)(temb)
        temb = mish(temb)

        flat = noisy_joints.reshape((noisy_joints.shape[0], -1))
        x = jnp.concatenate([cond, flat, temb], axis=-1)

        x = nn.Dense(self.hidden)(x)
        x = mish(x)
        for _ in range(2):
            h = nn.Dense(self.hidden)(x)
            h = mish(h)
            h = nn.Dense(self.hidden)(h)
            x = x + h
            x = mish(x)

        eps = nn.Dense(self.chunk_size * (self.action_dim - 1))(x)
        eps = eps.reshape((-1, self.chunk_size, self.action_dim - 1))

        grip = nn.Dense(64)(cond)
        grip = mish(grip)
        grip = nn.Dense(self.chunk_size)(grip)
        grip = grip[:, :, None]

        return eps, grip


class COLAModel3Arm:
    """Three-arm CoLA: frozen Octo backbone, trainable per-arm adapters.

    Parameters (all trainable; Octo is not included):
        encoder_{a,b,c}       own self-representation -> 64-d message
        decoder_{x}_{p}       partner p's message -> 768-d, as read by x (6 total)
        action_head_{a,b,c}   [self | dec_p | dec_q] -> action chunk
        proprio_{a,b,c}       7-d joint state -> 64-d (use_proprio only)
    """

    def __init__(
        self,
        octo_checkpoint: str = "hf://rail-berkeley/octo-base",
        use_proprio: bool = False,
        use_overhead: bool = False,
        use_wrist: bool = True,
        split_gripper: bool = False,
        use_diffusion: bool = False,
        diffusion_unet: bool = False,
    ):
        """
        use_proprio:    add each arm's own joint state to its features.
        use_overhead:   add the fixed overhead camera's features to every arm.
        use_wrist:      use each arm's own wrist camera; False with use_overhead
                        gives an overhead-only model.
        split_gripper:  emit the gripper as a BCE logit (see CoordinationHead).
        use_diffusion:  diffusion head for the joints; the gripper stays a logit.
        diffusion_unet: 1D U-Net denoiser instead of the flat residual MLP.
        """
        print("Loading Octo backbone...")
        self.octo = OctoModel.load_pretrained(octo_checkpoint)
        print("Octo loaded (frozen)")

        # Same task prompt as the two-arm model.
        self.task = self.octo.create_tasks(texts=["coordinate with partner"])

        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.use_wrist = use_wrist
        if not (use_wrist or use_overhead):
            raise ValueError('CoLA needs a camera: set use_wrist, use_overhead or both.')
        self.split_gripper = split_gripper
        self.use_diffusion = use_diffusion
        self.diffusion_unet = diffusion_unet

        if use_diffusion:
            # Precomputed noise schedule.
            self.alpha_bars = cosine_alpha_bars(DIFFUSION_STEPS)
            # Advanced on every forward() so each control step samples fresh noise.
            self._sample_rng = jax.random.PRNGKey(0)

        # Adapters: one per arm, and one decoder per ordered pair.
        self.encoders = {a: MessageEncoder(message_dim=MESSAGE_DIM) for a in ARMS}
        # decoders[x][p] is "how arm x reads arm p's message".
        self.decoders = {
            x: {p: MessageDecoder(feature_dim=FEATURE_DIM) for p in PARTNERS[x]}
            for x in ARMS
        }
        if use_diffusion:
            Head = ConditionalUnet1D if diffusion_unet else DiffusionHead
            self.action_heads = {a: Head(action_dim=ACTION_DIM) for a in ARMS}
        else:
            self.action_heads = {
                a: CoordinationHead(action_dim=ACTION_DIM, split_gripper=split_gripper)
                for a in ARMS
            }

        rng = jax.random.PRNGKey(0)
        dummy_message = jnp.ones((1, MESSAGE_DIM))

        # Each arm's self-representation: its wrist and/or the overhead
        # features, plus its embedded state when enabled.
        self_dim = ((FEATURE_DIM if use_wrist else 0)
                    + (FEATURE_DIM if use_overhead else 0)
                    + (PROPRIO_DIM if use_proprio else 0))
        # The head input carries two decoded partner blocks.
        combined_dim = self_dim + (N_ARMS - 1) * FEATURE_DIM

        dummy_self = jnp.ones((1, self_dim))
        dummy_combined = jnp.ones((1, combined_dim))
        # Init shapes for the diffusion head: a noisy joint chunk and a noise level.
        dummy_noisy = jnp.zeros((1, CHUNK_SIZE, ACTION_DIM - 1))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)

        self.params = {}
        for a in ARMS:
            self.params[f'encoder_{a}'] = self.encoders[a].init(rng, dummy_self)
            for p in PARTNERS[a]:
                self.params[f'decoder_{a}_{p}'] = self.decoders[a][p].init(rng, dummy_message)
            self.params[f'action_head_{a}'] = (
                self.action_heads[a].init(rng, dummy_combined, dummy_noisy, dummy_t)
                if use_diffusion else
                self.action_heads[a].init(rng, dummy_combined)
            )

        if use_proprio:
            self.proprio_encoders = {a: ProprioEncoder(proprio_dim=PROPRIO_DIM) for a in ARMS}
            dummy_state = jnp.ones((1, STATE_DIM))
            for a in ARMS:
                self.params[f'proprio_{a}'] = self.proprio_encoders[a].init(rng, dummy_state)

        n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(self.params)))
        print(f"COLA 3-arm adapters initialized ({n_params:,} trainable params)")
        print(f"  arms: {ARMS} | topology: all-to-all "
              f"({N_ARMS} encoders, {N_ARMS * (N_ARMS - 1)} decoders)")
        print(f"  action chunking: chunk_size={CHUNK_SIZE}, action_dim={ACTION_DIM}")
        print(f"  self_dim={self_dim} -> combined_dim={combined_dim}")
        print(f"  proprioception: {'on' if use_proprio else 'OFF'}"
              f" | gripper head: {'logit (BCE)' if split_gripper else 'tanh (regression)'}")
        if use_diffusion:
            print(f"  denoiser: {'1D temporal U-Net (FiLM per block)' if diffusion_unet else 'flat residual MLP'}")

    # Shared plumbing

    def _self_repr(self, features, state, params, arm, features_o=None):
        """An arm's camera features (wrist, overhead or both) and its embedded state."""
        if self.use_overhead:
            if features_o is None:
                raise ValueError(
                    "use_overhead=True but no features_o was passed. Pass the "
                    "overhead features, or build the model with use_overhead=False."
                )
            features_o = jnp.asarray(features_o)
            features = (jnp.concatenate([features, features_o], axis=-1)
                        if self.use_wrist else features_o)
        if not self.use_proprio:
            return features
        if state is None:
            raise ValueError(
                f"use_proprio=True but no state was passed for arm {arm}. "
                f"Pass proprio_{arm}, or build the model with use_proprio=False."
            )
        embedded = self.proprio_encoders[arm].apply(
            params[f'proprio_{arm}'], jnp.asarray(state))
        return jnp.concatenate([features, embedded], axis=-1)

    def _combine(self, selves, params, use_messages: bool):
        """Per-arm [self | decoded partner p | decoded partner q].

        Partners follow PARTNERS order; changing it breaks existing checkpoints.
        """
        messages = {
            a: self.encoders[a].apply(params[f'encoder_{a}'], selves[a])
            for a in ARMS
        }
        if not use_messages:
            messages = {a: jnp.zeros_like(m) for a, m in messages.items()}

        combined = {}
        for a in ARMS:
            blocks = [selves[a]]
            for p in PARTNERS[a]:
                blocks.append(
                    self.decoders[a][p].apply(params[f'decoder_{a}_{p}'], messages[p])
                )
            combined[a] = jnp.concatenate(blocks, axis=-1)
        return combined

    def extract_octo_features(self, image: np.ndarray) -> jnp.ndarray:
        """Extract 768-dim features from the Octo-Base readout token (frozen)."""
        batch_size = image.shape[0]
        # Octo expects (batch, window, H, W, C)
        image_windowed = image[:, None, :, :, :]

        observations = {
            "image_primary": image_windowed,
            "timestep_pad_mask": np.ones((batch_size, 1), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((batch_size, 1), dtype=bool),
            },
        }

        task = jax.tree_util.tree_map(
            lambda x: np.broadcast_to(x, (batch_size, *x.shape[1:])) if hasattr(x, 'shape') else x,
            self.task,
        )

        transformer_outputs = self.octo.run_transformer(
            observations,
            task,
            observations["timestep_pad_mask"],
        )

        # (batch, FEATURE_DIM)
        readout_features = transformer_outputs["readout_action"].tokens[:, -1, 0, :]
        return jax.lax.stop_gradient(readout_features)

    # Forward passes

    def forward(
        self,
        images: Dict[str, np.ndarray],
        params=None,
        use_messages: bool = True,
        proprio: Dict[str, np.ndarray] = None,
        image_o: np.ndarray = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass from images, with all-to-all message passing.

        images: {'a': (B, H, W, 3), 'b': ..., 'c': ...} wrist views (use_wrist).
        proprio: {'a': (B, 7), ...} joint states (use_proprio).
        image_o: (B, H, W, 3) overhead view (use_overhead).
        use_messages=False zeros every message (no-message ablation).

        Returns {'a': (B, chunk, 7), 'b': ..., 'c': ...}.
        """
        if params is None:
            params = self.params
        proprio = proprio or {}

        features = {a: None for a in ARMS}
        if self.use_wrist:
            missing = [a for a in ARMS if a not in images]
            if missing:
                raise ValueError(f"forward() needs an image for every arm; missing {missing}")
            features = {a: self.extract_octo_features(images[a]) for a in ARMS}

        # Shared overhead view; required when the model uses it.
        features_o = None
        if self.use_overhead:
            if image_o is None:
                raise ValueError(
                    "use_overhead=True but forward() got no image_o. Render the "
                    "overhead camera and pass it, or build the model with "
                    "use_overhead=False."
                )
            features_o = self.extract_octo_features(image_o)

        selves = {
            a: self._self_repr(features[a], proprio.get(a), params, a, features_o)
            for a in ARMS
        }
        combined = self._combine(selves, params, use_messages)

        if self.use_diffusion:
            # Sample a chunk per arm. The RNG advances every call so control
            # steps don't reuse the same noise.
            keys = jax.random.split(self._sample_rng, N_ARMS + 1)
            self._sample_rng = keys[0]
            return {
                a: self.sample_actions(combined[a], params, a, keys[i + 1])
                for i, a in enumerate(ARMS)
            }

        return {
            a: self.action_heads[a].apply(params[f'action_head_{a}'], combined[a])
            for a in ARMS
        }

    def forward_from_features(
        self,
        features: Dict[str, np.ndarray],
        params=None,
        use_messages: bool = True,
        proprio: Dict[str, np.ndarray] = None,
        features_o: np.ndarray = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass from cached Octo features (used in training).

        features: {'a': (B, 768), 'b': ..., 'c': ...}; features_o: (B, 768)
        overhead view, required when the model uses it. With diffusion this
        returns the conditioning vectors, not actions.
        """
        if params is None:
            params = self.params
        proprio = proprio or {}

        selves = {
            a: self._self_repr(None if features[a] is None else jnp.asarray(features[a]),
                               proprio.get(a), params, a, features_o)
            for a in ARMS
        }
        combined = self._combine(selves, params, use_messages)

        if self.use_diffusion:
            return combined

        return {
            a: self.action_heads[a].apply(params[f'action_head_{a}'], combined[a])
            for a in ARMS
        }

    # Diffusion sampling

    def denoise(self, combined, noisy_joints, t, params, arm: str):
        """One denoiser call for one arm: (predicted noise, gripper logits)."""
        return self.action_heads[arm].apply(
            params[f'action_head_{arm}'], combined, noisy_joints, t)

    def sample_actions(self, combined, params, arm: str, rng, n_steps: int = None):
        """DDIM sampling (eta=0): noise -> action chunk in n_steps passes.

        Returns (B, chunk, ACTION_DIM): denoised joints with the gripper logit
        appended, the same layout as the deterministic head.
        """
        # 10 strided steps by default; COLA_DDIM_STEPS overrides it at eval time.
        if n_steps is None:
            n_steps = int(os.environ.get('COLA_DDIM_STEPS', '10'))
        ab = self.alpha_bars
        b = combined.shape[0]
        x = jax.random.normal(rng, (b, CHUNK_SIZE, ACTION_DIM - 1))

        ts = jnp.linspace(DIFFUSION_STEPS - 1, 0, n_steps).astype(jnp.int32)
        grip = None
        for i in range(n_steps):
            t = jnp.full((b,), ts[i])
            eps, grip = self.denoise(combined, x, t, params, arm)
            a_t = ab[ts[i]]
            # Clean action implied by the predicted noise.
            x0 = (x - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(a_t)
            x0 = jnp.clip(x0, -1.0, 1.0)   # joints are min/max normalised
            if i < n_steps - 1:
                a_prev = ab[ts[i + 1]]
                # DDIM update with eta=0: deterministic given the initial noise.
                x = jnp.sqrt(a_prev) * x0 + jnp.sqrt(1.0 - a_prev) * eps
            else:
                x = x0
        return jnp.concatenate([x, grip], axis=-1)


if __name__ == '__main__':
    # Shape smoke test with random inputs (no Octo download or dataset).
    print(f'ARMS   = {ARMS}')
    for a in ARMS:
        print(f'  {a} decodes {PARTNERS[a]}')

    self_dim = FEATURE_DIM * 2 + PROPRIO_DIM     # overhead + proprio
    combined_dim = self_dim + (N_ARMS - 1) * FEATURE_DIM
    print(f'\nself_dim={self_dim}  combined_dim={combined_dim}')

    rng = jax.random.PRNGKey(0)
    head = ConditionalUnet1D(action_dim=ACTION_DIM)
    cond = jnp.ones((4, combined_dim))
    noisy = jnp.zeros((4, CHUNK_SIZE, ACTION_DIM - 1))
    t = jnp.zeros((4,), dtype=jnp.int32)
    p = head.init(rng, cond, noisy, t)
    eps, grip = head.apply(p, cond, noisy, t)
    n = int(sum(x.size for x in jax.tree_util.tree_leaves(p)))
    print(f'ConditionalUnet1D: eps {eps.shape}, grip {grip.shape}, {n:,} params/arm')
    assert eps.shape == (4, CHUNK_SIZE, ACTION_DIM - 1)
    assert grip.shape == (4, CHUNK_SIZE, 1)
    print('OK')
