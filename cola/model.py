"""CoLA two-arm model: a frozen Octo-Base backbone plus trainable adapters.

Each arm encodes its features into a message, decodes its partner's message,
and predicts an action chunk from both.
"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from octo.model.octo_model import OctoModel
from typing import Tuple
import numpy as np


# Octo-Base readout_action width (Octo-Small is 384).
FEATURE_DIM = 768
MESSAGE_DIM = 64
ACTION_DIM = 7   # 6 joints + gripper
CHUNK_SIZE = 10  # actions predicted per query

STATE_DIM = 7      # 6 joint positions + gripper
PROPRIO_DIM = 64   # embedded state width

# Gripper column in an action vector (columns 0-5 are joints).
GRIPPER_IDX = 6


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
    """MLP head: [own features | decoded message] -> action chunk.

    With split_gripper, joints stay tanh-bounded and the gripper column is a
    raw logit (for BCE); apply a sigmoid or threshold at 0 before actuating.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    split_gripper: bool = False

    @nn.compact
    def __call__(self, combined_features):
        # combined -> 128 -> 64 -> chunk_size * action_dim, then reshape.
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
    10-step chunk can't take three downsamples. Set it via
    COLAModel(unet_dims=...).
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    down_dims: tuple = (128, 256)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, combined_features, noisy_joints, t):
        # Noise-level embedding and observation form one global conditioning
        # vector, as in the reference implementation.
        cond = jnp.concatenate([timestep_embedding(t), combined_features], axis=-1)
        cond = nn.Dense(256)(mish(nn.Dense(256)(cond)))

        x = noisy_joints                      # (B, chunk, 6)
        skips = []
        for dim in self.down_dims:
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            skips.append(x)
            # Stride-2 downsample in time (10 -> 5 -> 3 with two levels).
            x = nn.Conv(dim, kernel_size=(3,), strides=(2,), padding='SAME')(x)

        mid = self.down_dims[-1]
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond)
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond)

        for dim, skip in zip(reversed(self.down_dims), reversed(skips)):
            # Upsample, then trim to the skip's length (odd lengths don't double back).
            x = jnp.repeat(x, 2, axis=1)[:, :skip.shape[1]]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)

        n_joint = self.action_dim - 1
        eps = nn.Conv(n_joint, kernel_size=(1,))(x)

        # The binary gripper is predicted directly, outside the diffusion.
        g = nn.Dense(128)(combined_features)
        g = mish(g)
        g = nn.Dense(self.chunk_size)(g).reshape(-1, self.chunk_size, 1)
        return eps, g


class DiffusionHead(nn.Module):
    """Residual-MLP denoiser over the joint columns of an action chunk.

    Diffusion keeps multimodal actions apart where an L1 head would average
    them. The binary gripper is not diffused; it gets a separate logit.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    hidden: int = 512
    n_layers: int = 3

    @nn.compact
    def __call__(self, combined_features, noisy_joints, t):
        """Return (predicted noise, gripper logits).

        noisy_joints: (B, chunk, GRIPPER_IDX); t: (B,) integer noise levels.
        """
        b = combined_features.shape[0]
        x = jnp.concatenate([
            combined_features,
            noisy_joints.reshape(b, -1),
            timestep_embedding(t),
        ], axis=-1)

        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        # Residual blocks: a plain deep MLP trains poorly as a denoiser.
        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden)(x)
            h = nn.relu(h)
            h = nn.Dense(self.hidden)(h)
            x = nn.relu(x + h)

        n_joint = self.action_dim - 1
        eps = nn.Dense(self.chunk_size * n_joint)(x)
        eps = eps.reshape(b, self.chunk_size, n_joint)

        # Gripper logits come straight from the features, not the diffusion.
        g = nn.Dense(128)(combined_features)
        g = nn.relu(g)
        g = nn.Dense(self.chunk_size)(g).reshape(b, self.chunk_size, 1)
        return eps, g


class COLAModel:
    """Frozen Octo-Base backbone + trainable CoLA adapters for two arms."""

    def __init__(
        self,
        octo_checkpoint: str = "hf://rail-berkeley/octo-base",
        use_proprio: bool = False,
        use_overhead: bool = False,
        use_wrist: bool = True,
        split_gripper: bool = False,
        use_diffusion: bool = False,
        diffusion_unet: bool = False,
        unet_dims: tuple = None,
        message_dim: int = MESSAGE_DIM,
    ):
        """
        use_proprio:    add each arm's own joint state to its features.
        use_overhead:   add the fixed overhead camera's features to both arms.
        use_wrist:      use each arm's own wrist camera; False with use_overhead
                        gives an overhead-only model.
        split_gripper:  emit the gripper as a BCE logit (see CoordinationHead).
        use_diffusion:  diffusion head for the joints; the gripper stays a logit.
        diffusion_unet: 1D U-Net denoiser instead of the residual MLP.
        unet_dims:      U-Net channel widths per level; None keeps (128, 256).
        message_dim:    width of the message channel.

        The defaults keep older checkpoints loadable.
        """
        print("Loading Octo backbone...")
        self.octo = OctoModel.load_pretrained(octo_checkpoint)
        print("Octo loaded (frozen)")

        # Fixed task prompt shared by both arms.
        self.task = self.octo.create_tasks(texts=["coordinate with partner"])

        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.use_wrist = use_wrist
        if not (use_wrist or use_overhead):
            raise ValueError('CoLA needs a camera: set use_wrist, use_overhead or both.')
        self.split_gripper = split_gripper
        self.use_diffusion = use_diffusion
        if use_diffusion:
            # Precomputed noise schedule.
            self.alpha_bars = cosine_alpha_bars(DIFFUSION_STEPS)
            # Advanced on every forward() so each control step samples fresh noise.
            self._sample_rng = jax.random.PRNGKey(0)

        # Adapters. message_dim is stored so eval can rebuild a checkpoint at
        # the width it was trained with.
        self.message_dim = message_dim
        self.encoder_a = MessageEncoder(message_dim=message_dim)
        self.encoder_b = MessageEncoder(message_dim=message_dim)
        self.decoder_a = MessageDecoder(feature_dim=FEATURE_DIM)
        self.decoder_b = MessageDecoder(feature_dim=FEATURE_DIM)
        self.diffusion_unet = diffusion_unet
        self.unet_dims = tuple(unet_dims) if unet_dims else None
        if use_diffusion:
            Head = ConditionalUnet1D if diffusion_unet else DiffusionHead
            # down_dims only exists on the U-Net.
            kw = {'action_dim': ACTION_DIM}
            if diffusion_unet and self.unet_dims is not None:
                kw['down_dims'] = self.unet_dims
            self.action_head_a = Head(**kw)
            self.action_head_b = Head(**kw)
        else:
            self.action_head_a = CoordinationHead(action_dim=ACTION_DIM, split_gripper=split_gripper)
            self.action_head_b = CoordinationHead(action_dim=ACTION_DIM, split_gripper=split_gripper)

        rng = jax.random.PRNGKey(0)
        dummy_message = jnp.ones((1, message_dim))

        # Each arm's self-representation: its wrist and/or the overhead
        # features, plus its embedded state when enabled. It feeds both the
        # message encoder and the action head.
        self_dim = ((FEATURE_DIM if use_wrist else 0)
                    + (FEATURE_DIM if use_overhead else 0)
                    + (PROPRIO_DIM if use_proprio else 0))
        dummy_self = jnp.ones((1, self_dim))
        dummy_combined = jnp.ones((1, self_dim + FEATURE_DIM))
        # Init shapes for the diffusion head: a noisy joint chunk and a noise level.
        dummy_noisy = jnp.zeros((1, CHUNK_SIZE, ACTION_DIM - 1))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)

        self.params = {
            'encoder_a': self.encoder_a.init(rng, dummy_self),
            'encoder_b': self.encoder_b.init(rng, dummy_self),
            'decoder_a': self.decoder_a.init(rng, dummy_message),
            'decoder_b': self.decoder_b.init(rng, dummy_message),
            'action_head_a': (self.action_head_a.init(rng, dummy_combined, dummy_noisy, dummy_t)
                              if use_diffusion else
                              self.action_head_a.init(rng, dummy_combined)),
            'action_head_b': (self.action_head_b.init(rng, dummy_combined, dummy_noisy, dummy_t)
                              if use_diffusion else
                              self.action_head_b.init(rng, dummy_combined)),
        }

        if use_proprio:
            self.proprio_a = ProprioEncoder(proprio_dim=PROPRIO_DIM)
            self.proprio_b = ProprioEncoder(proprio_dim=PROPRIO_DIM)
            dummy_state = jnp.ones((1, STATE_DIM))
            self.params['proprio_a'] = self.proprio_a.init(rng, dummy_state)
            self.params['proprio_b'] = self.proprio_b.init(rng, dummy_state)

        n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(self.params)))
        print(f"COLA adapters initialized ({n_params:,} trainable params)")
        print(f"  action chunking: chunk_size={CHUNK_SIZE}, action_dim={ACTION_DIM}")
        print(f"  proprioception: {'on' if use_proprio else 'OFF'}"
              f" | gripper head: {'logit (BCE)' if split_gripper else 'tanh (regression)'}")

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
                "Pass proprio_a/proprio_b, or build the model with "
                "use_proprio=False."
            )
        encoder = self.proprio_a if arm == 'a' else self.proprio_b
        embedded = encoder.apply(params[f'proprio_{arm}'], jnp.asarray(state))
        return jnp.concatenate([features, embedded], axis=-1)

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

    def forward(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        params=None,
        use_messages: bool = True,
        msg_a_override=None,
        proprio_a: np.ndarray = None,
        proprio_b: np.ndarray = None,
        image_o: np.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass from images, with message passing between the arms.

        use_messages=False zeros the channel (no-message ablation).
        msg_a_override replaces the message B receives from A (swap intervention).
        """
        if params is None:
            params = self.params

        # Each arm's own wrist view (unused by an overhead-only model).
        features_a = self.extract_octo_features(image_a) if self.use_wrist else None
        features_b = self.extract_octo_features(image_b) if self.use_wrist else None

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

        self_a = self._self_repr(features_a, proprio_a, params, 'a', features_o)
        self_b = self._self_repr(features_b, proprio_b, params, 'b', features_o)

        msg_a = self.encoder_a.apply(params['encoder_a'], self_a)
        msg_b = self.encoder_b.apply(params['encoder_b'], self_b)

        if not use_messages:
            msg_a = jnp.zeros_like(msg_a)
            msg_b = jnp.zeros_like(msg_b)

        # Message-swap intervention: B receives a message recorded in another
        # episode. Applied after zeroing, so an override always wins.
        if msg_a_override is not None:
            ov = jnp.asarray(msg_a_override)
            if ov.shape != msg_a.shape:
                raise ValueError(
                    f'msg_a_override has shape {ov.shape}, expected '
                    f'{msg_a.shape} -- a mismatch would broadcast silently')
            msg_a = ov

        decoded_b = self.decoder_a.apply(params['decoder_a'], msg_b)  # A decodes B
        decoded_a = self.decoder_b.apply(params['decoder_b'], msg_a)  # B decodes A

        combined_a = jnp.concatenate([self_a, decoded_b], axis=-1)
        combined_b = jnp.concatenate([self_b, decoded_a], axis=-1)

        if self.use_diffusion:
            # Sample a chunk with the reverse process. The RNG advances every
            # call so control steps don't reuse the same noise.
            self._sample_rng, k_a, k_b = jax.random.split(self._sample_rng, 3)
            action_a = self.sample_actions(combined_a, params, 'a', k_a)
            action_b = self.sample_actions(combined_b, params, 'b', k_b)
            return action_a, action_b

        action_a = self.action_head_a.apply(params['action_head_a'], combined_a)
        action_b = self.action_head_b.apply(params['action_head_b'], combined_b)

        return action_a, action_b

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

    def denoise(self, combined, noisy_joints, t, params, arm: str):
        """One denoiser call for arm 'a' or 'b': (predicted noise, gripper logits)."""
        head = self.action_head_a if arm == 'a' else self.action_head_b
        return head.apply(params[f'action_head_{arm}'], combined, noisy_joints, t)

    def forward_from_features(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        params=None,
        use_messages: bool = True,
        msg_a_override=None,
        proprio_a: np.ndarray = None,
        proprio_b: np.ndarray = None,
        features_o: np.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass from cached Octo features (used in training).

        features_a/b/o are (batch, 768); features_o is the shared overhead view,
        required when the model uses it. Other arguments as in forward().
        """
        if params is None:
            params = self.params

        features_a = jnp.array(features_a)
        features_b = jnp.array(features_b)

        self_a = self._self_repr(features_a, proprio_a, params, 'a', features_o)
        self_b = self._self_repr(features_b, proprio_b, params, 'b', features_o)

        msg_a = self.encoder_a.apply(params['encoder_a'], self_a)
        msg_b = self.encoder_b.apply(params['encoder_b'], self_b)

        if not use_messages:
            msg_a = jnp.zeros_like(msg_a)
            msg_b = jnp.zeros_like(msg_b)

        # Message-swap intervention: B receives a message recorded in another
        # episode. Applied after zeroing, so an override always wins.
        if msg_a_override is not None:
            ov = jnp.asarray(msg_a_override)
            if ov.shape != msg_a.shape:
                raise ValueError(
                    f'msg_a_override has shape {ov.shape}, expected '
                    f'{msg_a.shape} -- a mismatch would broadcast silently')
            msg_a = ov

        decoded_b = self.decoder_a.apply(params['decoder_a'], msg_b)
        decoded_a = self.decoder_b.apply(params['decoder_b'], msg_a)

        combined_a = jnp.concatenate([self_a, decoded_b], axis=-1)
        combined_b = jnp.concatenate([self_b, decoded_a], axis=-1)

        if self.use_diffusion:
            # Diffusion: return the conditioning vectors; the trainer and the
            # sampler drive the head themselves.
            return combined_a, combined_b

        action_a = self.action_head_a.apply(params['action_head_a'], combined_a)
        action_b = self.action_head_b.apply(params['action_head_b'], combined_b)

        return action_a, action_b


if __name__ == '__main__':
    print("=" * 60)
    print("Testing COLA Handover Architecture (Octo-Base)")
    print("=" * 60)

    cola = COLAModel()

    batch_size = 4
    dummy_a = np.random.randint(0, 256, (batch_size, 256, 256, 3), dtype=np.uint8)
    dummy_b = np.random.randint(0, 256, (batch_size, 256, 256, 3), dtype=np.uint8)

    action_a, action_b = cola.forward(dummy_a, dummy_b)
    print(f"action_a: {action_a.shape}, range [{action_a.min():.3f}, {action_a.max():.3f}]")
    print(f"action_b: {action_b.shape}, range [{action_b.min():.3f}, {action_b.max():.3f}]")
    assert action_a.shape == (batch_size, CHUNK_SIZE, ACTION_DIM)
    assert action_b.shape == (batch_size, CHUNK_SIZE, ACTION_DIM)
    print("OK")
