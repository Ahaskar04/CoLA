"""
COLA: Coordination via Latent Adapters (Handover / Octo-Base)

Two-agent coordination for the handover task built on top of a frozen
Octo-Base backbone. Adapters are trainable; Octo is not.
"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from octo.model.octo_model import OctoModel
from typing import Tuple
import numpy as np


# Feature dim of Octo-Base's readout_action token.
# Octo-Small is 384; Octo-Base is 768 (confirmed via model config.json on HF).
FEATURE_DIM = 768
MESSAGE_DIM = 64
ACTION_DIM = 7   # 6 joints + 1 gripper (handover ViperX)
CHUNK_SIZE = 10  # action chunking: predict 10 consecutive actions per query

STATE_DIM = 7      # 6 joint positions + gripper finger position
PROPRIO_DIM = 64   # width the raw state is embedded to before concatenation

# Index of the gripper column in an action vector. Columns 0..5 are joints.
GRIPPER_IDX = 6


class MessageEncoder(nn.Module):
    """Compress 768-dim Octo features to a 64-dim message."""
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
    """Embed a 7-dim joint state so it is not drowned out by 768 vision dims.

    Concatenating the raw state onto the feature vector would give it 7 of 775
    input columns; a learned 64-dim embedding puts it on comparable footing.
    """
    proprio_dim: int = PROPRIO_DIM

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(64)(state)
        x = nn.relu(x)
        return nn.Dense(self.proprio_dim)(x)


class CoordinationHead(nn.Module):
    """Predict a chunk of `chunk_size` future actions from [own_features | decoded_partner_message].

    split_gripper changes the output parameterisation: joints stay bounded by
    tanh (their targets are min/max normalised to [-0.9, 0.9]), while the
    gripper column is emitted as a RAW LOGIT for binary_cross_entropy. The
    gripper is binary in the data -- the normalised column takes exactly two
    values, -0.9 and +0.9 -- so squashing it through tanh and regressing makes
    the loss reward hedging near the midpoint. Callers must apply a sigmoid, or
    threshold at 0, before sending the column to an actuator.
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


# --------------------------------------------------------------------------
# Diffusion action head
# --------------------------------------------------------------------------
# Number of noise levels used at TRAINING time. Sampling can take far fewer
# (see the DDIM stride in the evaluator): 100 gives a fine-grained schedule to
# learn from without forcing 100 forward passes per control step at rollout.
DIFFUSION_STEPS = 100
# Width of the sinusoidal timestep embedding. The denoiser has to behave very
# differently at t=5 (nearly clean) and t=95 (nearly pure noise), so it needs
# the noise level as an input -- and a raw integer is a poor MLP input, with no
# smoothness between neighbouring steps.
TIME_EMBED_DIM = 64


def cosine_alpha_bars(n_steps: int = DIFFUSION_STEPS) -> jnp.ndarray:
    """Cumulative signal-retention schedule (alpha-bar), cosine form.

    alpha_bar[t] is how much of the CLEAN action survives at noise level t:
    ~1.0 at t=0, ~0.0 at t=n_steps. Nichol & Dhariwal's cosine schedule spends
    more steps at low noise than the original linear one, which matters here
    because the fine positioning that decides a grasp lives in the last few
    denoising steps.
    """
    s = 0.008
    t = jnp.arange(n_steps + 1, dtype=jnp.float32) / n_steps
    f = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    ab = f / f[0]
    return jnp.clip(ab, 1e-4, 1.0)


def timestep_embedding(t, dim: int = TIME_EMBED_DIM):
    """Sinusoidal encoding of the noise level, as in transformer positions.

    t: (B,) integer noise levels -> (B, dim) float. Nearby timesteps get nearby
    embeddings, which a bare integer input would not give.
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t.astype(jnp.float32)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def mish(x):
    """Mish activation, as used throughout the reference ConditionalUnet1D."""
    return x * jnp.tanh(nn.softplus(x))


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, the U-Net's basic unit."""
    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        # x: (B, T, C) -- Flax convolves over the middle axis, which here is
        # the chunk's TIME axis. That is the whole point of using a CNN: the
        # flat-MLP head reshaped the chunk to 60 numbers and lost any notion of
        # which entries are adjacent in time.
        x = nn.Conv(self.out_channels, kernel_size=(self.kernel_size,),
                    padding='SAME')(x)
        groups = min(self.n_groups, self.out_channels)
        x = nn.GroupNorm(num_groups=groups)(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM conditioning between them.

    FiLM is the fix for the flat head's weakness. There, the observation was
    concatenated onto the input once and the network was free to ignore it --
    the loss only asks it to predict noise, and the noise lives in the action
    columns. Here the conditioning vector GENERATES a per-channel scale and
    bias applied to the activations, so every block's behaviour is set by the
    observation. There is no path through the network that routes around it.
    """
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
    """1D temporal U-Net denoiser, following Chi et al.'s CNN variant.

    Their recommendation is explicit: "We recommend starting with the CNN-based
    diffusion policy implementation as the first attempt at a new task", with
    the transformer reserved for cases needing extra tuning.

    Two differences from our MLP head, both measured as likely causes of its
    30.5% rollout: convolutions run ALONG the chunk so temporal structure is
    architectural rather than inferred, and FiLM injects the observation at
    every block rather than once at the input.

    down_dims is deliberately smaller than the reference [256,512,1024]: a
    10-step chunk cannot survive three downsamples, and 120 training episodes
    do not support that capacity.

    CAPACITY ABLATION. The U-Net beat the flat MLP head 88% to 30.5%, but that
    comparison cannot say WHY: temporal convolution, FiLM conditioning at every
    block, and simply having more parameters all changed at once. Shrinking
    down_dims holds the mechanism fixed and varies only the capacity, which
    separates "the arrangement helps" from "the size helps":

        (128, 256)  default -- the configuration that measured 88%
        (64, 128)   same depth, half the channels. Is 128/256 overkill?
        (128,)      one level, chunk 10 -> 5, no second downsample. Is the
                    second level earning its parameters or just adding them?

    Set it through COLAModel(unet_dims=...) so the trainer's --unet_dims flag
    reaches both arms' heads. Diffusion models overfit small datasets while
    still showing a falling train loss, so read the VAL curve and the rollout,
    not the train loss, when comparing these.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    down_dims: tuple = (128, 256)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, combined_features, noisy_joints, t):
        # Timestep embedding and observation are concatenated into ONE global
        # conditioning vector, which then drives every block -- matching the
        # reference, where global_feature = cat([diffusion_step_emb, global_cond]).
        cond = jnp.concatenate([timestep_embedding(t), combined_features], axis=-1)
        cond = nn.Dense(256)(mish(nn.Dense(256)(cond)))

        x = noisy_joints                      # (B, chunk, 6)
        skips = []
        for dim in self.down_dims:
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            skips.append(x)
            # Stride-2 downsample along time. With chunk_size=10 and two levels
            # the sequence goes 10 -> 5 -> 3, which is as far as it can usefully go.
            x = nn.Conv(dim, kernel_size=(3,), strides=(2,), padding='SAME')(x)

        mid = self.down_dims[-1]
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond)
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond)

        for dim, skip in zip(reversed(self.down_dims), reversed(skips)):
            # Nearest-neighbour upsample, then trim/pad to the skip's length:
            # odd sequence lengths do not double back exactly.
            x = jnp.repeat(x, 2, axis=1)[:, :skip.shape[1]]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond)

        n_joint = self.action_dim - 1
        eps = nn.Conv(n_joint, kernel_size=(1,))(x)

        # The gripper stays outside the diffusion: it is binary, and Gaussian
        # noise on a two-valued column is the bug that once left the hand shut.
        g = nn.Dense(128)(combined_features)
        g = mish(g)
        g = nn.Dense(self.chunk_size)(g).reshape(-1, self.chunk_size, 1)
        return eps, g


class DiffusionHead(nn.Module):


    """Denoiser over the JOINT columns of an action chunk.

    Why diffusion at all: the deterministic CoordinationHead is trained with L1,
    so when several actions are valid from one observation it predicts their
    AVERAGE. Reaching around a box's left side and its right side are both
    valid; the average is a reach through the box. That matches the observed
    failure -- arm B touches the box in 90% of episodes but secures it in 61.5%.
    A denoiser models the action DISTRIBUTION instead, so the modes stay apart.

    Why joints only: the gripper column is binary in this data (exactly -0.9 and
    +0.9 after normalisation) and is handled by a BCE logit head. Gaussian
    diffusion assumes a continuous variable, and regressing this column is the
    exact bug that once left the gripper permanently closed -- the BCE head took
    it to 98.4% accuracy, which full 7-dim diffusion would put back at risk.
    So: diffusion over columns 0..5, BCE logit for column 6, emitted together.

    Wider than CoordinationHead's 128->64 on purpose: a denoiser must work at
    every noise level, which is a substantially harder function than one
    feature vector -> one action.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    hidden: int = 512
    n_layers: int = 3

    @nn.compact
    def __call__(self, combined_features, noisy_joints, t):
        """(features, noisy joint chunk, noise level) -> (predicted noise, gripper logits).

        noisy_joints: (B, chunk, GRIPPER_IDX)
        t:            (B,) integer noise levels
        returns:      (B, chunk, GRIPPER_IDX), (B, chunk, 1)
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

        # The gripper never goes through the diffusion process: it is predicted
        # straight from the features, exactly as the deterministic head did.
        g = nn.Dense(128)(combined_features)
        g = nn.relu(g)
        g = nn.Dense(self.chunk_size)(g).reshape(b, self.chunk_size, 1)
        return eps, g


class COLAModel:
    """
    Frozen Octo-Base vision backbone + trainable coordination adapters.
    """

    def __init__(
        self,
        octo_checkpoint: str = "hf://rail-berkeley/octo-base",
        use_proprio: bool = False,
        use_overhead: bool = False,
        split_gripper: bool = False,
        use_diffusion: bool = False,
        diffusion_unet: bool = False,
        unet_dims: tuple = None,
        message_dim: int = MESSAGE_DIM,
    ):
        """
        use_overhead:  concatenate the fixed overhead camera's Octo features
                       onto BOTH arms' self-representation. The wrist cameras
                       move with the arm, so once the policy drifts off the
                       expert trajectory they show frames present in no demo;
                       a fixed third-person view keeps reporting where the box
                       actually is.
        use_proprio:   feed each arm its own joint state alongside its Octo
                       features. The wrist camera moves with the arm, so without
                       this the policy has to infer its own configuration from a
                       view that barely constrains it, while emitting ABSOLUTE
                       joint targets.
        split_gripper: emit the gripper column as a logit for BCE instead of a
                       tanh regression target. See CoordinationHead.

        use_diffusion: replace the deterministic joint head with a denoiser. The
                       L1-trained head predicts the AVERAGE of valid actions,
                       which for two ways round a box is a reach through it;
                       diffusion keeps the modes apart. The gripper column stays
                       on the BCE logit head either way -- it is binary, and
                       Gaussian diffusion assumes a continuous variable.

        diffusion_unet: use the 1D temporal U-Net denoiser instead of the flat
                       residual MLP. Convolutions run along the chunk so
                       temporal structure is architectural, and FiLM injects the
                       observation at every block rather than once at the input.
                       Chi et al. recommend the CNN variant as the first attempt
                       on a new task. Only meaningful with use_diffusion.

        unet_dims:     channel widths per U-Net level, e.g. (64, 128) or (128,).
                       None keeps ConditionalUnet1D's own default of (128, 256),
                       the configuration that measured 88%. This is the capacity
                       ablation knob -- see ConditionalUnet1D's docstring. Only
                       meaningful with diffusion_unet.

        All default to False/None so v1 checkpoints keep loading and v1 scripts
        keep producing identical numbers.
        """
        print("Loading Octo backbone...")
        self.octo = OctoModel.load_pretrained(octo_checkpoint)
        print("Octo loaded (frozen)")

        # Generic coordination task prompt (same as push-block setup)
        self.task = self.octo.create_tasks(texts=["coordinate with partner"])

        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.split_gripper = split_gripper
        self.use_diffusion = use_diffusion
        if use_diffusion:
            # Precompute the schedule once: the trainer needs alpha_bar[t] on
            # every batch and the sampler walks it backwards.
            self.alpha_bars = cosine_alpha_bars(DIFFUSION_STEPS)
            # Advanced on every forward() so each control step samples afresh.
            self._sample_rng = jax.random.PRNGKey(0)

        # Adapter modules
        # message_dim is the CHANNEL BOTTLENECK the ablations sweep. It is a
        # constructor argument rather than the module constant so a checkpoint
        # can record the width it was trained at; eval must rebuild at the same
        # width or the decoder's first Dense sees the wrong input shape and the
        # params fail to load.
        self.message_dim = message_dim
        self.encoder_a = MessageEncoder(message_dim=message_dim)
        self.encoder_b = MessageEncoder(message_dim=message_dim)
        self.decoder_a = MessageDecoder(feature_dim=FEATURE_DIM)
        self.decoder_b = MessageDecoder(feature_dim=FEATURE_DIM)
        self.diffusion_unet = diffusion_unet
        self.unet_dims = tuple(unet_dims) if unet_dims else None
        if use_diffusion:
            Head = ConditionalUnet1D if diffusion_unet else DiffusionHead
            # down_dims exists only on the U-Net; passing it to the flat
            # DiffusionHead would be a TypeError. Leaving it out when unet_dims
            # is None keeps the dataclass default (128, 256), so an unflagged
            # run reproduces the 88% configuration exactly.
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

        # The "self representation" each arm builds before talking to its
        # partner: vision, plus its own state when proprio is enabled. It feeds
        # BOTH the message encoder and the action head -- the partner's most
        # useful information about this arm is where it currently is, so
        # excluding proprio from the message would withhold exactly that.
        # Overhead adds a second FEATURE_DIM block: both arms attend over the
        # same third-person view, concatenated onto their own wrist features.
        self_dim = (FEATURE_DIM
                    + (FEATURE_DIM if use_overhead else 0)
                    + (PROPRIO_DIM if use_proprio else 0))
        dummy_self = jnp.ones((1, self_dim))
        dummy_combined = jnp.ones((1, self_dim + FEATURE_DIM))
        # Shapes the DiffusionHead needs at init: a noisy JOINT chunk (the
        # gripper column never enters the diffusion) and one noise level.
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
        """Concatenate an arm's own features with the overhead view and its state."""
        if self.use_overhead:
            if features_o is None:
                raise ValueError(
                    "use_overhead=True but no features_o was passed. Pass the "
                    "overhead features, or build the model with use_overhead=False."
                )
            features = jnp.concatenate([features, jnp.asarray(features_o)], axis=-1)
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
        """COLA forward pass with symmetric message passing.

        Args:
            use_messages: if False, the inter-agent message channel is zeroed
                out, so each arm acts only on its own Octo features with no
                coordination. This is the "L0 / no-coordination" baseline path
                for live images (mirrors forward_from_features(use_messages=False)).
        """
        if params is None:
            params = self.params

        features_a = self.extract_octo_features(image_a)
        features_b = self.extract_octo_features(image_b)

        # One fixed third-person view, shared by both arms. Must be supplied
        # whenever the model was built with use_overhead, or the concatenated
        # self-representation will not match the trained parameter shapes.
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

        # MESSAGE-SWAP INTERVENTION. Replace what B receives from A with a
        # message recorded from a DIFFERENT episode -- typically one whose
        # marker was another colour. A linear probe shows the colour is READABLE
        # in msg_a (73.3% over 150 episodes, against a matched no-message
        # control at exactly 33.3%), but readable is not used: that is Lowe et
        # al.'s positive-signalling / positive-listening distinction. If B
        # follows the SWAPPED colour's tray, the channel causally drives the
        # decision. If B still goes to the true colour's tray, it is getting the
        # colour from somewhere other than the message.
        #
        # Applied AFTER the use_messages zeroing so the two are composable and
        # an override always wins. Shape must match msg_a exactly; a mismatch
        # here would broadcast silently and corrupt every downstream action.
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
            # A denoiser has no single forward answer: draw a chunk by running
            # the reverse process. self._sample_rng advances every call so
            # successive control steps are not handed the identical noise draw
            # -- reusing one key would collapse the policy to a deterministic
            # function of the observation and throw away the reason for using
            # diffusion at all.
            self._sample_rng, k_a, k_b = jax.random.split(self._sample_rng, 3)
            action_a = self.sample_actions(combined_a, params, 'a', k_a)
            action_b = self.sample_actions(combined_b, params, 'b', k_b)
            return action_a, action_b

        action_a = self.action_head_a.apply(params['action_head_a'], combined_a)
        action_b = self.action_head_b.apply(params['action_head_b'], combined_b)

        return action_a, action_b

    def sample_actions(self, combined, params, arm: str, rng, n_steps: int = None):
        """DDIM sampling: pure noise -> an action chunk, in n_steps passes.

        Training uses DIFFUSION_STEPS (100) noise levels, but sampling every one
        would mean 100 forward passes per control step, which the rollout cannot
        afford. DDIM is deterministic given the starting noise and skips levels
        with little quality loss, so 10 strided steps stand in for 100.

        Returns (B, chunk, ACTION_DIM): denoised joints with the gripper LOGIT
        appended, matching what the deterministic head emits, so callers
        threshold column 6 exactly as before.
        """
        # 10 strided steps stand in for the 100 noise levels seen in training.
        # Bigger jumps assume the denoising direction holds across the gap being
        # skipped, so raising this trades eval time for fidelity to what the model
        # actually learned. COLA_DDIM_STEPS lets a rollout sweep it without
        # retraining -- the schedule is fixed at training time, the stride is not.
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
            # Back out the implied clean action from the predicted noise.
            x0 = (x - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(a_t)
            x0 = jnp.clip(x0, -1.0, 1.0)   # joints are min/max normalised
            if i < n_steps - 1:
                a_prev = ab[ts[i + 1]]
                # DDIM update with eta=0: no noise re-injected, so the sample is
                # a deterministic function of the starting draw.
                x = jnp.sqrt(a_prev) * x0 + jnp.sqrt(1.0 - a_prev) * eps
            else:
                x = x0
        return jnp.concatenate([x, grip], axis=-1)

    def denoise(self, combined, noisy_joints, t, params, arm: str):
        """One denoiser call: predict the noise in `noisy_joints`.

        arm selects which head's params to use ('a' or 'b'). Returns
        (predicted_noise, gripper_logits) -- the gripper is emitted straight
        from the conditioning features and never takes part in the diffusion.
        """
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
        """
        Forward pass from pre-extracted Octo features (skips the backbone).

        Used during training with cached features. At eval time, use forward()
        which runs Octo on live images.

        Args:
            features_a: (batch, 768) pre-extracted features for agent A
            features_b: (batch, 768) pre-extracted features for agent B
            features_o: (batch, 768) overhead-camera features, shared by both
                agents (required when the model was built with use_overhead)
            params: adapter params (for jax.grad)
            use_messages: if False, zero out messages (no-message ablation)
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

        # MESSAGE-SWAP INTERVENTION. Replace what B receives from A with a
        # message recorded from a DIFFERENT episode -- typically one whose
        # marker was another colour. A linear probe shows the colour is READABLE
        # in msg_a (73.3% over 150 episodes, against a matched no-message
        # control at exactly 33.3%), but readable is not used: that is Lowe et
        # al.'s positive-signalling / positive-listening distinction. If B
        # follows the SWAPPED colour's tray, the channel causally drives the
        # decision. If B still goes to the true colour's tray, it is getting the
        # colour from somewhere other than the message.
        #
        # Applied AFTER the use_messages zeroing so the two are composable and
        # an override always wins. Shape must match msg_a exactly; a mismatch
        # here would broadcast silently and corrupt every downstream action.
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
            # The denoiser needs (features, noisy chunk, noise level), which
            # only the training loop and the sampler have. Hand back the
            # conditioning vectors and let them drive the head; returning an
            # "action" here would mean sampling on every call, which the
            # trainer does not want and cannot differentiate through cheaply.
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
