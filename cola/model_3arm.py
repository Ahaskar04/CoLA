"""
COLA: Coordination via Latent Adapters -- THREE-ARM handover (Octo-Base).

Port of the 2-arm cola_architecture.py that produced 88-91% on handover-only
(diffusion + 1D temporal U-Net). Everything that decided that number is kept
where it could be: the cosine alpha-bar schedule, DDIM sampling with the
COLA_DDIM_STEPS override, FiLM-conditioned residual blocks, joints-only
diffusion with the gripper on a separate BCE logit.

WHAT CHANGED, and only this:

1. ARMS is now ('a', 'b', 'c') and every per-arm module is held in a dict keyed
   by arm rather than in hand-written _a/_b attributes. The 2-arm file wrote
   `self.encoder_a`, `self.encoder_b`, `params['action_head_a']`, ... which does
   not extend to three without triplicating every line.

2. ALL-TO-ALL message passing. The 2-arm model was a symmetric pair: A decoded
   B's message, B decoded A's. With three arms each arm encodes ONE message and
   decodes BOTH partners', so the combined vector carries two decoded blocks:

       combined_x = [ self_x | dec_{x<-p} | dec_{x<-q} ]   (p, q the partners)

   Partners are concatenated in fixed ARMS order (never the arm's own slot), so
   arm B always reads A in the first block and C in the second. That ordering is
   part of the checkpoint contract -- swapping it silently feeds the decoders
   each other's inputs.

   Chosen over a chain A<->B<->C because during turn_b arm B is simultaneously
   releasing from A and presenting to C, and a chain would force that through
   one shared channel. It costs one extra FEATURE_DIM block on the head input.

3. One decoder PER ORDERED PAIR, not per arm. decoders['b']['a'] is "how B reads
   A". Sharing a single decoder per receiver would make A's and C's messages
   pass through identical weights, which is exactly the distinction B needs
   during the two-legged handover.

The message channel stays MESSAGE_DIM=64 per edge, so the bottleneck the
ablations probe is unchanged in width -- there are simply more edges.
"""

import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from octo.model.octo_model import OctoModel


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

# The three arms, in the order the dataset writes them: A picks the box up and
# hands to B, B turns 180 deg and hands on to C. This tuple fixes the
# concatenation order of decoded partner messages, so it is part of the
# checkpoint contract -- see PARTNERS below.
ARMS = ('a', 'b', 'c')
N_ARMS = len(ARMS)

# For each arm, its partners in fixed ARMS order. Arm B reads A then C, always.
PARTNERS: Dict[str, Tuple[str, ...]] = {
    arm: tuple(o for o in ARMS if o != arm) for arm in ARMS
}


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
    """Predict a chunk of `chunk_size` future actions from [own | decoded partners].

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
# (see the DDIM stride below): 100 gives a fine-grained schedule to learn from
# without forcing 100 forward passes per control step at rollout.
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

    Two differences from the MLP head, both measured as likely causes of its
    30.5% rollout on the 2-arm task: convolutions run ALONG the chunk so
    temporal structure is architectural rather than inferred, and FiLM injects
    the observation at every block rather than once at the input.

    down_dims is deliberately smaller than the reference [256,512,1024]: a
    10-step chunk cannot survive three downsamples, and a few hundred training
    episodes do not support that capacity.
    """
    action_dim: int = ACTION_DIM
    chunk_size: int = CHUNK_SIZE
    down_dims: Tuple[int, ...] = (128, 256)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, cond, noisy_joints, t):
        """cond: (B, D) observation. noisy_joints: (B, chunk, 6). t: (B,).

        Body ported verbatim from the 2-arm ConditionalUnet1D that reached 88%
        on handover-only, so the 3-arm run differs from it only in the task and
        the message topology -- not the denoiser. The parameter is still named
        `cond` rather than `combined_features` to keep this file's callers
        unchanged.
        """
        # Timestep embedding and observation are concatenated into ONE global
        # conditioning vector, which then drives every block -- matching the
        # reference, where global_feature = cat([diffusion_step_emb, global_cond]).
        cond_g = jnp.concatenate([timestep_embedding(t), cond], axis=-1)
        cond_g = nn.Dense(256)(mish(nn.Dense(256)(cond_g)))

        x = noisy_joints                      # (B, chunk, 6)
        skips = []
        for dim in self.down_dims:
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            skips.append(x)
            # Stride-2 downsample along time. With chunk_size=10 and two levels
            # the sequence goes 10 -> 5 -> 3, which is as far as it can usefully go.
            # This is what makes it a U-Net: the coarse level sees the shape of
            # the manoeuvre, the skips carry the fine detail back.
            x = nn.Conv(dim, kernel_size=(3,), strides=(2,), padding='SAME')(x)

        mid = self.down_dims[-1]
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond_g)
        x = ConditionalResidualBlock1D(mid, self.kernel_size, self.n_groups)(x, cond_g)

        for dim, skip in zip(reversed(self.down_dims), reversed(skips)):
            # Nearest-neighbour upsample, then trim/pad to the skip's length:
            # odd sequence lengths do not double back exactly.
            x = jnp.repeat(x, 2, axis=1)[:, :skip.shape[1]]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)
            x = ConditionalResidualBlock1D(dim, self.kernel_size, self.n_groups)(x, cond_g)

        n_joint = self.action_dim - 1
        eps = nn.Conv(n_joint, kernel_size=(1,))(x)

        # The gripper stays outside the diffusion: it is binary, and Gaussian
        # noise on a two-valued column is the bug that once left the hand shut.
        g = nn.Dense(128)(cond)
        g = mish(g)
        g = nn.Dense(self.chunk_size)(g).reshape(-1, self.chunk_size, 1)
        return eps, g


class DiffusionHead(nn.Module):
    """Flat residual-MLP denoiser over the JOINT columns of an action chunk.

    Kept for ablation parity with the 2-arm study, where it scored 30.5% against
    the U-Net's 88-91%. Use ConditionalUnet1D unless you are reproducing that
    comparison.

    Why diffusion at all: the deterministic CoordinationHead is trained with L1,
    so when several actions are valid from one observation it predicts their
    AVERAGE. Reaching around a box's left side and its right side are both
    valid; the average is a reach through the box.

    Why joints only: the gripper column is binary in this data (exactly -0.9 and
    +0.9 after normalisation) and is handled by a BCE logit head. Gaussian
    diffusion assumes a continuous variable, and regressing this column is the
    exact bug that once left the gripper permanently closed.
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
    """Three-arm COLA: frozen Octo backbone, trainable per-arm adapters.

    Parameter tree (all trainable; Octo is not in here):

        encoder_{a,b,c}          own self-repr -> 64-d message
        decoder_{x}_{p}          partner p's message -> 768-d, as read by x
                                 (6 of these: one per ORDERED pair)
        action_head_{a,b,c}      [self | dec_p | dec_q] -> action chunk
        proprio_{a,b,c}          7-d joint state -> 64-d  (use_proprio only)
    """

    def __init__(
        self,
        octo_checkpoint: str = "hf://rail-berkeley/octo-base",
        use_proprio: bool = False,
        use_overhead: bool = False,
        split_gripper: bool = False,
        use_diffusion: bool = False,
        diffusion_unet: bool = False,
    ):
        """
        use_overhead:  concatenate the fixed overhead camera's Octo features
                       onto EVERY arm's self-representation. The wrist cameras
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
        use_diffusion: replace the deterministic joint head with a denoiser.
        diffusion_unet: use the 1D temporal U-Net denoiser instead of the flat
                       residual MLP. This is the configuration that produced
                       88-91% on the 2-arm task. Only meaningful with
                       use_diffusion.
        """
        print("Loading Octo backbone...")
        self.octo = OctoModel.load_pretrained(octo_checkpoint)
        print("Octo loaded (frozen)")

        # Generic coordination task prompt (same as the 2-arm setup, so the
        # frozen backbone sees an identical instruction and its features stay
        # comparable across the two studies).
        self.task = self.octo.create_tasks(texts=["coordinate with partner"])

        self.use_proprio = use_proprio
        self.use_overhead = use_overhead
        self.split_gripper = split_gripper
        self.use_diffusion = use_diffusion
        self.diffusion_unet = diffusion_unet

        if use_diffusion:
            # Precompute the schedule once: the trainer needs alpha_bar[t] on
            # every batch and the sampler walks it backwards.
            self.alpha_bars = cosine_alpha_bars(DIFFUSION_STEPS)
            # Advanced on every forward() so each control step samples afresh.
            self._sample_rng = jax.random.PRNGKey(0)

        # --- adapter modules, one per arm / per ordered pair ---
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

        # The "self representation" each arm builds before talking to its
        # partners: vision, plus its own state when proprio is enabled. It feeds
        # BOTH the message encoder and the action head -- a partner's most
        # useful information about this arm is where it currently is, so
        # excluding proprio from the message would withhold exactly that.
        self_dim = (FEATURE_DIM
                    + (FEATURE_DIM if use_overhead else 0)
                    + (PROPRIO_DIM if use_proprio else 0))
        # TWO decoded partner blocks now, not one. This is the only shape change
        # the 3-arm port makes to the head input.
        combined_dim = self_dim + (N_ARMS - 1) * FEATURE_DIM

        dummy_self = jnp.ones((1, self_dim))
        dummy_combined = jnp.ones((1, combined_dim))
        # Shapes the diffusion head needs at init: a noisy JOINT chunk (the
        # gripper column never enters the diffusion) and one noise level.
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

    # ----------------------------------------------------------------------
    # Shared plumbing
    # ----------------------------------------------------------------------

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
                f"Pass proprio_{arm}, or build the model with use_proprio=False."
            )
        embedded = self.proprio_encoders[arm].apply(
            params[f'proprio_{arm}'], jnp.asarray(state))
        return jnp.concatenate([features, embedded], axis=-1)

    def _combine(self, selves, params, use_messages: bool):
        """self-reprs -> per-arm [self | decoded partner p | decoded partner q].

        Partners are concatenated in fixed PARTNERS order (which follows ARMS),
        so arm B always reads A first and C second. Changing that order
        invalidates every existing checkpoint.
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

    # ----------------------------------------------------------------------
    # Forward passes
    # ----------------------------------------------------------------------

    def forward(
        self,
        images: Dict[str, np.ndarray],
        params=None,
        use_messages: bool = True,
        proprio: Dict[str, np.ndarray] = None,
        image_o: np.ndarray = None,
    ) -> Dict[str, jnp.ndarray]:
        """COLA forward pass from LIVE images, with all-to-all message passing.

        Args:
            images: {'a': (B,H,W,3), 'b': ..., 'c': ...} wrist views.
            proprio: {'a': (B,7), ...} joint states, when use_proprio.
            image_o: (B,H,W,3) overhead view, when use_overhead.
            use_messages: if False, every message is zeroed, so each arm acts
                only on its own features with no coordination. This is the
                "L0 / no-coordination" baseline path.

        Returns {'a': (B, chunk, 7), 'b': ..., 'c': ...}.
        """
        if params is None:
            params = self.params
        proprio = proprio or {}

        missing = [a for a in ARMS if a not in images]
        if missing:
            raise ValueError(f"forward() needs an image for every arm; missing {missing}")

        features = {a: self.extract_octo_features(images[a]) for a in ARMS}

        # One fixed third-person view, shared by all arms. Must be supplied
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

        selves = {
            a: self._self_repr(features[a], proprio.get(a), params, a, features_o)
            for a in ARMS
        }
        combined = self._combine(selves, params, use_messages)

        if self.use_diffusion:
            # A denoiser has no single forward answer: draw a chunk by running
            # the reverse process. self._sample_rng advances every call so
            # successive control steps are not handed the identical noise draw
            # -- reusing one key would collapse the policy to a deterministic
            # function of the observation and throw away the reason for using
            # diffusion at all.
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
        """Forward pass from pre-extracted Octo features (skips the backbone).

        Used during training with cached features. At eval time, use forward()
        which runs Octo on live images.

        Under use_diffusion this returns the CONDITIONING vectors, not actions --
        the denoiser needs (features, noisy chunk, noise level), which only the
        training loop and the sampler have. Returning an "action" here would mean
        sampling on every call, which the trainer does not want and cannot
        differentiate through cheaply.

        Args:
            features: {'a': (B, 768), 'b': ..., 'c': ...}
            features_o: (B, 768) overhead features shared by all arms
                (required when the model was built with use_overhead)
        """
        if params is None:
            params = self.params
        proprio = proprio or {}

        selves = {
            a: self._self_repr(jnp.asarray(features[a]), proprio.get(a),
                               params, a, features_o)
            for a in ARMS
        }
        combined = self._combine(selves, params, use_messages)

        if self.use_diffusion:
            return combined

        return {
            a: self.action_heads[a].apply(params[f'action_head_{a}'], combined[a])
            for a in ARMS
        }

    # ----------------------------------------------------------------------
    # Diffusion sampling
    # ----------------------------------------------------------------------

    def denoise(self, combined, noisy_joints, t, params, arm: str):
        """One denoiser call: predict the noise in `noisy_joints`.

        arm selects which head's params to use. Returns (predicted_noise,
        gripper_logits) -- the gripper is emitted straight from the conditioning
        features and never takes part in the diffusion.
        """
        return self.action_heads[arm].apply(
            params[f'action_head_{arm}'], combined, noisy_joints, t)

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
        # COLA_DDIM_STEPS lets a rollout sweep the stride without retraining --
        # the schedule is fixed at training time, the stride is not.
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


if __name__ == '__main__':
    # Shape smoke test with random features -- no Octo download, no dataset.
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
