"""U-Net diffusion action head for Octo-1.5 batches.

octo ships UNetDDPMActionHead, but it predates the 1.5 data format and cannot
be used as-is:
  * loss() takes (action_pad_mask, timestep_pad_mask) -- the reverse of every
    other octo head -- so the shared training loop would swap the masks;
  * it assumes a per-dimension action_pad_mask of shape (B, A) and noisy
    actions without a window axis, while 1.5 batches carry (B, W, H, A).
  * its use_map / flatten_tokens defaults are the tuple (False,), which is
    truthy, so the default construction trips its own assertion.

This head keeps octo's ConditionalUnet1D denoiser (Chi et al.'s 1D temporal
U-Net: convolutions run ALONG the action chunk, FiLM conditioning on the
readout embedding and diffusion time at every block) and wraps it in
DiffusionActionHead's 1.5-format loss and DDPM sampler.

The U-Net halves the chunk once per level but the last, so action_horizon
must be divisible by 2**(len(down_features)-1): 4 for the default size.
There are no pretrained weights for it: the head always starts cold.
"""
import logging
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from octo.model.components.action_heads import continuous_loss
from octo.model.components.base import TokenGroup
from octo.model.components.diffusion import cosine_beta_schedule
from octo.model.components.transformer import MAPHead
from octo.model.components.unet import ConditionalUnet1D
from octo.utils.typing import PRNGKey


class UNetActionHead(nn.Module):
    readout_key: str
    use_map: bool = False
    action_horizon: int = 12
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"

    down_features: Tuple[int, ...] = (256, 512, 1024)
    mid_layers: int = 2
    kernel_size: int = 5
    n_groups: int = 8
    time_features: int = 128
    diffusion_steps: int = 100
    n_diffusion_samples: int = 1

    def setup(self):
        # One halving per level except the last, so the chunk must survive
        # len(down_features)-1 of them: 4 for the (256,512,1024) default, 2 for
        # a two-level (128,256) CoLA-sized head.
        div = 2 ** (len(self.down_features) - 1)
        assert self.action_horizon % div == 0, (
            f"UNet with {len(self.down_features)} levels halves the chunk "
            f"{len(self.down_features) - 1}x; action_horizon {self.action_horizon} "
            f"must be divisible by {div}")
        if self.use_map:
            self.map_head = MAPHead()
        self.unet = ConditionalUnet1D(
            down_features=tuple(self.down_features), mid_layers=self.mid_layers,
            kernel_size=self.kernel_size, n_groups=self.n_groups,
            time_features=self.time_features)
        self.action_proj = nn.Dense(self.action_dim)
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.cumprod(self.alphas)

    def __call__(self, transformer_outputs: Dict[str, TokenGroup],
                 time: Optional[ArrayLike] = None,
                 noisy_actions: Optional[ArrayLike] = None,
                 train: bool = True) -> Array:
        """noisy_actions (..., B, W, H, A), time (..., B, W, 1) -> eps, same shape."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4
        if self.use_map:
            emb = self.map_head(token_group, train=train)[:, :, 0]
        else:
            emb = token_group.tokens.mean(axis=-2)            # (B, W, E)

        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError("Must provide time and noisy_actions")
        elif self.is_initializing():
            time = jnp.zeros((*emb.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*emb.shape[:2], self.action_horizon, self.action_dim), dtype=jnp.float32)

        lead = noisy_actions.shape[:-2]                        # (..., B, W)
        emb = jnp.broadcast_to(emb, (*lead, emb.shape[-1]))
        x = noisy_actions.reshape(-1, self.action_horizon, self.action_dim)
        eps = self.unet(emb.reshape(-1, emb.shape[-1]), action=x,
                        time=time.reshape(-1, 1).astype(jnp.float32), train=train)
        eps = self.action_proj(eps)                            # (N, H, A)
        return eps.reshape(*lead, self.action_horizon, self.action_dim)

    def loss(self, transformer_outputs: Dict[str, TokenGroup], actions: ArrayLike,
             timestep_pad_mask: ArrayLike, action_pad_mask: ArrayLike,
             train: bool = True) -> Tuple[Array, Dict[str, Array]]:
        """Same argument order and batch shapes as octo-1.5's DiffusionActionHead."""
        batch_size, window_size = timestep_pad_mask.shape
        actions = jnp.clip(actions, -self.max_action, self.max_action)   # (B, W, H, A)

        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(
            time_key, (self.n_diffusion_samples, batch_size, window_size, 1),
            0, self.diffusion_steps)
        noise = jax.random.normal(noise_key, (self.n_diffusion_samples,) + actions.shape)

        scale = jnp.sqrt(self.alpha_hats[time])[..., None]    # (n, B, W, 1, 1)
        std = jnp.sqrt(1 - self.alpha_hats[time])[..., None]
        noisy_actions = scale * actions[None] + std * noise

        pred_eps = self(transformer_outputs, train=train, time=time,
                        noisy_actions=noisy_actions)

        mask = (timestep_pad_mask[:, :, None, None] & action_pad_mask)[None]
        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(self, transformer_outputs: Dict[str, TokenGroup], rng: PRNGKey,
                       train: bool = True, embodiment_action_dim: Optional[int] = None,
                       *args, sample_shape: tuple = (), **kwargs) -> Array:
        """DDPM ancestral sampling; returns the last window step, (..., B, H, A)."""
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        module, variables = self.unbind()
        shape = (*sample_shape, batch_size, window_size, self.action_horizon, self.action_dim)
        action_mask = jnp.ones(shape, dtype=bool)
        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)

        def scan_fn(carry, t):
            x, rng = carry
            input_time = jnp.broadcast_to(t, (*x.shape[:-2], 1))
            eps = module.apply(variables, transformer_outputs, input_time, x, train=train)
            x = (x - (1 - self.alphas[t]) / jnp.sqrt(1 - self.alpha_hats[t]) * eps) \
                / jnp.sqrt(self.alphas[t])
            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=x.shape)
            x = x + (t > 0) * (jnp.sqrt(self.betas[t]) * z)
            x = jnp.clip(x, -self.max_action, self.max_action)
            x = jnp.where(action_mask, x, jnp.sqrt(1 - self.alpha_hats[t]) * z)
            return (x, rng), ()

        rng, key = jax.random.split(rng)
        (actions, _), () = jax.lax.scan(
            scan_fn, (jax.random.normal(key, shape), rng),
            jnp.arange(self.diffusion_steps - 1, -1, -1))
        return actions[..., -1, :, :]
