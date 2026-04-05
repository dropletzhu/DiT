# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DiT Training Pipeline with Diffusion Loss."""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindone.diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

__all__ = ["DiTDiffusionWithLoss", "DDPMTrainLoss"]

logger = logging.getLogger(__name__)


class DiTDiffusionWithLoss(nn.Cell):
    """Training pipeline for DiT model with diffusion noise prediction loss.

    Args:
        network: The DiT transformer model for denoising.
        vae: VAE model for encoding images to latents.
        noise_scheduler: Noise scheduler (DDPM, DDIM, etc.) for diffusion training.
        latent_scaling_factor: Scaling factor for VAE latents.
        class_dropout_prob: Probability of dropping class label for classifier-free guidance.
        latent_channels: Number of latent channels.
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: Optional[nn.Cell] = None,
        noise_scheduler: Optional[Any] = None,
        latent_scaling_factor: float = 0.18215,
        class_dropout_prob: float = 0.1,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.latent_scaling_factor = latent_scaling_factor
        self.class_dropout_prob = class_dropout_prob
        self.latent_channels = latent_channels

        self.num_classes = 1000
        self.null_class_label = self.num_classes

        logger.info(f"Latent scaling factor: {self.latent_scaling_factor}")

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode images to latents using VAE.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            latents: Encoded latents [B, latent_channels, H//8, W//8]
        """
        if self.vae is None:
            return images

        h = self.vae.encode(images, return_dict=False)[0]
        
        # h contains [mean, logvar] of shape (B, 8, H, W)
        # Split into mean and logvar
        mean = h[:, :4, :, :]
        logvar = h[:, 4:, :, :]
        
        # Sample: z = mean + std * noise
        std = ops.exp(0.5 * logvar)
        noise = mint.randn_like(mean)
        latents = mean + std * noise
        
        latents = ops.stop_gradient(latents * self.latent_scaling_factor)

        return latents

    def prepare_class_labels(
        self,
        class_labels: Tensor,
        guidance_scale: Optional[float] = None,
    ) -> Tensor:
        """Prepare class labels for training, handling classifier-free guidance dropout.

        Args:
            class_labels: Original class labels [B]
            guidance_scale: Guidance scale (if CFG is used during training)

        Returns:
            Prepared class labels with null class dropout
        """
        if self.class_dropout_prob > 0:
            mask = mint.rand([class_labels.shape[0]], dtype=ms.float32) < self.class_dropout_prob
            class_labels = mint.where(mask, ms.tensor(self.null_class_label, dtype=ms.int32), class_labels)

        return class_labels

    def construct(
        self,
        images: Tensor,
        class_labels: Tensor,
        latent: Optional[Tensor] = None,
        guidance_scale: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass for training.

        Args:
            images: Input images [B, C, H, W]
            class_labels: Class labels [B]
            latent: Optional pre-computed latents [B, latent_channels, H', W']
            guidance_scale: Optional guidance scale for CFG

        Returns:
            loss: Computed MSE loss
            log_dict: Dictionary containing loss info
        """
        bsz = images.shape[0]

        if latent is None:
            latent = self.encode_images(images)

        latent = latent.to(ms.float32)

        noise = mint.randn_like(latent)
        timesteps = mint.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            dtype=ms.int64,
        )

        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        class_labels = self.prepare_class_labels(class_labels, guidance_scale)

        model_pred = self.network(
            noisy_latent,
            timestep=timesteps,
            class_labels=class_labels,
        )[0]

        loss = ops.mse_loss(model_pred, noise, reduction="mean")

        log_dict = {
            "loss": loss,
            "mse_loss": loss,
            "lr": self.network.learning_rate if hasattr(self.network, "learning_rate") else 0.0,
        }

        return loss, log_dict


class DDPMTrainLoss(nn.Cell):
    """DDPM training loss wrapper that computes noise prediction loss.

    Args:
        network: The DiT transformer model.
        scheduler: The noise scheduler.
        latent_scaling_factor: Scaling factor for VAE latents.
    """

    def __init__(
        self,
        network: nn.Cell,
        scheduler: Any,
        latent_scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.network = network
        self.scheduler = scheduler
        self.latent_scaling_factor = latent_scaling_factor

    def construct(
        self,
        latents: Tensor,
        noise: Tensor,
        timesteps: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """Compute noise prediction loss.

        Args:
            latents: Noisy latents [B, C, H, W]
            noise: Ground truth noise [B, C, H, W]
            timesteps: Timesteps [B]
            class_labels: Class labels [B]

        Returns:
            loss: MSE loss
        """
        model_pred = self.network(
            latents,
            timestep=timesteps,
            class_labels=class_labels,
        )[0]

        loss = ops.mse_loss(model_pred, noise, reduction="mean")
        return loss


class FlowMatchingLoss(nn.Cell):
    """Flow Matching training loss wrapper.

    Args:
        network: The DiT transformer model.
    """

    def __init__(self, network: nn.Cell):
        super().__init__()
        self.network = network

    def construct(
        self,
        x0: Tensor,
        x1: Tensor,
        timesteps: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """Compute velocity prediction loss for flow matching.

        Args:
            x0: Clean latents (noise)
            x1: Target data (clean image)
            timesteps: Timesteps [B]
            class_labels: Class labels [B]

        Returns:
            loss: MSE loss
        """
        t = timesteps.float32() / 1000.0
        t = t.reshape(-1, 1, 1, 1)

        xt = t * x1 + (1 - t) * x0

        velocity = x1 - x0

        model_pred = self.network(
            xt,
            timestep=timesteps,
            class_labels=class_labels,
        )[0]

        loss = ops.mse_loss(model_pred, velocity, reduction="mean")
        return loss
