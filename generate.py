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

"""Inference script for DiT (Diffusion Transformer) model."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint, ops

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mindone.diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiTTransformer2DModel,
    DiTPipeline,
    DPMSolverMultistepScheduler,
)
from mindone.utils import set_logger

# Parse args first to get mode
def parse_args():
    parser = argparse.ArgumentParser(description="DiT Image Generation Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Output directory")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Model dtype")
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddim", "ddpm", "dpm"], help="Scheduler type")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter")
    parser.add_argument("--class_labels", type=str, default=None, help="Comma-separated class labels (e.g., 'tench,goldfish,hen') or 'random'")
    parser.add_argument("--pretrained_model", type=str, default=None, help="HuggingFace model ID or path (alternative to checkpoint)")
    parser.add_argument("--vae_model", type=str, default=None, help="VAE model path or HuggingFace model ID")
    parser.add_argument("--mode", type=str, default="pynative", choices=["pynative", "graph"], help="MindSpore execution mode")
    return parser.parse_args()

args = parse_args()
ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE)

logger = logging.getLogger(__name__)

IMAGENET_CLASS_LABELS = {
    "tench": 0,
    "goldfish": 1,
    "great_white_shark": 2,
    "tiger_shark": 3,
    "hammerhead": 4,
    "electric_ray": 5,
    "cock": 6,
    "hen": 7,
    "ostrich": 8,
    "brambling": 9,
    "goldfinch": 10,
    "house_finch": 11,
    "robin": 12,
    "bulbul": 13,
    "jay": 14,
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    ms.set_seed(seed)
    np.random.seed(seed)


def load_dit_model(
    checkpoint_path: Optional[str],
    pretrained_model: Optional[str],
    image_size: int,
    dtype: str = "fp32",
) -> Tuple[DiTTransformer2DModel, int]:
    """Load DiT model from checkpoint or pretrained.

    Args:
        checkpoint_path: Path to checkpoint file
        pretrained_model: HuggingFace model ID or path
        image_size: Image size for generation
        dtype: Model dtype

    Returns:
        Tuple of (model, sample_size)
    """
    if pretrained_model:
        logger.info(f"Loading pretrained model: {pretrained_model}")
        model = DiTTransformer2DModel.from_pretrained(pretrained_model)
        sample_size = model.config.sample_size
    elif checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            model = DiTTransformer2DModel.from_pretrained(checkpoint_path)
            sample_size = model.config.sample_size
        else:
            sample_size = image_size // 8
            model = DiTTransformer2DModel(
                in_channels=4,
                out_channels=4,
                patch_size=2,
                num_attention_heads=16,
                attention_head_dim=72,
                num_layers=28,
                sample_size=sample_size,
                num_embeds_ada_norm=1000,
            )
            param_dict = ms.load_checkpoint(checkpoint_path)
            ms.load_param_into_net(model, param_dict)
            logger.info(f"Loaded checkpoint with {len(param_dict)} parameters")
    else:
        raise ValueError("Either --checkpoint or --pretrained_model must be specified")

    if dtype == "fp16":
        model = model.to(ms.float16)
    elif dtype == "bf16":
        model = model.to(ms.bfloat16)

    return model, sample_size


def generate_with_pipeline(
    pipeline: DiTPipeline,
    class_labels: List[int],
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    generator: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Generate images using DiT pipeline.

    Args:
        pipeline: DiT pipeline
        class_labels: List of class labels
        num_images_per_prompt: Number of images per prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        generator: Random generator

    Returns:
        List of generated images
    """
    output = pipeline(
        class_labels=class_labels,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        return_dict=True,
    )

    return output.images


def generate_step_by_step(
    model: DiTTransformer2DModel,
    vae: AutoencoderKL,
    scheduler: Any,
    class_labels: List[int],
    latent_size: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    eta: float = 0.0,
    generator: Optional[np.random.Generator] = None,
    dtype: str = "fp32",
) -> np.ndarray:
    """Generate images step by step without pipeline.

    Args:
        model: DiT model
        vae: VAE model
        scheduler: Scheduler
        class_labels: List of class labels
        latent_size: Latent grid size
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        eta: DDIM eta parameter
        generator: Random generator
        dtype: Model dtype

    Returns:
        Generated images as numpy array
    """
    batch_size = len(class_labels)

    latents = ms.ops.randn(
        (batch_size, 4, latent_size, latent_size),
        dtype=ms.float32 if dtype == "fp32" else (ms.float16 if dtype == "fp16" else ms.bfloat16),
    )
    if generator is not None:
        latents = ms.Tensor.from_numpy(generator.random(latents.shape).astype(np.float32))

    scheduler.set_timesteps(num_inference_steps)

    class_labels_tensor = ms.Tensor(class_labels, dtype=ms.int32)
    class_null = ms.Tensor([1000] * batch_size, dtype=ms.int32)

    for i, t in enumerate(scheduler.timesteps):
        latent_model_input = mint.cat([latents] * 2) if guidance_scale > 1 else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        class_labels_input = mint.cat([class_labels_tensor, class_null], 0) if guidance_scale > 1 else class_labels_tensor

        timesteps = t
        if not ops.is_tensor(timesteps):
            timesteps = ms.tensor([timesteps], dtype=ms.int32)
        timesteps = timesteps.broadcast_to((latent_model_input.shape[0],))

        noise_pred = model(latent_model_input, timestep=timesteps, class_labels=class_labels_input)[0]

        if guidance_scale > 1:
            eps, rest = noise_pred[:, :4], noise_pred[:, 4:]
            cond_eps, uncond_eps = mint.split(eps, batch_size, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = mint.cat([half_eps, half_eps], dim=0)
            noise_pred = mint.cat([eps, rest], dim=1)

        latents = scheduler.step(noise_pred, t, latents, eta=eta)[0]

    latents = 1 / vae.config.scaling_factor * latents
    samples = vae.decode(latents)[0]

    samples = (samples / 2 + 0.5).clip(0, 1)
    samples = samples.permute(0, 2, 3, 1).float().asnumpy()

    return samples


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_logger(name="", output_dir=args.output_dir, rank=0)
    logger.info(f"Arguments: {args}")

    set_seed(args.seed)

    dtype_map = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}

    sample_size = args.image_size // 8  # latent size = image_size / 8

    if args.pretrained_model:
        logger.info(f"Loading pretrained model: {args.pretrained_model}")
        model = DiTTransformer2DModel.from_pretrained(args.pretrained_model)
        sample_size = model.config.sample_size
    else:
        # sample_size already computed above based on image_size
        model = DiTTransformer2DModel(
            in_channels=4,
            out_channels=8,  # learn_sigma=True to match PyTorch DiT
            patch_size=2,
            num_attention_heads=16,
            attention_head_dim=72,
            num_layers=28,
            sample_size=sample_size,
            num_embeds_ada_norm=1000,
        )
        param_dict = ms.load_checkpoint(args.checkpoint)
        not_loaded = ms.load_param_into_net(model, param_dict, strict_load=False)
        if not_loaded:
            logger.warning(f"Parameters not fully loaded, {len(not_loaded)} params missing")
        
        # Set PyTorch pos_embed (it wasn't saved in checkpoint due to mindspore issue)
        # Load the correct PyTorch checkpoint based on image_size
        import torch
        if args.image_size == 512:
            pt_checkpoint_path = "/home/ma-user/work/temp/pretrained-pytorch/DiT-XL-2-512x512.pt"
        else:
            pt_checkpoint_path = "/home/ma-user/work/temp/pretrained-pytorch/DiT-XL-2-256x256.pt"
        pt_ckpt = torch.load(pt_checkpoint_path, map_location='cpu')
        if 'pos_embed' in pt_ckpt:
            pos_embed_tensor = ms.Tensor(pt_ckpt['pos_embed'].numpy())
            for param in model.get_parameters():
                if param.name == "pos_embed.pos_embed":
                    param.set_data(pos_embed_tensor)
                    logger.info(f"Set PyTorch pos_embed, row 1 sample: {param.asnumpy()[0, 1, :5]}")
                    break
        
        logger.info(f"Loaded checkpoint with {len(param_dict)} parameters")

    logger.info(f"Model loaded, sample_size: {sample_size}")

    logger.info(f"Loading VAE from {args.vae_model}...")
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    vae = vae.to(dtype_map[args.dtype])
    vae.set_train(False)

    logger.info("Loading scheduler...")
    if args.scheduler == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
    elif args.scheduler == "dpm":
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
        )
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
        )

    logger.info("Creating pipeline...")
    pipeline = DiTPipeline(
        transformer=model,
        vae=vae,
        scheduler=scheduler,
    )

    if args.class_labels and args.class_labels != "random":
        label_names = [l.strip() for l in args.class_labels.split(",")]
        class_labels = [IMAGENET_CLASS_LABELS.get(name, 0) for name in label_names]
        if len(class_labels) < args.num_images:
            class_labels = (class_labels * args.num_images)[: args.num_images]
    elif args.class_labels == "random":
        class_labels = [np.random.randint(0, 1000) for _ in range(args.num_images)]
    else:
        # Default to hen (class 7) to match PyTorch DiT reference
        class_labels = [7] * args.num_images

    logger.info(f"Class labels: {class_labels}")

    logger.info("Generating images...")
    generator = np.random.default_rng(args.seed)

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    all_images = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_images)
        batch_labels = class_labels[start_idx:end_idx]

        logger.info(f"Generating batch {batch_idx + 1}/{num_batches} with labels {batch_labels}")

        images = generate_with_pipeline(
            pipeline=pipeline,
            class_labels=batch_labels,
            num_images_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        all_images.extend(images)

    logger.info(f"Saving {len(all_images)} images to {args.output_dir}")
    for i, img in enumerate(all_images):
        img_path = os.path.join(args.output_dir, f"generated_{i:04d}.png")
        img.save(img_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
