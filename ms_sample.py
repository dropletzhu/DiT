# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Sampling Script for Ascend NPU

import argparse
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from PIL import Image
from tqdm import tqdm

from ms_models import DiT_models


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        betas = np.linspace(1e-4, 0.02, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")
    return betas.astype(np.float32)


class SimpleDDPM:
    def __init__(self, betas):
        self.num_timesteps = len(betas)
        betas = ms.Tensor(betas, dtype=ms.float32)
        alphas = 1.0 - betas
        self.alphas_cumprod = ops.CumProd()(alphas, 0)
    
    def p_sample(self, model, x, y, cfg_scale):
        # Simple forward with CFG
        t_tensor = ms.Tensor([0] * x.shape[0], dtype=ms.int32)
        model_output = model(x, t_tensor, y, cfg_scale)
        
        # Simple denoising
        x_prev = x - model_output[:, :4] * 0.1
        
        return x_prev
    
    def p_sample_loop(self, model, shape, y, cfg_scale, progress=True):
        x = ops.StandardNormal()(shape).astype(ms.float32)
        
        print(f"Initial x shape: {x.shape}")
        
        indices = list(range(self.num_timesteps))
        if progress:
            indices = tqdm(indices, desc="Sampling")
        
        for i in indices:
            x = self.p_sample(model, x, y, cfg_scale)
        
        return x


def main(args):
    import sys
    ms.set_context(device_target="Ascend", device_id=0)
    ms.set_seed(args.seed)
    
    print("Starting MindSpore DiT inference...", flush=True)
    
    # Create model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    model.set_train(False)
    print(f"Model created: {args.model}")
    
    # Create diffusion
    betas = get_named_beta_schedule("linear", args.num_sampling_steps)
    diffusion = SimpleDDPM(betas)
    print(f"Diffusion created with {args.num_sampling_steps} steps")
    
    # Sample
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279][:args.num_samples]
    n = len(class_labels)
    z = ms.numpy.randn(n, 4, latent_size, latent_size).astype(np.float32)
    y = ms.Tensor(class_labels, dtype=ms.int32)
    
    print(f"Generating {n} samples...")
    # Duplicate for CFG - double the batch
    z = ops.Concat(0)([z, z])
    y_null = ms.Tensor([1000] * n, dtype=ms.int32)
    y = ops.Concat(0)([y, y_null])
    
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, y, cfg_scale=args.cfg_scale, progress=True
    )
    
    # Convert to numpy and save
    samples_np = samples.asnumpy()
    samples_np = (samples_np - samples_np.min()) / (samples_np.max() - samples_np.min() + 1e-8)
    samples_np = (samples_np * 255).astype(np.uint8)
    
    # Save samples
    for i in range(min(n, 8)):
        img = Image.fromarray(samples_np[i].transpose(1, 2, 0))
        img.save(f"ms_sample_{i}.png")
    
    print(f"Saved {n} samples to ms_sample_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
