# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Sampling Script for Ascend NPU with ONNX VAE

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindspore'))

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from PIL import Image
from tqdm import tqdm
import math

from ms_models import DiT_models
from vae_utils import MindSporeVAE


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000.0 / num_diffusion_timesteps
        betas = np.linspace(scale * 1e-4, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "squaredcos_cap_v2":
        betas = betas_for_alpha_bar(num_diffusion_timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")
    return betas.astype(np.float32)


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(base_num_steps, selection):
    if isinstance(selection, str):
        if selection.startswith("ddim"):
            step_str = selection[4:]
            if not step_str:
                num_steps = base_num_steps
            else:
                num_steps = int(step_str)
        else:
            num_steps = int(selection)
        
        if num_steps < base_num_steps:
            ratio = base_num_steps / num_steps
            steps = np.arange(0, num_steps) * ratio + ratio - 1
            steps = np.round(steps).astype(int).tolist()
            return [int(s) for s in steps]
        else:
            return list(range(base_num_steps))
    
    if len(selection) == base_num_steps:
        return selection
    
    assert len(selection) < base_num_steps
    gap = base_num_steps // len(selection)
    return [i * gap for i in range(len(selection))]


class GaussianDiffusion:
    def __init__(self, betas, num_classes=1000, timesteps=None):
        self.num_timesteps = len(betas)
        self.num_classes = num_classes
        self.timesteps = timesteps if timesteps is not None else list(range(self.num_timesteps))
        betas = ms.Tensor(betas, dtype=ms.float32)
        alphas = 1.0 - betas
        self.alphas_cumprod = ops.CumProd()(alphas, 0)
        
        self.betas = betas
        self.alphas_cumprod_prev = ops.Concat(0)(
            [ms.Tensor([1.0], dtype=ms.float32), self.alphas_cumprod[:-1]]
        )
        self.sqrt_recip_alphas_cumprod = ops.Sqrt()(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ops.Sqrt()(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = betas * (self.alphas_cumprod_prev / self.alphas_cumprod)
        self.posterior_variance = ops.Concat(0)(
            [ms.Tensor([0.0], dtype=ms.float32), self.posterior_variance[1:]]
        )
        self.posterior_log_variance_clipped = ops.Log()(ops.Maximum()(self.posterior_variance, ms.Tensor(1e-20, dtype=ms.float32)))
        self.posterior_mean_coef1 = betas * ops.Sqrt()(self.alphas_cumprod_prev / self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * ops.Sqrt()(alphas / self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = ops.GatherD()(a.expand_dims(0), 0, t.reshape(-1, 1)).reshape(batch_size, *([1] * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = ops.StandardNormal()(x_start.shape).astype(ms.float32)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_start.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x, t, y, cfg_scale, clip_denoised=True):
        half = x.shape[0] // 2
        
        combined = ops.Concat(0)([x[:half], x[:half]])
        
        model_output = model(combined, t, y, cfg_scale)
        
        x_recon = self.predict_start_from_noise(x, t, model_output[:, :4])
        
        if clip_denoised:
            x_recon = ops.clip_by_value(x_recon, -1.0, 1.0)
        
        model_mean, posterior_variance, model_log_variance = self.q_posterior_mean_variance(x_recon, x, t)
        
        return model_mean, posterior_variance, model_log_variance

    def p_sample(self, model, x, t, y, cfg_scale, clip_denoised=True):
        batch_size = x.shape[0]
        t_tensor = ms.Tensor([t] * batch_size, dtype=ms.int32)
        
        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t_tensor, y, cfg_scale, clip_denoised)
        
        noise = ops.StandardNormal()(x.shape).astype(ms.float32)
        
        if t > 0:
            return model_mean + ops.Exp()(0.5 * model_log_variance) * noise
        else:
            return model_mean

    def p_sample_loop(self, model, shape, y, cfg_scale, progress=True):
        x = ops.StandardNormal()(shape).astype(ms.float32)
        
        indices = self.timesteps
        if progress:
            indices = tqdm(indices, desc="Sampling")
        
        for i in indices:
            x = self.p_sample(model, x, i, y, cfg_scale, clip_denoised=True)
        
        return x


def main(args):
    ms.set_context(device_target="Ascend", device_id=0)
    ms.set_seed(args.seed)
    
    print("Starting MindSpore DiT inference with VAE...", flush=True)
    
    latent_size = args.image_size // 8
    
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    model.set_train(False)
    print(f"Model created: {args.model}")
    
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}...")
        ms.load_checkpoint(args.ckpt, model, strict_load=True)
        print("Checkpoint loaded successfully")
    
    betas = get_named_beta_schedule("linear", 1000)
    # Rescale betas to match PyTorch: scale = 1000/250 = 4
    # PyTorch uses 250 steps with scale factor 4
    scale = 1000.0 / args.num_sampling_steps
    betas = betas * scale
    # Then sample from these betas using space_timesteps
    timesteps = space_timesteps(1000, str(args.num_sampling_steps))
    diffusion = GaussianDiffusion(betas, num_classes=args.num_classes, timesteps=timesteps)
    print(f"Diffusion created with 1000 base steps, sampling with {len(timesteps)} steps (timesteps: {timesteps[:5]}...{timesteps[-5:]})")
    
    if args.vae_path:
        print(f"Loading VAE from {args.vae_path}...")
        vae = MindSporeVAE(args.vae_path)
        print("VAE loaded successfully")
    
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    
    z = ops.StandardNormal()((n * 2, 4, latent_size, latent_size)).astype(ms.float32)
    y = ms.Tensor(class_labels + [1000] * n, dtype=ms.int32)
    
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, y, args.cfg_scale, progress=True
    )
    
    samples, _ = ops.Split(0, 2)(samples)
    
    if args.vae_path:
        print("Decoding latents with VAE...")
        samples_np = samples.asnumpy()
        samples_np = samples_np / 0.18215
        samples = vae.decode(samples_np)
    else:
        samples_np = samples.asnumpy() / 0.18215
    
    samples_np = (samples_np - samples_np.min()) / (samples_np.max() - samples_np.min() + 1e-8)
    samples_np = (samples_np * 255).astype(np.uint8)
    
    C, H, W = samples_np.shape[1:]
    
    if C == 3:
        img_array = samples_np.transpose(0, 2, 3, 1)
    elif C == 4:
        img_array = samples_np[:, :3].transpose(0, 2, 3, 1)
    else:
        img_array = samples_np.transpose(0, 2, 3, 1)
    
    rows, cols = 2, 4
    img_h, img_w = H, W
    grid = np.zeros((img_h * rows, img_w * cols, 3), dtype=np.uint8)
    
    for idx in range(min(n, 8)):
        row = idx // cols
        col = idx % cols
        grid[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img_array[idx]
    
    Image.fromarray(grid).save(args.output)
    print(f"Saved 8 samples as {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to DiT checkpoint")
    parser.add_argument("--vae-path", type=str, default=None, help="Path to VAE ONNX model")
    parser.add_argument("--output", type=str, default="ms_sample.png", help="Output image file")
    args = parser.parse_args()
    main(args)
