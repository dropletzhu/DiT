# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Inference Script for Ascend NPU

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import mint, ops

sys.path.insert(0, str(Path(__file__).parent))

from ms_models import DiT_models, DiT
from diffusion import create_diffusion
from mindone.diffusers import AutoencoderKL

os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'


class WrappedDiTModel(nn.Cell):
    """Wrapper for DiT model to handle classifier-free guidance - matching PyTorch's forward_with_cfg"""
    def __init__(self, model, cfg_scale=4.0):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale
    
    def construct(self, x, t, y):
        # When x has more samples than y, apply CFG
        # x has shape [2*batch_size, 4, H, W] and y has shape [batch_size] 
        if y.shape[0] * 2 == x.shape[0]:
            half = y.shape[0]
            x1 = x[:half]
            x2 = x[half:]
            combined = mint.cat([x1, x2], dim=0)
            combined_y = mint.cat([y, y], dim=0)
            
            output = self.model(combined, t, combined_y)
            
            eps = output[:, :3, :, :]
            rest = output[:, 3:, :, :]
            
            eps_uncond, eps_cond = eps[:half], eps[half:]
            eps_cfg = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
            
            return mint.cat([eps_cfg, rest], dim=1)
        else:
            # No CFG needed
            output = self.model(x, t, y)
            return output


def parse_args():
    parser = argparse.ArgumentParser(description="DiT Image Generation Inference on NPU")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MindSpore DiT checkpoint")
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./ms_output", help="Output directory")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per generation")
    parser.add_argument("--num_inference_steps", type=int, default=250, help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device ID")
    return parser.parse_args()


def set_seed(seed: int):
    ms.set_seed(seed)
    np.random.seed(seed)


def load_dit_model(checkpoint_path: str, model_name: str, image_size: int = 256) -> DiT:
    """Load DiT model from MindSpore checkpoint."""
    latent_size = image_size // 8
    
    model = DiT_models[model_name]()
    
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.ckpt'):
        param_dict = ms.load_checkpoint(checkpoint_path)
        not_load = ms.load_param_into_net(model, param_dict)
        
        if not_load:
            print(f"Warning: {len(not_load)} parameters not loaded")
            for name in not_load[:5]:
                print(f"  Not loaded: {name}")
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Using random weights (no checkpoint loaded)")

    model.set_train(False)
    return model


def generate_images(
    model: DiT,
    vae: AutoencoderKL,
    diffusion,
    class_labels: list,
    latent_size: int,
    num_inference_steps: int = 250,
    guidance_scale: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate images using DDIM sampling matching PyTorch sample.py logic."""
    import numpy as onp
    
    batch_size = len(class_labels)
    
    ms.set_seed(seed)
    latents = mint.randn((batch_size, 4, latent_size, latent_size), dtype=ms.float32)
    
    class_labels_tensor = ms.Tensor(class_labels, dtype=ms.int32)
    class_null = ms.Tensor([1000] * batch_size, dtype=ms.int32)
    
    timesteps = sorted(diffusion.use_timesteps, reverse=True)
    alphas_cumprod = diffusion.alphas_cumprod
    alphas_cumprod_prev = diffusion.alphas_cumprod_prev
    
    # Pre-broadcast t_tensor
    t_tensor_base = ms.Tensor([0], dtype=ms.int32)
    
    for i, t in enumerate(timesteps):
        # Create timestep tensor with proper shape
        t_tensor = ms.Tensor([t], dtype=ms.int32)
        
        # CFG: duplicate latent and labels
        latent_model_input = mint.cat([latents, latents], dim=0)
        class_labels_input = mint.cat([class_labels_tensor, class_null], dim=0)
        
        # Create properly shaped timestep
        t_expanded = t_tensor.reshape(1)
        t_expanded = t_expanded.broadcast_to((latent_model_input.shape[0],))
        
        # Run model
        noise_pred = model(latent_model_input, t_expanded, class_labels_input)
        
        idx = diffusion.timestep_map.index(t) if t in diffusion.timestep_map else i
        alpha_bar = alphas_cumprod[idx]
        alpha_bar_prev = alphas_cumprod_prev[idx]
        
        sqrt_alpha_bar = onp.sqrt(alpha_bar)
        sqrt_1_alpha_bar = onp.sqrt(1 - alpha_bar)
        sqrt_alpha_bar_prev = onp.sqrt(alpha_bar_prev)
        sqrt_1_alpha_bar_prev = onp.sqrt(1 - alpha_bar_prev)
        
        # Extract eps from noise_pred and apply CFG (matching PyTorch's forward_with_cfg)
        eps_full = noise_pred[:, :3, :, :]
        rest = noise_pred[:, 3:, :, :]
        
        eps_uncond, eps_cond = eps_full[:batch_size], eps_full[batch_size:]
        eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # Compute pred_xstart
        pred_xstart = (latents - mint.mul(mint.Tensor(sqrt_1_alpha_bar, dtype=ms.float32), eps_cfg)) / mint.Tensor(sqrt_alpha_bar, dtype=ms.float32)
        pred_xstart = mint.clip(pred_xstart, -1, 1)
        
        # Compute mean prediction
        mean_pred = (
            mint.mul(mint.Tensor(sqrt_alpha_bar_prev, dtype=ms.float32), pred_xstart)
            + mint.mul(mint.Tensor(sqrt_1_alpha_bar_prev, dtype=ms.float32), (latents - mint.mul(mint.Tensor(sqrt_alpha_bar, dtype=ms.float32), pred_xstart)) / mint.Tensor(sqrt_1_alpha_bar, dtype=ms.float32))
        )
        
        if i < len(timesteps) - 1:
            latents = mean_pred + mint.randn_like(latents) * 0.0
        else:
            latents = mean_pred
    
    latents = 1 / 0.18215 * latents
    samples = vae.decode(latents)[0]
    
    samples = (samples / 2 + 0.5).clip(0, 1)
    samples = samples.permute(0, 2, 3, 1).float().asnumpy()
    
    return samples


def main(args):
    ms.set_context(device_target="Ascend", device_id=args.device_id, mode=ms.PYNATIVE_MODE)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting DiT inference on NPU (device_id={args.device_id})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.image_size}")
    
    set_seed(args.seed)
    
    sample_size = args.image_size // 8
    
    print("Loading DiT model...")
    model = load_dit_model(args.checkpoint, args.model, args.image_size)
    print(f"Model loaded, sample_size: {sample_size}")
    
    print("Loading VAE...")
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
    else:
        vae = AutoencoderKL.from_pretrained("/home/ma-user/work/temp/sd-vae-ft-mse")
    vae = vae.to(ms.float32)
    vae.set_train(False)
    print("VAE loaded")
    
    print("Loading diffusion...")
    diffusion = create_diffusion(str(args.num_inference_steps))
    print(f"Diffusion loaded with {args.num_inference_steps} steps")
    
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279][:args.num_images]
    if len(class_labels) < args.num_images:
        class_labels = class_labels * (args.num_images // len(class_labels) + 1)
    class_labels = class_labels[:args.num_images]
    
    print(f"Class labels: {class_labels}")
    print(f"Generating {args.num_images} images...")
    
    all_images = []
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_images)
        batch_labels = class_labels[start_idx:end_idx]
        
        print(f"Generating batch {batch_idx + 1}/{num_batches} with labels {batch_labels}")
        
        images = generate_images(
            model=model,
            vae=vae,
            diffusion=diffusion,
            class_labels=batch_labels,
            latent_size=sample_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg_scale,
            seed=args.seed + batch_idx,
        )
        
        for i, img in enumerate(images):
            img_path = os.path.join(args.output_dir, f"generated_{start_idx + i:04d}.png")
            from PIL import Image
            Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        
        all_images.extend(images)
        print(f"Batch {batch_idx + 1} completed")
    
    print(f"Saved {len(all_images)} images to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main(parse_args())