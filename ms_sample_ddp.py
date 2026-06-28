#!/usr/bin/env python3
"""
MindSpore DiT Distributed Inference Script
Samples a large number of images from a pre-trained DiT model using multiple NPUs.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Supports Ascend NPU. For a simple single-NPU sampling script, see ms_sample.py.
"""
import argparse
import os
import sys
import numpy as np
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

if 'ASCEND_HOME_PATH' not in os.environ:
    _candidates = [
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/ascend-toolkit/8.3.RC1',
    ]
    for _p in _candidates:
        if os.path.isfile(os.path.join(_p, 'lib64', 'libascendcl.so')):
            os.environ['ASCEND_HOME_PATH'] = _p
            break

import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size
from mindspore import mint

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/data0/ms_models/mindone')

from mindone.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def forward_with_cfg(model, x, t, y, cfg_scale):
    """
    Forward pass with classifier-free guidance.
    CRITICAL: Matches PyTorch implementation exactly - applies CFG to only first 3 channels!
    """
    half = x[:x.shape[0] // 2]
    combined = mint.cat([half, half], dim=0)
    
    model_out = model(combined, t, y)
    
    eps = model_out[:, :3, :, :]
    rest = model_out[:, 3:, :, :]
    
    cond_eps, uncond_eps = mint.split(eps, eps.shape[0] // 2, dim=0)
    
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = mint.cat([half_eps, half_eps], dim=0)
    
    return mint.cat([eps, rest], dim=1)


def main(args):
    # Initialize distributed communication
    init()
    rank = get_rank()
    world_size = get_group_size()
    
    # Set seed for reproducibility (matching PyTorch's seed scheme)
    seed = args.global_seed * world_size + rank
    ms.set_seed(seed)
    np.random.seed(seed)
    
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Saving .png samples at {args.sample_dir}")
    
    # Load DiT model
    print(f"[Rank {rank}] Loading DiT model...")
    latent_size = args.image_size // 8
    
    if args.checkpoint and args.checkpoint.endswith('.ckpt'):
        # MindSpore checkpoint from ms_train.py - use DiTTransformer2DModel
        from mindone.diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
        model = DiTTransformer2DModel(
            in_channels=4,
            out_channels=8,
            patch_size=2,
            num_attention_heads=16,
            attention_head_dim=72,
            num_layers=28,
            sample_size=latent_size,
            num_embeds_ada_norm=args.num_classes,
        )
    else:
        # PyTorch checkpoint or no checkpoint - use DiT_models
        from mindone.models.dit import DiT_models
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=True,
        )
    
    # Load checkpoint
    if args.checkpoint:
        if args.checkpoint.endswith('.pt'):
            pt_ckpt = torch.load(args.checkpoint, map_location='cpu')
            param_dict = {}
            for k, v in pt_ckpt.items():
                param_dict[k] = ms.Parameter(ms.Tensor(v.numpy()))
        else:
            raw_ckpt = ms.load_checkpoint(args.checkpoint)
            # Prefer MODEL weights over EMA (EMA was never updated during MS training)
            model_keys = {k.replace("model.model._backbone.", ""): v for k, v in raw_ckpt.items() if k.startswith("model.model._backbone.")}
            if model_keys:
                param_dict = model_keys
                if "model.pos_embed.pos_embed" in raw_ckpt:
                    param_dict["pos_embed.pos_embed"] = raw_ckpt["model.pos_embed.pos_embed"]
                print(f"[Rank {rank}] Using MODEL weights ({len(model_keys)} params)")
            else:
                param_dict = raw_ckpt
        
        not_loaded, unmatched = ms.load_param_into_net(model, param_dict, strict_load=False)
        if not_loaded:
            print(f"[Rank {rank}] Warning: {len(not_loaded)} parameters not loaded: {not_loaded[:5]}...")
        if unmatched:
            print(f"[Rank {rank}] Info: {len(unmatched)} checkpoint keys unmatched: {unmatched[:3]}...")
        
        # Handle pos_embed separately (not a Parameter, invisible to load_param_into_net)
        if "pos_embed.pos_embed" in param_dict:
            pe_ckpt = param_dict["pos_embed.pos_embed"]
            pe_data = pe_ckpt.data if isinstance(pe_ckpt, ms.Parameter) else pe_ckpt
            model.pos_embed.pos_embed.assign_value(pe_data)
            print(f"[Rank {rank}] Manually set pos_embed.pos_embed from checkpoint")
        else:
            print(f"[Rank {rank}] WARNING: pos_embed.pos_embed not found")
    else:
        print(f"[Rank {rank}] No checkpoint provided, using randomly initialized model")
    
    model.set_train(False)
    print(f"[Rank {rank}] Model loaded, latent_size: {latent_size}")
    
    # Load VAE
    print(f"[Rank {rank}] Loading VAE...")
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
    else:
        vae = AutoencoderKL.from_pretrained("/data0/ms_models/DiT/checkpoints/sd-vae-ft-mse")
    vae = vae.to(ms.float32)
    vae.set_train(False)
    print(f"[Rank {rank}] VAE loaded")
    
    # Use PyTorch's diffusion library for exact matching
    print(f"[Rank {rank}] Creating diffusion with {args.num_sampling_steps} steps...")
    from diffusion import create_diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))
    timestep_map = diffusion.timestep_map
    print(f"[Rank {rank}] Timestep map: {timestep_map[:10]}... (total {len(timestep_map)})")
    
    # Calculate how many samples each rank needs to generate
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    
    assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by per_proc_batch_size"
    iterations = int(samples_needed_this_gpu // n)
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    
    total = 0
    for _ in pbar:
        # Generate noise using PyTorch's RNG (matching PyTorch's seed scheme)
        torch.manual_seed(seed + total)
        z_torch = torch.randn(n, 4, latent_size, latent_size)
        z_np = z_torch.numpy().astype(np.float32)
        z = ms.Tensor(z_np)
        
        # Generate random class labels
        y_np = np.random.randint(0, args.num_classes, (n,)).astype(np.int32)
        y = ms.Tensor(y_np)
        
        # Prepare for CFG
        z = mint.cat([z, z], dim=0)
        y_null = ms.Tensor(np.array([1000] * n, dtype=np.int32))
        y = mint.cat([y, y_null], dim=0)
        
        # DDPM sampling loop using PyTorch's diffusion parameters
        x = z
        timesteps = list(range(args.num_sampling_steps))[::-1]
        
        for i in timesteps:
            # Map respaced timestep to original timestep (for model input)
            original_t = timestep_map[i]
            t = ms.Tensor(np.array([original_t] * x.shape[0], dtype=np.int32))
            
            # Get model prediction with CFG
            model_output = forward_with_cfg(model, x, t, y, args.cfg_scale)
            
            # Split model output into epsilon and variance
            C = 4
            epsilon = model_output[:, :C, :, :]
            model_var_values = model_output[:, C:, :, :]
            
            # Use PyTorch's diffusion parameters (respaced index i)
            sqrt_alpha_cumprod_t = ms.Tensor(np.array([np.sqrt(diffusion.alphas_cumprod[i])]), dtype=ms.float32)
            sqrt_one_minus_alpha_cumprod_t = ms.Tensor(np.array([np.sqrt(1.0 - diffusion.alphas_cumprod[i])]), dtype=ms.float32)
            
            # Predict x_0 (no clamp - matches PT clip_denoised=False)
            pred_xstart = (x - sqrt_one_minus_alpha_cumprod_t * epsilon) / sqrt_alpha_cumprod_t
            
            # Compute posterior variance
            min_log = ms.Tensor(np.array([diffusion.posterior_log_variance_clipped[i]]), dtype=ms.float32)
            max_log = ms.Tensor(np.array([np.log(diffusion.betas[i])]), dtype=ms.float32)
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            
            # Compute mean using PyTorch's posterior coefficients
            coef1 = ms.Tensor(np.array([diffusion.posterior_mean_coef1[i]]), dtype=ms.float32)
            coef2 = ms.Tensor(np.array([diffusion.posterior_mean_coef2[i]]), dtype=ms.float32)
            
            mean = coef1 * pred_xstart + coef2 * x
            
            # Sample x_{t-1} using PyTorch's RNG
            if i > 0:
                noise_torch = torch.randn(x.shape[0], 4, latent_size, latent_size)
                noise = ms.Tensor(noise_torch.numpy())
                x = mean + mint.exp(0.5 * model_log_variance) * noise
            else:
                x = mean
        
        # Take only the conditional part
        x = x[:n]
        
        # Decode with VAE
        decoded = vae.decode(x / 0.18215)
        if isinstance(decoded, tuple):
            samples = decoded[0]
        else:
            samples = decoded.sample
        
        # Convert to numpy and save
        samples_np = samples.asnumpy()
        samples_np = (samples_np + 1.0) / 2.0
        samples_np = np.clip(samples_np, 0, 1)
        samples_np = (samples_np * 255).astype(np.uint8)
        samples_np = np.transpose(samples_np, (0, 2, 3, 1))
        
        # Save samples to disk with interleaved indexing
        for i, sample in enumerate(samples_np):
            index = i * world_size + rank + total
            Image.fromarray(sample).save(f"{args.sample_dir}/{index:06d}.png")
        
        total += global_batch_size
    
    # Rank 0 creates the .npz file
    if rank == 0:
        create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)
        print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="MindSpore DiT Distributed Sampling")
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=["DiT-XL/2", "DiT-L/2", "DiT-B/2"],
                        help="Model name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to DiT checkpoint (PyTorch .pt or MindSpore .ckpt)")
    parser.add_argument("--sample-dir", type=str, default="ms_samples",
                        help="Output directory for samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=8,
                        help="Batch size per NPU")
    parser.add_argument("--num-fid-samples", type=int, default=50000,
                        help="Total number of FID samples to generate")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256,
                        help="Output image size")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-sampling-steps", type=int, default=250,
                        help="Number of denoising steps")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="Global random seed")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to VAE model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
