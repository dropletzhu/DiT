#!/usr/bin/env python3
"""
MindSpore DiT Inference Script - Fully Aligned with PyTorch sample.py
Uses PyTorch's diffusion library for exact matching.
"""
import argparse
import os
import sys
import numpy as np
import mindspore as ms
from mindspore import mint
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/data0/ms_models/mindone')

from mindone.models.dit import DiT_models
from mindone.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


def forward_with_cfg(model, x, t, y, cfg_scale):
    """
    Forward pass with classifier-free guidance.
    CRITICAL: Matches PyTorch implementation exactly - applies CFG to only first 3 channels!
    """
    half = x[:x.shape[0] // 2]
    combined = mint.cat([half, half], dim=0)
    
    # Forward pass - model returns 8 channels (4 epsilon + 4 variance)
    model_out = model(combined, t, y)
    
    # CRITICAL: Apply CFG to only first 3 channels (matching PyTorch)
    eps = model_out[:, :3, :, :]
    rest = model_out[:, 3:, :, :]
    
    # Split into conditional and unconditional
    cond_eps, uncond_eps = mint.split(eps, eps.shape[0] // 2, dim=0)
    
    # Apply CFG
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = mint.cat([half_eps, half_eps], dim=0)
    
    # Return full 8-channel output
    return mint.cat([eps, rest], dim=1)


def main(args):
    # Set seed
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set context
    ms.set_context(device_target="Ascend", device_id=args.device_id, mode=ms.PYNATIVE_MODE)
    
    print(f"Starting DiT inference on NPU (device_id={args.device_id})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image size: {args.image_size}")
    print(f"Seed: {args.seed}")
    
    # Create model
    print("Loading DiT model...")
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=True,
    )
    
    # Load checkpoint
    print("Loading checkpoint...")
    import torch
    if args.checkpoint.endswith('.pt'):
        pt_ckpt = torch.load(args.checkpoint, map_location='cpu')
        param_dict = {}
        for k, v in pt_ckpt.items():
            param_dict[k] = ms.Parameter(ms.Tensor(v.numpy()))
    else:
        param_dict = ms.load_checkpoint(args.checkpoint)
    
    not_loaded = ms.load_param_into_net(model, param_dict, strict_load=False)
    if not_loaded:
        print(f"Warning: {len(not_loaded)} parameters not loaded")
    
    model.set_train(False)
    print(f"Model loaded, latent_size: {latent_size}")
    
    # Load VAE
    print("Loading VAE...")
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
    else:
        vae = AutoencoderKL.from_pretrained("/data0/ms_models/DiT/checkpoints/sd-vae-ft-mse")
    vae = vae.to(ms.float32)
    vae.set_train(False)
    print("VAE loaded")
    
    # Use PyTorch's diffusion library for exact matching
    print(f"Creating diffusion with {args.num_sampling_steps} steps...")
    from diffusion import create_diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Get the timestep map (maps respaced timesteps to original timesteps)
    timestep_map = diffusion.timestep_map
    print(f"Timestep map: {timestep_map[:10]}... (total {len(timestep_map)})")
    
    # Generate class labels - EXACTLY match PyTorch sample.py
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    
    # Generate noise - use PyTorch's random generator to match PyTorch version
    torch.manual_seed(args.seed)
    z_torch = torch.randn(n, 4, latent_size, latent_size)
    z_np = z_torch.numpy().astype(np.float32)
    z = ms.Tensor(z_np)
    y = ms.Tensor(class_labels, dtype=ms.int32)
    
    # Prepare for CFG
    z = mint.cat([z, z], dim=0)
    y_null = ms.Tensor([1000] * n, dtype=ms.int32)
    y = mint.cat([y, y_null], dim=0)
    
    print(f"Generating {n} images with CFG scale {args.cfg_scale}...")
    print(f"Class labels: {class_labels}")
    
    # Generate noise using PyTorch's random generator to match PyTorch version
    import torch
    torch.manual_seed(args.seed)
    z_torch = torch.randn(n, 4, latent_size, latent_size)
    z_np = z_torch.numpy().astype(np.float32)
    z = ms.Tensor(z_np)
    
    y = ms.Tensor(class_labels, dtype=ms.int32)
    
    # Prepare for CFG
    z = mint.cat([z, z], dim=0)
    y_null = ms.Tensor([1000] * n, dtype=ms.int32)
    y = mint.cat([y, y_null], dim=0)
    
    # DDPM sampling loop - using PyTorch's diffusion parameters
    samples = z
    timesteps = list(range(args.num_sampling_steps))[::-1]
    
    for i in tqdm(timesteps):
        # Map respaced timestep to original timestep (for model input)
        original_t = timestep_map[i]
        t = ms.Tensor([original_t] * samples.shape[0], dtype=ms.int32)
        
        # Get model prediction with CFG
        model_output = forward_with_cfg(model, samples, t, y, args.cfg_scale)
        
        # Split model output into epsilon and variance
        C = 4
        epsilon = model_output[:, :C, :, :]
        model_var_values = model_output[:, C:, :, :]
        
        # Use PyTorch's diffusion parameters (respaced index i)
        sqrt_alpha_cumprod_t = ms.Tensor(np.array([np.sqrt(diffusion.alphas_cumprod[i])]), dtype=ms.float32)
        sqrt_one_minus_alpha_cumprod_t = ms.Tensor(np.array([np.sqrt(1.0 - diffusion.alphas_cumprod[i])]), dtype=ms.float32)
        
        # Predict x_0
        pred_xstart = (samples - sqrt_one_minus_alpha_cumprod_t * epsilon) / sqrt_alpha_cumprod_t
        pred_xstart = mint.clamp(pred_xstart, -1.0, 1.0)
        
        # Compute posterior variance
        min_log = ms.Tensor(np.array([diffusion.posterior_log_variance_clipped[i]]), dtype=ms.float32)
        max_log = ms.Tensor(np.array([np.log(diffusion.betas[i])]), dtype=ms.float32)
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1.0 - frac) * min_log
        
        # Compute mean using PyTorch's posterior coefficients
        coef1 = ms.Tensor(np.array([diffusion.posterior_mean_coef1[i]]), dtype=ms.float32)
        coef2 = ms.Tensor(np.array([diffusion.posterior_mean_coef2[i]]), dtype=ms.float32)
        
        mean = coef1 * pred_xstart + coef2 * samples
        
        # Sample x_{t-1} - generate noise using PyTorch's RNG to match exactly
        if i > 0:
            # Generate noise with same shape as samples, using PyTorch's RNG
            noise_torch = torch.randn(samples.shape[0], 4, latent_size, latent_size)
            noise = ms.Tensor(noise_torch.numpy())
            samples = mean + mint.exp(0.5 * model_log_variance) * noise
        else:
            samples = mean
    
    # Take only the conditional part
    samples = samples[:n]
    
    print("Decoding with VAE...")
    # Decode with VAE
    decoded = vae.decode(samples / 0.18215)
    if isinstance(decoded, tuple):
        samples = decoded[0]
    else:
        samples = decoded.sample
    
    # Convert to numpy and save using torchvision's save_image (matches PyTorch exactly)
    samples_np = samples.asnumpy()
    
    import torch
    from torchvision.utils import save_image
    samples_tensor = torch.from_numpy(samples_np)
    save_image(samples_tensor, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    
    print(f"Saved sample.png")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT Image Generation Inference on NPU")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiT checkpoint")
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="Model name")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--num_sampling_steps", type=int, default=250, help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE model")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device ID")
    args = parser.parse_args()
    main(args)
