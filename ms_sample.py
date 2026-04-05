#!/usr/bin/env python3
"""
MindSpore DiT Inference Script using mindone's DiTPipeline
"""
import argparse
import os
import sys
import numpy as np
import mindspore as ms

sys.path.insert(0, '/home/ma-user/work/xql/dongfeng/mindone/examples/dit/scripts')

from mindone.diffusers import DiTTransformer2DModel, AutoencoderKL, DDIMScheduler, DiTPipeline
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="DiT Image Generation Inference on NPU")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MindSpore DiT checkpoint")
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


def main(args):
    # Set seed first
    np.random.seed(args.seed)
    
    # Set context
    ms.set_context(device_target="Ascend", device_id=args.device_id, mode=ms.PYNATIVE_MODE)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting DiT inference on NPU (device_id={args.device_id})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image size: {args.image_size}")
    
    # Create model
    print("Loading DiT model...")
    sample_size = args.image_size // 8
    model = DiTTransformer2DModel(
        in_channels=4,
        out_channels=8,
        patch_size=2,
        num_attention_heads=16,
        attention_head_dim=72,
        num_layers=28,
        sample_size=sample_size,
        num_embeds_ada_norm=1000,
    )
    
    # Load checkpoint
    param_dict = ms.load_checkpoint(args.checkpoint)
    not_loaded = ms.load_param_into_net(model, param_dict, strict_load=False)
    if not_loaded[0]:
        print(f"Warning: {len(not_loaded[0])} parameters not loaded")
    
    # Set pos_embed from PyTorch checkpoint
    pt_ckpt = torch.load('pretrained_models/DiT-XL-2-256x256.pt', map_location='cpu')
    for param in model.get_parameters():
        if param.name == 'pos_embed.pos_embed':
            param.set_data(ms.Tensor(pt_ckpt['pos_embed'].numpy()))
            break
    
    model.set_train(False)
    print(f"Model loaded, sample_size: {sample_size}")
    
    # Load VAE
    print("Loading VAE...")
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
    else:
        vae = AutoencoderKL.from_pretrained("/home/ma-user/work/temp/sd-vae-ft-mse")
    vae = vae.to(ms.float32)
    vae.set_train(False)
    print("VAE loaded")
    
    # Create scheduler
    print("Creating scheduler...")
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(args.num_inference_steps)
    print(f"Scheduler loaded with {args.num_inference_steps} steps")
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = DiTPipeline(
        transformer=model,
        vae=vae,
        scheduler=scheduler,
    )
    
    # Generate images
    class_labels = [207, 360, 387, 974][:args.num_images]
    print(f"Class labels: {class_labels}")
    print(f"Generating {args.num_images} images...")
    
    # Use numpy random generator
    generator = np.random.default_rng(args.seed)
    
    # Generate
    output = pipeline(
        class_labels=class_labels,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
        return_dict=True,
    )
    
    # Save images
    for i, img in enumerate(output.images):
        img.save(os.path.join(args.output_dir, f"generated_{i:04d}.png"))
    
    print(f"Saved {len(output.images)} images to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main(parse_args())