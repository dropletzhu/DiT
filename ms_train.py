# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Training Script for Ascend NPU

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import mindspore as ms
from mindspore import mint, nn, ops

sys.path.insert(0, str(Path(__file__).parent))

os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'


class RandomLatentDataset:
    """Random latent dataset for training without VAE."""
    def __init__(self, num_samples, latent_size=32, num_classes=1000):
        self.num_samples = num_samples
        self.latent_size = latent_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        latent = ms.Tensor(np.random.randn(4, self.latent_size, self.latent_size).astype(np.float32), dtype=ms.float32)
        label = ms.Tensor([idx % self.num_classes], dtype=ms.int32)
        return latent, label


class DataLoader:
    """Simple data loader."""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        
        latents = []
        labels = []
        for latent, label in batch:
            latents.append(latent)
            labels.append(label)
        
        latents = ops.Stack()(latents)
        labels = ops.Concat()(labels)
        
        self.current_idx += self.batch_size
        return latents, labels


def main(args):
    ms.set_context(
        device_target="Ascend", 
        device_id=args.device_id,
        mode=ms.PYNATIVE_MODE,
    )
    
    ms.set_seed(args.global_seed)
    
    rank = 0
    world_size = 1
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Starting training on {args.model}")
        print(f"Device: Ascend NPU")
    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8"
    
    from mindone.diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
    
    latent_size = args.image_size // 8
    
    model = DiTTransformer2DModel(
        in_channels=4,
        out_channels=8,
        patch_size=2,
        num_attention_heads=16,
        attention_head_dim=72,
        num_layers=28,
        sample_size=latent_size,
        num_embeds_ada_norm=1000,
    )
    model.class_dropout_prob = 0.1
    
    from copy import deepcopy
    ema = deepcopy(model)
    
    print(f"Model parameters: {sum(p.size for p in model.get_parameters())}")
    
    betas = np.linspace(1e-4, 0.02, 1000, dtype=np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod = ms.Tensor(alphas_cumprod, dtype=ms.float32)
    
    from mindspore.nn import Adam
    optimizer = Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=0)
    
    num_samples = args.num_samples or 10000
    dataset = RandomLatentDataset(num_samples, latent_size=latent_size)
    print(f"Dataset contains {num_samples} samples (random latents)")
    
    train_steps = 0
    running_loss = 0
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        dataloader = DataLoader(dataset, batch_size=args.global_batch_size // world_size, shuffle=True)
        
        for batch_idx, (latents, labels) in enumerate(dataloader):
            batch_size = latents.shape[0]
            
            t = mint.randint(0, 1000, (batch_size,), dtype=ms.int64)
            
            noise = mint.randn_like(latents)
            
            sqrt_alpha_prod = ops.Sqrt()(ops.Gather()(alphas_cumprod, t, 0).reshape(-1, 1, 1, 1))
            sqrt_one_minus_alpha_prod = ops.Sqrt()(1 - ops.Gather()(alphas_cumprod, t, 0).reshape(-1, 1, 1, 1))
            noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
            
            class_labels = labels
            if model.class_dropout_prob > 0:
                mask = mint.rand((batch_size,), dtype=ms.float32) < model.class_dropout_prob
                null_class = ms.Tensor([1000] * batch_size, dtype=ms.int32)
                class_labels = mint.where(mask, null_class, labels)
            
            model.set_grad(True)
            model_pred = model(noisy_latents, timestep=t, class_labels=class_labels)[0]
            
            loss = ops.mse_loss(model_pred, noise, reduction='mean')
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            decay = 0.9999
            ema_params = dict(ema.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                if name in ema_params:
                    ema_p = ema_params[name]
                    ema_p.set_value(ops.lerp(param.value(), ema_p.value(), ms.Tensor(decay, dtype=ms.float32)))
            
            loss_val = float(loss.asnumpy())
            running_loss += loss_val
            train_steps += 1
            
            if train_steps % args.log_every == 0 and rank == 0:
                avg_loss = running_loss / args.log_every
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                running_loss = 0
            
            if args.max_steps is not None and train_steps >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping training...")
                if rank == 0:
                    checkpoint_path = f"{args.results_dir}/ckpt_{train_steps:07d}.ckpt"
                    ms.save_checkpoint(ema, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                return
        
        if rank == 0 and (epoch + 1) % args.ckpt_every == 0:
            checkpoint_path = f"{args.results_dir}/ckpt_{epoch+1:04d}.ckpt"
            ms.save_checkpoint(ema, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    if rank == 0:
        print("Training completed!")
        final_ckpt = f"{args.results_dir}/final.ckpt"
        ms.save_checkpoint(ema, final_ckpt)
        print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="ms_train_output", help="Output directory")
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="Model name")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--global-batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--global-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=10, help="Log frequency")
    parser.add_argument("--ckpt-every", type=int, default=1, help="Checkpoint frequency (epochs)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--device-id", type=int, default=0, help="NPU device ID")
    
    args = parser.parse_args()
    main(args)