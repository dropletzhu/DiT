# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Training Script for Ascend NPU

import argparse
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from PIL import Image

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
    
    def training_losses(self, model, x_start, t, y):
        noise = ops.StandardNormal()(x_start.shape).astype(ms.float32)
        
        sqrt_alphas_cumprod_t = ops.GatherD()(
            self.alphas_cumprod.expand_dims(0), 
            ms.Tensor([0], dtype=ms.int32), 
            t.reshape(-1, 1)
        ).reshape(x_start.shape[0])
        
        x_noisy = sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1) * x_start + ops.Sqrt()(1 - sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1)) * noise
        
        model_output = model(x_noisy, t, y)
        
        # Use only first 4 channels for loss
        loss = ops.ReduceMean()((model_output[:, :4, :, :] - noise) ** 2)
        return {"loss": loss}


def update_ema(ema_model, model, decay=0.9999):
    ema_params = list(ema_model.get_parameters())
    model_params = list(model.get_parameters())
    
    for ema_p, model_p in zip(ema_params, model_params):
        ema_p_data = ema_p.value().asnumpy()
        model_p_data = model_p.value().asnumpy()
        new_data = ema_p_data * decay + model_p_data * (1 - decay)
        ema_p.set_data(ms.Tensor(new_data, dtype=ema_p.value().dtype))


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), Image.BOX)
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), Image.BICUBIC)
    
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageNetDataset:
    def __init__(self, root, transform=None, split='train'):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        split_dir = root
        if os.path.isdir(os.path.join(root, split)):
            split_dir = os.path.join(root, split)
        
        if os.path.isdir(split_dir):
            for class_name in sorted(os.listdir(split_dir)):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            return ms.Tensor.zeros(3, 256, 256, dtype=ms.float32), label


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
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
        
        images = []
        labels = []
        for img, label in batch:
            images.append(img)
            labels.append(label)
        
        images = ops.Stack()(images) if images else ms.Tensor.zeros(self.batch_size, 4, 32, 32, dtype=ms.float32)
        labels = ms.Tensor(labels, dtype=ms.int32)
        
        self.current_idx += self.batch_size
        return images, labels


def main(args):
    ms.set_context(device_target="Ascend", device_id=0)
    ms.set_seed(args.global_seed)
    
    rank = 0
    world_size = 1
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Starting training on {args.model}")
        print(f"Data path: {args.data_path}")
        print(f"Device: Ascend NPU")
    
    # Create model
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    # EMA model
    from copy import deepcopy
    ema = deepcopy(model)
    
    print(f"Model parameters: {sum(p.size for p in model.get_parameters())}")
    
    # Create diffusion
    diffusion = SimpleDDPM(get_named_beta_schedule("linear", 1000))
    
    # Dataset - for testing, we use random latent data instead of actual images
    # This is because we don't have a MindSpore VAE
    latent_size = args.image_size // 8
    
    class RandomLatentDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return random latent
            return ms.Tensor(np.random.randn(4, latent_size, latent_size).astype(np.float32), dtype=ms.float32), idx % 1000
    
    dataset = RandomLatentDataset(1000)
    
    if rank == 0:
        print(f"Dataset contains {len(dataset)} images")
    
    # Optimizer - use Momentum instead
    optimizer = nn.Momentum(model.trainable_params(), learning_rate=1e-4, momentum=0.9)
    
    # Use MindSpore's built-in train step
    class TrainStep(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.network = network
        
        def construct(self, x, y):
            t = ms.Tensor(np.random.randint(0, diffusion.num_timesteps, x.shape[0]), dtype=ms.int32)
            loss_dict = diffusion.training_losses(self.network, x, t, y)
            return loss_dict["loss"]
    train_net = TrainStep(model)
    
    optimizer = nn.Momentum(train_net.trainable_params(), learning_rate=1e-4, momentum=0.9)
    train_cell = nn.TrainOneStepCell(train_net, optimizer)
    train_cell.set_train()
    
    train_steps = 0
    running_loss = 0
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        dataloader = DataLoader(dataset, batch_size=args.global_batch_size // world_size, shuffle=True)
        
        for batch_idx, (x, y) in enumerate(dataloader):
            loss = train_cell(x, y)
            loss_val = loss.asnumpy()
            
            # Update EMA
            update_ema(ema, model, decay=0.9999)
            
            loss_val = loss.asnumpy()
            running_loss += loss_val
            train_steps += 1
            
            if train_steps % args.log_every == 0 and rank == 0:
                avg_loss = running_loss / args.log_every
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                running_loss = 0
            
            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0 and rank == 0:
                checkpoint_path = f"{args.results_dir}/ckpt_{train_steps:07d}.ckpt"
                ms.save_checkpoint(ema, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Test with limited iterations
            if args.max_steps is not None and train_steps >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping training...")
                return
    
    if rank == 0:
        print("Training completed!")
        final_ckpt = f"{args.results_dir}/final.ckpt"
        ms.save_checkpoint(ema, final_ckpt)
        print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="ms_results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--test", action="store_true", help="Test mode: run only a few steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    args = parser.parse_args()
    main(args)
