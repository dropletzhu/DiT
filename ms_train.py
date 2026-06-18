# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Training Script for Ascend NPU
# Aligned with PyTorch DDP training (train.py)

import argparse
import json
import os
import sys
import numpy as np
from glob import glob
from pathlib import Path
from time import time
from copy import deepcopy
from PIL import Image

# Ensure Ascend environment is detectable before importing mindspore
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
from mindspore import mint, nn, ops
from mindspore.communication import init, get_rank, get_group_size, release
from mindspore.ops import clip_by_global_norm
import mindspore.dataset as ds
import mindspore.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

from mindone.diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from mindone.diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"

LATENT_SCALING_FACTOR = 0.18215
NUM_TIMESTEPS = 1000


#################################################################################
#                             Image Preprocessing                               #
#################################################################################

def center_crop_arr(pil_image, image_size):
    """Center cropping from ADM — same as PyTorch train.py."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


_DATASET_CACHE = {}  # (data_dir, num_classes) -> (paths, labels)


def _walk_dataset(data_dir, num_classes=1000):
    """Walk ImageNet train directory once and cache the result."""
    key = (data_dir, num_classes)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    train_dir = Path(data_dir) / "train"
    image_paths = []
    labels = []
    for class_idx, class_dir in enumerate(sorted(train_dir.iterdir())):
        if class_idx >= num_classes:
            break
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append(img_path)
                labels.append(class_idx)
    result = (image_paths, labels)
    _DATASET_CACHE[key] = result
    return result


class ImageNetDataset:
    """ImageNet dataset — preprocessing matches PyTorch train.py."""

    def __init__(self, image_paths, labels, image_size=256):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.num_classes = max(labels) + 1 if labels else 1000

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = center_crop_arr(img, self.image_size)
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)
        return arr.astype(np.float32), self.labels[idx]


def _create_graph_dataset(data_dir, image_size, batch_size, num_workers, world_size, rank):
    """MindSpore-native dataset for graph mode — no PIL dependency."""
    import mindspore.dataset.vision as vision

    transform_list = [
        vision.Decode(),
        vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
        vision.CenterCrop(image_size),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        vision.HWC2CHW(),
    ]
    dataset = ds.ImageFolderDataset(
        str(Path(data_dir) / "train"),
        num_shards=world_size,
        shard_id=rank,
        shuffle=True,
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(operations=transform_list, num_parallel_workers=num_workers)
    dataset = dataset.project(columns=["image", "label"])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


#################################################################################
#                             EMA Update                                       #
#################################################################################

def update_ema(ema_model, model, decay=0.9999):
    """Step EMA model toward current model — same as PyTorch train.py."""
    ema_params = {p.name: p for p in ema_model.get_parameters()}
    model_params = {p.name: p for p in model.get_parameters()}
    for name, param in model_params.items():
        if name in ema_params:
            ema_p = ema_params[name]
            ema_p.set_data(
                ops.lerp(param.value(), ema_p.value(), ms.Tensor(decay, dtype=ms.float32))
            )


def requires_grad(model, flag=True):
    for p in model.get_parameters():
        p.requires_grad = flag


#################################################################################
#                     Graph-Mode Forward + Loss Cell                            #
#################################################################################

class DiTLossCell(nn.Cell):
    """
    DIterForward + MSE loss, compiled for ms.GRAPH_MODE.
    The outer loop calls grad_fn(latents, t, noise, class_labels).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def construct(self, noisy_latents, t, noise, class_labels):
        pred = self.model(noisy_latents, timestep=t, class_labels=class_labels)[0]
        pred = pred[:, :4, :, :]
        return ops.mse_loss(pred, noise, reduction="mean")


#################################################################################
#                             Training Loop                                     #
#################################################################################

def main(rank, args):
    # --------------------------------------------------------------------------
    #  MindSpore context (init() must come before any tensor / parameter
    #  creation for HCCL to work correctly)
    # --------------------------------------------------------------------------
    exec_mode = ms.GRAPH_MODE if args.exec_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=exec_mode, device_target="Ascend")
    if exec_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O1'})
        ms.set_context(compile_cache_path='/tmp/ms_compile_cache')

    # --------------------------------------------------------------------------
    #  Distributed init
    # --------------------------------------------------------------------------
    if "RANK_ID" in os.environ:
        init()
        rank = get_rank()
        world_size = get_group_size()
    else:
        world_size = 1

    assert args.global_batch_size % world_size == 0
    batch_size = args.global_batch_size // world_size
    seed = args.global_seed * world_size + rank
    ms.set_seed(seed)
    np.random.seed(seed)

    print(f"[Rank {rank}/{world_size}] Starting training. seed={seed}, batch_size={batch_size}")

    # --------------------------------------------------------------------------
    #  Experiment directory / logger
    # --------------------------------------------------------------------------
    experiment_dir = ""
    checkpoint_dir = ""
    log_handle = None

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_handle = open(f"{experiment_dir}/log.txt", "a", buffering=1)
        print(f"Experiment directory: {experiment_dir}")

    def log(msg):
        if log_handle is not None:
            print(msg)
            log_handle.write(f"[{time():.0f}] {msg}\n")

    # --------------------------------------------------------------------------
    #  Build model, EMA, VAE (AFTER init(), each on its own NPU)
    # --------------------------------------------------------------------------
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8

    model_configs = {
        "DiT-XL/2": dict(num_attention_heads=16, attention_head_dim=72, num_layers=28, patch_size=2),
        "DiT-L/2":  dict(num_attention_heads=16, attention_head_dim=48, num_layers=24, patch_size=2),
        "DiT-B/2":  dict(num_attention_heads=12, attention_head_dim=48, num_layers=12, patch_size=2),
    }
    cfg = model_configs[args.model]
    model = DiTTransformer2DModel(
        in_channels=4,
        out_channels=8,
        sample_size=latent_size,
        num_embeds_ada_norm=args.num_classes,
        **cfg,
    )
    ema = deepcopy(model)
    requires_grad(ema, False)

    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.set_train(False)
    for p in vae.get_parameters():
        p.requires_grad = False

    n_params = sum(p.size for p in model.get_parameters())
    log(f"Model parameters: {n_params:,}")

    # --------------------------------------------------------------------------
    #  Diffusion constants (linear schedule, same as PyTorch)
    # --------------------------------------------------------------------------
    betas = np.linspace(1e-4, 0.02, NUM_TIMESTEPS, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = ms.Tensor(np.sqrt(alphas_cumprod), dtype=ms.float32)
    sqrt_one_minus_alphas_cumprod = ms.Tensor(np.sqrt(1.0 - alphas_cumprod), dtype=ms.float32)

    gather_op = ops.Gather()

    # --------------------------------------------------------------------------
    #  Optimizer (same hyperparams as PyTorch: AdamW, lr=1e-4, weight_decay=0)
    # --------------------------------------------------------------------------
    amp_enabled = args.amp and not args.no_amp
    if amp_enabled and args.amp_level != "O0":
        from mindspore.train.amp import auto_mixed_precision
        model = auto_mixed_precision(
            model, amp_level=args.amp_level,
            dtype=ms.bfloat16 if args.dtype == "bf16" else ms.float16,
        )

    optimizer = nn.AdamWeightDecay(
        model.trainable_params(),
        learning_rate=args.lr,
        weight_decay=0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )

    # --------------------------------------------------------------------------
    #  Checkpoint resume
    # --------------------------------------------------------------------------
    train_steps = 0
    if args.ckpt:
        log(f"Loading checkpoint from {args.ckpt}")
        ckpt = ms.load_checkpoint(args.ckpt)
        model_keys = {k.replace("model.", ""): v for k, v in ckpt.items() if k.startswith("model.")}
        ema_keys = {k.replace("ema.", ""): v for k, v in ckpt.items() if k.startswith("ema.")}
        ms.load_param_into_net(model, model_keys)
        ms.load_param_into_net(ema, ema_keys)
        meta_path = str(Path(args.ckpt).with_suffix("")) + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            train_steps = meta.get("train_steps", 0)
            log(f"Resumed from step {train_steps}")

    # --------------------------------------------------------------------------
    #  Preparation for training
    # --------------------------------------------------------------------------
    if not args.ckpt:
        update_ema(ema, model, decay=0)

    model.set_train(True)
    ema.set_train(False)

    log_steps = 0
    running_loss = 0.0
    start_time = time()
    compiled = False

    log(f"Training for {args.epochs} epochs, max_steps={args.max_steps}")
    log(f"AMP: {amp_enabled}, dtype: {args.dtype}, grad_clip: {args.grad_clip}")

    # --------------------------------------------------------------------------
    #  Training loop
    # --------------------------------------------------------------------------
    # Walk the dataset once (cached across epochs)
    image_paths, labels_list = _walk_dataset(args.data_path, args.num_classes)
    log(f"Dataset: {len(image_paths)} images")

    # Build graph-mode loss cell and grad fn once
    loss_cell = DiTLossCell(model)
    grad_fn = ms.value_and_grad(loss_cell, None, model.trainable_params())

    for epoch in range(args.epochs):
        if args.exec_mode == "graph":
            dataset = _create_graph_dataset(
                args.data_path, args.image_size, batch_size,
                args.num_workers, world_size, rank,
            )
            dataset_iterator = dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
        else:
            raw_dataset = ImageNetDataset(image_paths, labels_list, args.image_size)
            dataset = ds.GeneratorDataset(
                source=raw_dataset,
                column_names=["image", "label"],
                num_shards=world_size,
                shard_id=rank,
                shuffle=True,
                num_parallel_workers=args.num_workers,
                python_multiprocessing=True,
            )
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset_iterator = dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)

        for numpy_images, numpy_labels in dataset_iterator:
            images = ms.Tensor(numpy_images, dtype=ms.float32)
            labels = ms.Tensor(numpy_labels, dtype=ms.int32).reshape(-1)

            # ---- VAE encode (no grad, pynative) ----
            h = vae.encode(images, return_dict=False)[0]
            mean = h[:, :4, :, :]
            logvar = h[:, 4:, :, :]
            std = ops.exp(0.5 * logvar)
            noise_latent = mint.randn_like(mean)
            latents = mean + std * noise_latent
            latents = ops.stop_gradient(latents * LATENT_SCALING_FACTOR)

            # ---- Diffusion noise + schedule (pynative) ----
            noise = mint.randn_like(latents)
            bsz = latents.shape[0]
            t = mint.randint(0, NUM_TIMESTEPS, (bsz,), dtype=ms.int64)

            sqrt_ac = gather_op(sqrt_alphas_cumprod, t, 0).reshape(-1, 1, 1, 1)
            sqrt_one_minus_ac = gather_op(sqrt_one_minus_alphas_cumprod, t, 0).reshape(-1, 1, 1, 1)
            noisy_latents = sqrt_ac * latents + sqrt_one_minus_ac * noise

            # ---- Class-label dropout (CFG, pynative) ----
            class_labels = labels
            if args.class_dropout_prob > 0:
                drop_mask = mint.rand((bsz,), dtype=ms.float32) < args.class_dropout_prob
                null_label = ops.fill(ms.int32, (bsz,), args.num_classes)
                class_labels = mint.where(drop_mask, null_label, labels)

            # ---- Forward + backward (compiled in graph-mode) ----
            loss, grads = grad_fn(noisy_latents, t, noise, class_labels)

            # ---- Gradient all-reduce (distributed) ----
            if world_size > 1:
                for g in grads:
                    ops.AllReduce(ops.ReduceOp.SUM)(g)
                    g /= world_size

            if args.grad_clip > 0:
                grads = clip_by_global_norm(grads, clip_norm=args.grad_clip)

            optimizer(grads)

            # ---- EMA update ----
            update_ema(ema, model)

            # ---- Logging ----
            loss_val = float(loss.asnumpy())
            running_loss += loss_val
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                end_time = time()
                elapsed = end_time - start_time
                avg_loss = running_loss / log_steps if log_steps > 0 else 0.0

                if not compiled:
                    comp_time = elapsed
                    compiled = True
                    # Reset timer for next interval to measure true training speed
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()
                    log(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                        f"Compilation: {comp_time:.1f}s, Epoch: {epoch}")
                else:
                    steps_per_sec = log_steps / elapsed if elapsed > 0 else 0.0
                    if world_size > 1:
                        loss_t = ms.Tensor([avg_loss], dtype=ms.float32)
                        ops.AllReduce(ops.ReduceOp.SUM)(loss_t)
                        avg_loss = float(loss_t.asnumpy()[0]) / world_size

                    log(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                        f"Steps/Sec: {steps_per_sec:.2f}, Epoch: {epoch}")
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

            # ---- Checkpoint save ----
            if train_steps % args.ckpt_every == 0 and train_steps > 0 and rank == 0:
                ckpt_list = []
                for p in model.get_parameters():
                    ckpt_list.append({"name": "model." + p.name, "data": p.data})
                for p in ema.get_parameters():
                    ckpt_list.append({"name": "ema." + p.name, "data": p.data})
                ckpt_path = str(Path(checkpoint_dir) / f"{train_steps:07d}.ckpt")
                ms.save_checkpoint(ckpt_list, ckpt_path)
                meta = {"train_steps": train_steps, "args": vars(args)}
                with open(str(Path(ckpt_path).with_suffix("")) + "_meta.json", "w") as f:
                    json.dump(meta, f)
                log(f"Saved checkpoint -> {ckpt_path}")

            # ---- Early stopping ----
            if args.max_steps is not None and train_steps >= args.max_steps:
                log(f"Reached max_steps={args.max_steps}, stopping")
                if rank == 0:
                    ckpt_list = []
                    for p in model.get_parameters():
                        ckpt_list.append({"name": "model." + p.name, "data": p.data})
                    for p in ema.get_parameters():
                        ckpt_list.append({"name": "ema." + p.name, "data": p.data})
                    final_path = str(Path(checkpoint_dir) / f"final_{train_steps:07d}.ckpt")
                    ms.save_checkpoint(ckpt_list, final_path)
                    meta = {"train_steps": train_steps, "args": vars(args)}
                    with open(str(Path(final_path).with_suffix("")) + "_meta.json", "w") as f:
                        json.dump(meta, f)
                    log(f"Saved final checkpoint -> {final_path}")
                if log_handle is not None:
                    log_handle.close()
                return

    # --------------------------------------------------------------------------
    #  Done
    # --------------------------------------------------------------------------
    log("Training completed!")
    if rank == 0:
        ckpt_list = []
        for p in model.get_parameters():
            ckpt_list.append({"name": "model." + p.name, "data": p.data})
        for p in ema.get_parameters():
            ckpt_list.append({"name": "ema." + p.name, "data": p.data})
        final_path = str(Path(checkpoint_dir) / "final.ckpt")
        ms.save_checkpoint(ckpt_list, final_path)
        meta = {"train_steps": train_steps, "args": vars(args)}
        with open(str(Path(final_path).with_suffix("")) + "_meta.json", "w") as f:
            json.dump(meta, f)
        log(f"Saved final checkpoint -> {final_path}")

    if log_handle is not None:
        log_handle.close()
    if world_size > 1:
        release()


#################################################################################
#                             Argument Parsing                                  #
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=["DiT-XL/2", "DiT-L/2", "DiT-B/2"])
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae-path", type=str, default="/home/ma-user/work/temp/sd-vae-ft-mse")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--amp-level", type=str, default="O2", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--exec-mode", type=str, default="graph", choices=["pynative", "graph"])
    parser.add_argument("--nproc-per-node", type=int, default=1)

    args = parser.parse_args()

    if args.nproc_per_node > 1:
        if args.nproc_per_node > args.num_workers:
            args.num_workers = args.nproc_per_node
        mp.spawn(main, args=(args,), nprocs=args.nproc_per_node)
    else:
        main(0, args)
