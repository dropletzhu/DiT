# AGENTS.md - DiT Project Guidelines

## Project Overview

This is the DiT (Scalable Diffusion Models with Transformers) repository from Meta Research. It contains PyTorch implementations of diffusion transformer models for image generation.

## Language

Python 3.8+, PyTorch 2.0+

## Build/Run Commands

### Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate DiT

# Or pip
pip install -r requirements.txt
```

### Running Scripts

```bash
# Single GPU sampling
python sample.py --image-size 512 --seed 1

# Training (single node, multi-GPU)
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-XL/2 --data-path /path/to/imagenet/train

# Multi-GPU sampling for FID evaluation
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

### Testing

This project has **no formal test suite**. To verify changes:
1. Run a quick sampling test: `python sample.py --model DiT-XL/2 --seed 42`
2. For training changes, run a few training iterations with a small batch

### Linting

No formal linter is configured. Follow the code style below.

## Code Style Guidelines

### Imports

```python
# Standard library imports
import os
import argparse
import logging
from time import time

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from timm.models.vision_transformer import PatchEmbed, Attention

# Local imports
from models import DiT_models
from diffusion import create_diffusion
from utils import enable_tf32, get_device_str
```

- Group: stdlib, third-party, local (blank line between groups)
- Sort alphabetically within groups

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `update_ema`, `train_steps`)
- **Classes**: `PascalCase` (e.g., `DiTBlock`, `TimestepEmbedder`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_IMAGE_SIZE`)
- **Private methods**: prefix with underscore (e.g., `_basic_init`)

### Type Annotations

Use type hints for function signatures when the types aren't obvious:
```python
def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
```

### Documentation

```python
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def forward(self, x, c):
        """
        Forward pass of DiT block.
        x: (N, T, D) tensor of spatial inputs
        c: (N, D) tensor of conditioning
        """
```

### Error Handling

- Use assertions for parameter validation and invariants
- Raise descriptive exceptions for recoverable errors

### Neural Network Modules

Inherit from `nn.Module`:
```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

### Device Management

Use the utils abstraction layer for device support (GPU/NPU/CPU):
```python
from utils import get_device_str, enable_tf32, get_autocast

device = get_device_str()
enable_tf32(True)  # Enable TF32 on Ampere GPUs for speed
```

### Performance Tips

- Use `@torch.no_grad()` decorator for inference
- Use `torch.no_grad():` context for inference blocks
- Enable TF32 for training/sampling on Ampere GPUs (A100, etc.)
- Use mixed precision (AMP) via `get_autocast` and `get_amp_scaler` from utils

### Common Patterns

**EMA (Exponential Moving Average):**
```python
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

**Distributed Training:**
```python
import torch.distributed as dist
dist.init_process_group(backend)
# Use torchrun for launching: torchrun --nnodes=1 --nproc_per_node=N script.py
```

### File Structure

```
DiT/
├── models.py          # DiT model definitions
├── train.py           # Training script (DDP)
├── sample.py          # Sampling from pretrained models
├── sample_ddp.py      # Distributed sampling for FID eval
├── download.py        # Model checkpoint downloading
├── requirements.txt  # Python dependencies
├── environment.yml    # Conda environment
├── utils/             # Device abstraction utilities
│   └── device_utils.py
├── diffusion/         # Diffusion utilities
│   ├── gaussian_diffusion.py
│   ├── respace.py
│   └── timestep_sampler.py
└── visuals/           # Sample images
```

### Common Arguments

Training: `--data-path`, `--model`, `--image-size`, `--epochs`, `--global-batch-size`, `--amp`, `--no-amp`, `--dtype`, `--vae-path`
Sampling: `--model`, `--image-size`, `--cfg-scale`, `--num-sampling-steps`, `--seed`, `--ckpt`, `--vae-path`