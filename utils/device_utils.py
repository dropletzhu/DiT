# device_utils.py - Device abstraction for GPU and NPU support
import torch
import argparse
from typing import Optional

_device_override = None

def set_device_override(device_type):
    """Override device detection (for CLI args)"""
    global _device_override
    _device_override = device_type

def get_device_type():
    """Detect and return device type: 'npu', 'cuda', 'cpu'"""
    global _device_override
    if _device_override is not None:
        return _device_override
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device():
    """Return torch.device object"""
    return torch.device(get_device_type())


def get_device_str():
    """Return device string for tensor creation"""
    return get_device_type()


def get_device_count():
    """Return number of available devices"""
    dev_type = get_device_type()
    if dev_type == 'npu':
        return torch.npu.device_count()
    elif dev_type == 'cuda':
        return torch.cuda.device_count()
    return 1


def set_device(device_id: int):
    """Set current device"""
    dev_type = get_device_type()
    if dev_type == 'npu':
        torch.npu.set_device(device_id)
    elif dev_type == 'cuda':
        torch.cuda.set_device(device_id)


def synchronize():
    """Synchronize device"""
    dev_type = get_device_type()
    if dev_type == 'npu':
        torch.npu.synchronize()
    elif dev_type == 'cuda':
        torch.cuda.synchronize()


def enable_tf32(enabled: bool = True):
    """Enable TF32 precision (only effective for CUDA, ignored for NPU)"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def get_distributed_backend():
    """Return distributed backend: 'hccl' (NPU) / 'nccl' (CUDA) / 'gloo' (CPU)"""
    device_type = get_device_type()
    if device_type == 'npu':
        return 'hccl'
    elif device_type == 'cuda':
        return 'nccl'
    else:
        return 'gloo'


def get_autocast(enabled: bool = True, dtype=torch.float16):
    """Return autocast context manager"""
    dev_type = get_device_type()
    if dev_type in ('npu', 'cuda') and enabled:
        return torch.amp.autocast(device_type=dev_type, dtype=dtype)
    return torch.amp.autocast(device_type='cpu', dtype=torch.float32, enabled=False)


def get_amp_scaler(enabled: bool = True):
    """Return GradScaler (only for NPU/CUDA)"""
    dev_type = get_device_type()
    if dev_type in ('npu', 'cuda') and enabled:
        return torch.amp.GradScaler(dev_type)
    return None


def is_npu():
    """Check if device is NPU"""
    return get_device_type() == 'npu'


def is_cuda():
    """Check if device is CUDA"""
    return get_device_type() == 'cuda'


def is_available():
    """Check if any accelerator is available"""
    return get_device_type() != 'cpu'


def add_device_args(parser: argparse.ArgumentParser):
    """Add device-related arguments to argparse parser"""
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "npu", "cuda", "cpu"],
        help="Device to use: auto (detect), npu, cuda, or cpu (default: auto)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision (default: True)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="AMP dtype: fp16 or bf16 (default: fp16)"
    )
    return parser
