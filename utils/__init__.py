# utils/__init__.py - Device abstraction utilities for GPU and NPU support

import torch


def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_str():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_type(device):
    return str(device).split(":")[0] if ":" in str(device) else str(device)


def get_device_count():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.device_count()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def set_device(device_id):
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device_id)
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_id)


def set_device_override(device_str):
    pass


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def enable_tf32(enabled=True):
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = enabled


def get_distributed_backend():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "hccl"
    return "nccl"


def get_autocast(enabled, dtype):
    if enabled:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.amp.autocast(enabled=enabled, dtype=dtype)
        elif torch.cuda.is_available():
            return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    return None


def get_amp_scaler(device=None):
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.amp.GradScaler()
    elif torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda():
    return torch.cuda.is_available()


def is_available():
    return torch.cuda.is_available() or (hasattr(torch, "npu") and torch.npu.is_available())


def add_device_args(parser):
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "npu", "cpu"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
