# utils/__init__.py - Device abstraction utilities for GPU and NPU support

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def get_device_str():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_device_type(device):
    return str(device).split(":")[0] if ":" in str(device) else str(device)


def get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.device_count()
    return 1


def set_device(device_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device_id)


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
    return "nccl"


def get_autocast(enabled, dtype):
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype) if enabled else None


def get_amp_scaler():
    return torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda():
    return torch.cuda.is_available()


def is_available():
    return torch.cuda.is_available() or (hasattr(torch, "npu") and torch.npu.is_available())


def add_device_args(parser):
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
