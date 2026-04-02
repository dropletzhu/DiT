# Convert PyTorch DiT checkpoint to MindSpore DiT format

import os
import re
import torch
import mindspore as ms
from mindspore.common.parameter import Parameter


def convert_pytorch_to_mindspore(torch_ckpt_path: str, ms_ckpt_path: str):
    """Convert PyTorch DiT checkpoint to MindSpore DiT format."""
    print(f"Loading PyTorch checkpoint from {torch_ckpt_path}")
    checkpoint = torch.load(torch_ckpt_path, map_location="cpu")
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    else:
        state_dict = checkpoint
    
    print(f"Loaded {len(state_dict)} keys from PyTorch checkpoint")
    
    ms_state_dict = {}
    
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        
        v_np = v.numpy()
        new_key = k
        
        # Skip pos_embed - MindSpore generates its own
        if k == "pos_embed":
            print(f"  Skipped: {k}")
            continue
        
        # Replace "attn.qkv." with just "qkv."
        if "attn.qkv." in k:
            new_key = k.replace("attn.qkv.", "qkv.")
        
        # Replace "attn.proj." with just "proj."
        elif "attn.proj." in k:
            new_key = k.replace("attn.proj.", "proj.")
        
        # Replace "mlp.fc1." with "mlp.0." (first linear layer in MLP)
        elif "mlp.fc1." in k:
            new_key = k.replace("mlp.fc1.", "mlp.0.")
        
        # Replace "mlp.fc2." with "mlp.2." (second linear layer in MLP)
        elif "mlp.fc2." in k:
            new_key = k.replace("mlp.fc2.", "mlp.2.")
        
        # Replace "y_embedder.embedding_table.weight" -> "y_embedder.embedding_table.embedding_table"
        elif k == "y_embedder.embedding_table.weight":
            new_key = "y_embedder.embedding_table.embedding_table"
        
        # Skip norm layers (PyTorch uses elementwise_affine=False)
        elif ".norm1." in k or ".norm2." in k:
            print(f"  Skipped: {k}")
            continue
        
        # Skip final_layer.norm_final
        elif "final_layer.norm_final." in k:
            print(f"  Skipped: {k}")
            continue
        
        print(f"  Mapped: {k} -> {new_key}")
        ms_state_dict[new_key] = ms.Tensor(v_np, dtype=ms.float32)
    
    print(f"\nConverted {len(ms_state_dict)} parameters")
    
    # Create parameter dict for saving
    param_dict = {}
    for k, v in ms_state_dict.items():
        param_dict[k] = Parameter(v, name=k)
    
    os.makedirs(os.path.dirname(ms_ckpt_path) or ".", exist_ok=True)
    ms.save_checkpoint(param_dict, ms_ckpt_path)
    print(f"Saved MindSpore checkpoint to {ms_ckpt_path}")
    
    return ms_state_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PyTorch DiT to MindSpore DiT")
    parser.add_argument("--input", type=str, required=True, help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="MindSpore checkpoint output path")
    args = parser.parse_args()
    
    convert_pytorch_to_mindspore(args.input, args.output)
