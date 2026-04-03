"""
Convert PyTorch DiT checkpoint to MindSpore format with correct parameter name mapping.
"""
import torch
import mindspore as ms
import numpy as np
from collections import OrderedDict

def convert_checkpoint(pt_path, ms_path):
    # Load PyTorch checkpoint
    pt_ckpt = torch.load(pt_path, map_location='cpu')
    print(f"Loaded PyTorch checkpoint from {pt_path}")
    
    # Create MindSpore parameter dict
    ms_params = OrderedDict()
    
    # Mapping from PyTorch names to MindSpore names
    param_mapping = {
        'y_embedder.embedding_table.weight': 'y_embedder.embedding_table.embedding_table',
    }
    
    for pt_name, pt_param in pt_ckpt.items():
        # Apply parameter name mapping
        ms_name = param_mapping.get(pt_name, pt_name)
        
        # Handle different naming conventions
        # PyTorch: blocks.X.attn.qkv.weight -> MindSpore: blocks.X.qkv.weight
        # PyTorch: blocks.X.mlp.fc1.weight -> MindSpore: blocks.X.mlp.0.weight
        # PyTorch: blocks.X.mlp.fc2.weight -> MindSpore: blocks.X.mlp.2.weight
        ms_name = ms_name.replace('.attn.', '.')
        ms_name = ms_name.replace('.mlp.fc1.', '.mlp.0.')
        ms_name = ms_name.replace('.mlp.fc2.', '.mlp.2.')
        
        # Convert to MindSpore tensor
        ms_params[ms_name] = ms.Parameter(ms.Tensor(pt_param.numpy(), dtype=ms.float32), requires_grad=False)
    
    # Save MindSpore checkpoint
    ms.save_checkpoint(ms_params, ms_path)
    print(f"Saved MindSpore checkpoint to {ms_path}")
    print(f"Total parameters: {len(ms_params)}")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/ma-user/work/xql/dongfeng/DiT')
    convert_checkpoint(
        '/home/ma-user/work/xql/dongfeng/DiT/pretrained_models/DiT-XL-2-256x256.pt',
        '/home/ma-user/work/xql/dongfeng/DiT/ms_checkpoints/DiT-XL-2-256x256_converted.ckpt'
    )
