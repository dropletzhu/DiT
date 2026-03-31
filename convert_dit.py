import torch
import os
import mindspore as ms
from models import DiT_models

OUTPUT_DIR = "/home/ma-user/work/xql/dongfeng/DiT/mindspore"
PT_DIT_PATH = "/home/ma-user/work/xql/dongfeng/DiT/pretrained_models/DiT-XL-2-256x256.pt"

print("Loading DiT-XL-2 pretrained weights...")

checkpoint = torch.load(PT_DIT_PATH, map_location="cpu")

if "model" in checkpoint:
    state_dict = checkpoint["model"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "ema" in checkpoint:
    state_dict = checkpoint["ema"]
else:
    state_dict = checkpoint

print(f"Loaded {len(state_dict)} parameters from PyTorch checkpoint")

def convert_key(pt_key):
    """Convert PyTorch key to MindSpore key"""
    ms_key = pt_key
    
    # y_embedder.embedding_table.weight -> y_embedder.embedding_table.embedding_table
    if 'y_embedder.embedding_table.weight' in ms_key:
        ms_key = 'y_embedder.embedding_table.embedding_table'
    
    # blocks.X.attn.qkv.weight -> blocks.X.qkv.weight
    if '.attn.qkv.' in ms_key:
        ms_key = ms_key.replace('.attn.qkv.', '.qkv.')
    # blocks.X.attn.proj.weight -> blocks.X.proj.weight
    elif '.attn.proj.' in ms_key:
        ms_key = ms_key.replace('.attn.proj.', '.proj.')
    # blocks.X.mlp.fc1.weight -> blocks.X.mlp.0.weight
    if '.mlp.fc1.' in ms_key:
        ms_key = ms_key.replace('.mlp.fc1.', '.mlp.0.')
    # blocks.X.mlp.fc2.weight -> blocks.X.mlp.2.weight
    if '.mlp.fc2.' in ms_key:
        ms_key = ms_key.replace('.mlp.fc2.', '.mlp.2.')
    
    # Note: LayerNorm in PyTorch DiT uses elementwise_affine=False, so no weight/bias
    # Only adaLN_modulation layers have learnable parameters
    
    return ms_key

ms_state_dict = {}
matched = 0

for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        new_k = convert_key(k)
        ms_state_dict[new_k] = ms.Tensor(v.numpy(), dtype=ms.float32)
        matched += 1

print(f"Converted {matched} parameters")

param_list = []
for name, value in ms_state_dict.items():
    param_list.append({"name": name, "data": value})

output_ckpt = os.path.join(OUTPUT_DIR, "dit_xl_2.ckpt")

ms.save_checkpoint(param_list, output_ckpt)

print(f"Saved {len(param_list)} parameters to {output_ckpt}")

print("\nDiT conversion completed!")
