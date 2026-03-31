import torch
import os
import sys
import numpy as np

print("Starting PyTorch to MindSpore model conversion...")

PYTORCH_VAE_PATH = "/home/ma-user/work/temp/sd-vae-ft-mse"
OUTPUT_DIR = "/home/ma-user/work/xql/dongfeng/DiT/mindspore"

from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained(PYTORCH_VAE_PATH)
vae.eval()

print("Converting VAE decoder to ONNX...")

class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, latent):
        return self.vae.decode(latent).sample

decoder = VAEDecoder(vae)

latent = torch.randn(1, 4, 32, 32)

torch.onnx.export(
    decoder,
    (latent,),
    os.path.join(OUTPUT_DIR, "vae_decoder.onnx"),
    input_names=['latent'],
    output_names=['image'],
    dynamic_axes={'latent': {0: 'batch'}, 'image': {0: 'batch'}},
    opset_version=17
)

print(f"VAE decoder saved to {OUTPUT_DIR}/vae_decoder.onnx")

print("\nModel conversion completed!")
print(f"Files saved to: {OUTPUT_DIR}")
