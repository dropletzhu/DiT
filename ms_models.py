# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# MindSpore DiT Model Implementation for Ascend NPU

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Cell):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell([
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        ])
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
        freqs = ms.Tensor(freqs, dtype=ms.float32)
        args = t[:, None].astype(ms.float32) * freqs[None]
        embedding = ops.Concat(-1)([ops.Cos()(args), ops.Sin()(args)])
        if dim % 2:
            embedding = ops.Concat(-1)([embedding, ops.ZerosLike()(embedding[:, :1])])
        return embedding

    def construct(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Cell):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def construct(self, labels, training):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Cell):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1 = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        self.qkv = nn.Dense(hidden_size, 3 * hidden_size, has_bias=True)
        self.proj = nn.Dense(hidden_size, hidden_size, has_bias=True)
        self.norm2 = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        # Initialize LayerNorm gamma to ones, beta to zeros
        self.norm1.gamma.set_data(ms.Tensor(np.ones(hidden_size, dtype=np.float32)))
        self.norm1.beta.set_data(ms.Tensor(np.zeros(hidden_size, dtype=np.float32)))
        self.norm2.gamma.set_data(ms.Tensor(np.ones(hidden_size, dtype=np.float32)))
        self.norm2.beta.set_data(ms.Tensor(np.zeros(hidden_size, dtype=np.float32)))
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.SequentialCell([
            nn.Dense(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dense(mlp_hidden_dim, hidden_size),
        ])
        self.adaLN_modulation = nn.SequentialCell([
            nn.SiLU(),
            nn.Dense(hidden_size, 6 * hidden_size, has_bias=True)
        ])

    def construct(self, x, c):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.Split(-1, 6)(mod)
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm)
        q, k, v = ops.Split(-1, 3)(qkv)
        
        B, N, _ = q.shape
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = ops.Sqrt()(ms.Tensor([self.head_dim], dtype=ms.float32))
        attn = ops.BatchMatMul()(q, k.transpose(0, 1, 3, 2)) / scale
        attn = ops.Softmax(-1)(attn)
        attn = ops.BatchMatMul()(attn, v)
        
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, self.hidden_size)
        attn = self.proj(attn)
        
        x = x + gate_msa.unsqueeze(1) * attn
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Cell):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm((hidden_size,), epsilon=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell([
            nn.SiLU(),
            nn.Dense(hidden_size, 2 * hidden_size, has_bias=True)
        ])
        # Initialize LayerNorm gamma to ones, beta to zeros
        self.norm_final.gamma.set_data(ms.Tensor(np.ones(hidden_size, dtype=np.float32)))
        self.norm_final.beta.set_data(ms.Tensor(np.zeros(hidden_size, dtype=np.float32)))

    def construct(self, x, c):
        shift, scale = ops.Split(-1, 2)(self.adaLN_modulation(c))
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Cell):
    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=1152, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)

    def construct(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.reshape(B, self.num_patches, -1)
        return x


class DiT(nn.Cell):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        
        pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches ** 0.5))
        self.pos_embed = ms.Parameter(ms.Tensor(pos_embed, dtype=ms.float32).unsqueeze(0), requires_grad=False)

        self.blocks = nn.CellList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.numpy.rand(cell.weight.shape).astype(np.float32) * 0.02)
                if cell.bias is not None:
                    cell.bias.set_data(ms.numpy.zeros(cell.bias.shape, dtype=np.float32))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.numpy.rand(*cell.weight.shape).astype(np.float32) * 0.02)
                if cell.bias is not None:
                    cell.bias.set_data(ms.numpy.zeros(cell.bias.shape, dtype=np.float32))

        for block in self.blocks:
            block.adaLN_modulation[-1].weight.set_data(ms.numpy.zeros(block.adaLN_modulation[-1].weight.shape, dtype=np.float32))
            block.adaLN_modulation[-1].bias.set_data(ms.numpy.zeros(block.adaLN_modulation[-1].bias.shape, dtype=np.float32))

        self.final_layer.adaLN_modulation[-1].weight.set_data(ms.numpy.zeros(self.final_layer.adaLN_modulation[-1].weight.shape, dtype=np.float32))
        self.final_layer.adaLN_modulation[-1].bias.set_data(ms.numpy.zeros(self.final_layer.adaLN_modulation[-1].bias.shape, dtype=np.float32))
        self.final_layer.linear.weight.set_data(ms.numpy.zeros(self.final_layer.linear.weight.shape, dtype=np.float32))
        self.final_layer.linear.bias.set_data(ms.numpy.zeros(self.final_layer.linear.bias.shape, dtype=np.float32))

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = ops.Transpose()(x, (0, 5, 1, 3, 2, 4))
        x = x.reshape(x.shape[0], c, h * p, h * p)
        return x

    def construct(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        training = self.training if hasattr(self, 'training') else False
        y = self.y_embedder(y, training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
    
    def forward(self, x, t, y):
        return self.construct(x, t, y)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half_idx = x.shape[0] // 2
        combined = ops.Concat(0)([x[:half_idx], x[:half_idx]])
        model_out = self.forward(combined, t, y)
        # Use all 4 channels for latent
        eps = model_out[:, :4]
        cond_eps = eps[:half_idx]
        uncond_eps = eps[half_idx:]
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # Duplicate to match original batch size
        return ops.Concat(0)([half_eps, half_eps])


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
