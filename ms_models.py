# MindSpore DiT Model - wrapper for mindone's DiTTransformer2DModel

from mindone.diffusers import DiTTransformer2DModel

# Export DiT models dict for compatibility
DiT_models = {
    'DiT-XL/2': lambda **kwargs: DiTTransformer2DModel(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        num_layers=28,
        patch_size=2,
        sample_size=kwargs.get('input_size', 32) // 2,
        num_embeds_ada_norm=1000,
    ),
    'DiT-L/2': lambda **kwargs: DiTTransformer2DModel(
        num_attention_heads=16,
        attention_head_dim=64,
        in_channels=4,
        out_channels=8,
        num_layers=24,
        patch_size=2,
        sample_size=kwargs.get('input_size', 32) // 2,
        num_embeds_ada_norm=1000,
    ),
    'DiT-B/2': lambda **kwargs: DiTTransformer2DModel(
        num_attention_heads=12,
        attention_head_dim=64,
        in_channels=4,
        out_channels=8,
        num_layers=12,
        patch_size=2,
        sample_size=kwargs.get('input_size', 32) // 2,
        num_embeds_ada_norm=1000,
    ),
}

# For backward compatibility
DiT = DiTTransformer2DModel