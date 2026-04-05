# MindSpore DiT (Scalable Diffusion Models with Transformers)

MindSpore implementation of DiT for Ascend NPU, based on [mindone](https://github.com/mindspore-lab/mindone).

## 环境配置

```bash
# 安装 MindSpore 和 mindone
pip install mindspore mindone

# 克隆 mindone (如需要)
git clone https://github.com/mindspore-lab/mindone.git
```

## 快速开始

### 1. 下载 PyTorch 预训练权重

```bash
# DiT-XL/2 (256x256)
wget https://dl.fbaipublicfiles.com/dit/dit-xl-2-256x256.pt

# DiT-XL/2 (512x512)
wget https://dl.fbaipublicfiles.com/dit/dit-xl-2-512x512.pt
```

### 2. 转换检查点

```bash
# 转换 256x256 模型
python -m mindone.examples.dit.scripts.convert_dit_checkpoint \
    --model dit_xl_2 \
    --input <path-to-dit-xl-2-256x256.pt> \
    --output DiT-XL-2-256x256.ckpt

# 转换 512x512 模型
python -m mindone.examples.dit.scripts.convert_dit_checkpoint \
    --model dit_xl_2 \
    --input <path-to-dit-xl-2-512x512.pt> \
    --output DiT-XL-2-512x512.ckpt \
    --image_size 512
```

### 3. 推理

```bash
# 使用 generate.py (推荐)
python generate.py \
    --checkpoint DiT-XL-2-256x256.ckpt \
    --image-size 256 \
    --num-samples 4 \
    --cfg-scale 4.0 \
    --seed 42

# 或使用 ms_sample.py
python ms_sample.py \
    --checkpoint DiT-XL-2-256x256.ckpt \
    --image-size 256 \
    --num-samples 4
```

### 4. 训练

```bash
python ms_train.py \
    --data-path /path/to/imagenet-mini \
    --vae-path /path/to/sd-vae-ft-mse \
    --image-size 256 \
    --epochs 100 \
    --global-batch-size 8 \
    --lr 1e-4 \
    --max-steps 100000
```

## 模型架构

| 模型 | 参数量 | 隐藏维度 | 头数 | 层数 |
|------|--------|----------|------|------|
| DiT-S/2 | 33M | 384 | 6 | 12 |
| DiT-B/2 | 130M | 768 | 12 | 12 |
| DiT-L/2 | 457M | 1024 | 16 | 24 |
| DiT-XL/2 | 657M | 1152 | 16 | 28 |

## 推理脚本说明

### generate.py (推荐)

基于 mindone 的完整推理脚本。

**参数说明:**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--checkpoint` | (必填) | MindSpore 检查点路径 |
| `--image-size` | 256 | 图像尺寸 (256 或 512) |
| `--num-samples` | 1 | 生成样本数量 |
| `--cfg-scale` | 4.0 | Classifier-free guidance 比例 |
| `--num-sampling-steps` | 50 | 采样步数 |
| `--seed` | 42 | 随机种子 |
| `--class-label` | None | 指定类别 (0-999) |
| `--vae-path` | None | VAE 路径 (用于重建) |

### ms_sample.py

简化的推理脚本。

**参数说明:**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--checkpoint` | (必填) | MindSpore 检查点路径 |
| `--model` | DiT-XL/2 | 模型架构 |
| `--image-size` | 256 | 图像尺寸 |
| `--num-samples` | 8 | 生成样本数量 |
| `--cfg-scale` | 4.0 | CFG 比例 |
| `--num-sampling-steps` | 10 | 采样步数 |
| `--seed` | 0 | 随机种子 |

## 训练脚本说明

### ms_train.py

使用 VAE 编码 ImageNet 图像进行训练。

**参数说明:**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--data-path` | (必填) | ImageNet 数据路径 |
| `--vae-path` | /home/ma-user/work/temp/sd-vae-ft-mse | VAE 路径 |
| `--results-dir` | ms_train_output | 输出目录 |
| `--image-size` | 256 | 图像尺寸 |
| `--epochs` | 1 | 训练轮数 |
| `--global-batch-size` | 4 | 批量大小 |
| `--lr` | 1e-4 | 学习率 |
| `--max-steps` | None | 最大步数 |
| `--log-every` | 10 | 日志频率 |
| `--ckpt-every` | 1 | 检查点保存频率 |
| `--num-samples` | None | 使用样本数 (默认全部) |
| `--device-id` | 0 | NPU 设备 ID |

**示例:**

```bash
# 训练 1000 步
python ms_train.py \
    --data-path /home/ma-user/work/temp/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini \
    --vae-path /home/ma-user/work/temp/sd-vae-ft-mse \
    --max-steps 1000 \
    --global-batch-size 4

# 完整训练
python ms_train.py \
    --data-path /path/to/imagenet/train \
    --vae-path /path/to/sd-vae-ft-mse \
    --epochs 100 \
    --global-batch-size 8 \
    --lr 1e-4
```

## 检查点格式

MindSpore 检查点包含以下参数:
- `pos_embed.pos_embed`: 位置嵌入
- `patch_embed.*`: 补丁嵌入层
- `transformer_blocks.*`: Transformer 块
- `final_layer.*`: 最终输出层
- ` timestep_embedder.*`: 时间步嵌入
- `label_embedder.*`: 类别嵌入

## 与 PyTorch 版本对比

| 特性 | PyTorch | MindSpore |
|------|---------|-----------|
| 推理脚本 | sample.py | generate.py / ms_sample.py |
| 训练脚本 | train.py | ms_train.py |
| 模型定义 | models.py | mindone DiTTransformer2DModel |
| 设备支持 | CUDA | Ascend NPU |

## 注意事项

1. **mindone 依赖**: 使用 mindone 的 DiTTransformer2DModel 和相关模块
2. **VAE**: 训练需要 VAE 进行图像编码,推荐使用 sd-vae-ft-mse
3. **ImageNet**: 数据集需要包含 train/val 目录结构,每个类别一个子目录
4. **设备**: 脚本默认使用 Ascend NPU (device_id=0)
5. **混合精度**: 当前版本未启用 AMP

## 目录结构

```
DiT/
├── generate.py          # 推理脚本 (mindone)
├── ms_sample.py         # 简化推理脚本
├── ms_train.py          # 训练脚本
├── ms_models.py         # 模型封装
├── dit/                 # mindone 训练模块
│   ├── train_pipeline.py
│   └── dataset.py
└── ms_checkpoints/      # 转换后的检查点
    └── DiT-XL-2-256x256.ckpt
```

## 许可证

与原始 DiT 项目相同，采用 CC-BY-NC 许可证。