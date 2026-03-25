# MindSpore DiT (Scalable Diffusion Models with Transformers)

MindSpore implementation of DiT for Ascend NPU.

## 环境配置

```bash
# 安装 MindSpore
pip install mindspore

# 或使用 conda
conda install mindspore -c mindspore
```

## 模型架构

支持以下 DiT 模型配置:

| 模型 | 参数量 | 隐藏维度 | 头数 | 层数 |
|------|--------|----------|------|------|
| DiT-S/2 | 33M | 384 | 6 | 12 |
| DiT-B/2 | 130M | 768 | 12 | 12 |
| DiT-L/2 | 457M | 1024 | 16 | 24 |
| DiT-XL/2 | 657M | 1152 | 16 | 28 |

## 推理

### 命令行参数

```bash
python ms_sample.py [选项]
```

**参数说明:**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model` | DiT-XL/2 | 模型架构 |
| `--image-size` | 256 | 图像尺寸 (256 或 512) |
| `--num-classes` | 1000 | 类别数量 |
| `--num-samples` | 8 | 生成样本数量 |
| `--cfg-scale` | 4.0 | Classifier-free guidance 比例 |
| `--num-sampling-steps` | 10 | 采样步数 |
| `--seed` | 0 | 随机种子 |

### 使用示例

```bash
# 使用 DiT-S/2 生成 256x256 图像
python ms_sample.py --model DiT-S/2 --image-size 256 --num-sampling-steps 20

# 使用 DiT-XL/2 生成 512x512 图像
python ms_sample.py --model DiT-XL/2 --image-size 512 --num-sampling-steps 50 --cfg-scale 4.0
```

### 推理输出

生成的图像保存在当前目录，文件名格式为 `ms_sample_0.png`, `ms_sample_1.png`, ...

## 训练

### 命令行参数

```bash
python ms_train.py --data-path <数据集路径> [选项]
```

**参数说明:**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--data-path` | (必填) | ImageNet 训练数据路径 |
| `--results-dir` | ms_results | 结果保存目录 |
| `--model` | DiT-XL/2 | 模型架构 |
| `--image-size` | 256 | 图像尺寸 |
| `--num-classes` | 1000 | 类别数量 |
| `--epochs` | 1 | 训练轮数 |
| `--global-batch-size` | 4 | 全局批量大小 |
| `--global-seed` | 0 | 随机种子 |
| `--log-every` | 10 | 日志输出频率 |
| `--ckpt-every` | 100 | 检查点保存频率 |
| `--test` | False | 测试模式 (只运行几步) |

### 使用示例

```bash
# 使用 DiT-S/2 训练 (测试模式)
python ms_train.py --model DiT-S/2 --data-path /path/to/imagenet/train --test

# 使用 DiT-B/2 完整训练
python ms_train.py --model DiT-B/2 --data-path /path/to/imagenet/train \
    --epochs 100 --global-batch-size 16 --image-size 256
```

### 检查点

训练过程中检查点保存在 `--results-dir` 指定的目录:
- 格式: `ckpt_0001000.ckpt`, `ckpt_0002000.ckpt`, ...
- 最终检查点: `final.ckpt`

### 加载检查点进行推理

```python
import mindspore as ms
from ms_models import DiT_models

model = DiT_models["DiT-S/2"](input_size=32, num_classes=1000)
ms.load_checkpoint("ms_results/final.ckpt", model)
model.set_train(False)
```

## 与 PyTorch 版本对比

| 特性 | PyTorch | MindSpore |
|------|---------|-----------|
| 推理脚本 | sample.py | ms_sample.py |
| 训练脚本 | train.py | ms_train.py |
| 模型定义 | models.py | ms_models.py |
| 设备支持 | CUDA/NPU | Ascend NPU |

## 注意事项

1. **当前版本使用简化 Diffusion**: 使用简化的 DDPM 采样和训练过程
2. **数据集**: 完整训练需要 ImageNet 数据集
3. **设备**: 脚本默认使用 Ascend NPU (device_id=0)
4. **混合精度**: 当前版本未启用 AMP

## 许可证

与原始 DiT 项目相同，采用 CC-BY-NC 许可证。
