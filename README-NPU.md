# DiT on Ascend NPU - PyTorch & MindSpore

本文档描述了 DiT 在华为 Ascend 910B3 NPU 上的移植、训练和推理方案，包括 PyTorch (torch_npu) 和 MindSpore 两种实现。

## 目录

- [1. 环境配置](#1-环境配置)
- [2. PyTorch 版本 NPU 适配](#2-pytorch-版本-npu-适配)
- [3. MindSpore 版本实现](#3-mindspore-版本实现)
- [4. 训练和推理使用方法](#4-训练和推理使用方法)
- [5. 性能对比](#5-性能对比)
- [6. 移植过程中的修改和优化](#6-移植过程中的修改和优化)
- [7. 优化路线图](#7-优化路线图)

---

## 1. 环境配置与使用方法

### 1.1 硬件要求

- Ascend 910B3 NPU (64GB HBM)
- 推荐 8 卡配置用于训练，单卡可用于推理
- 同时支持 NVIDIA GPU (CUDA 11.8+)

### 1.2 软件依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | |
| CANN Toolkit | 8.3.RC1 | Ascend NPU 驱动 |
| CANN Kernels 910B | 8.3.RC1 | 910B 算子库 |
| PyTorch | 2.9.0 | 通用深度学习框架 |
| torch-npu | 2.9.0 | PyTorch NPU 适配插件 |
| torchvision | 0.24.0 | 图像工具库 |
| MindSpore | 2.8.0 | 华为原生框架 |
| mindone | 0.5.0 | MindSpore DiT 模型库 |
| diffusers | 0.38.0 | VAE 模型加载 |
| timm | 1.0.26 | PyTorch 模型组件 |
| transformers | 4.57.1 | HuggingFace 工具 |
| numpy | 1.24.0 | |
| scipy | 1.15.3 | |
| pillow | 12.2.0 | 图像处理 |
| opencv-python | 4.11.0.86 | |
| tqdm | 4.67.3 | 进度条 |

### 1.3 环境创建

#### 方式一：Conda 一键创建（推荐）

```bash
# 使用提供的 environment-npu.yml 创建环境
conda env create -f environment-npu.yml
conda activate DiT-NPU
```

#### 方式二：手动安装

```bash
# 1. 创建 conda 环境
conda create -n DiT-NPU python=3.10 -y
conda activate DiT-NPU

# 2. 安装 CANN (NPU 驱动和算子库)
conda install -c ascend cann-toolkit=8.3.RC1 cann-kernels-910b=8.3.RC1

# 3. 安装 MindSpore
conda install -c mindspore mindspore=2.8.0

# 4. 安装 PyTorch 和 torch-npu
pip install torch==2.9.0 torch-npu==2.9.0 torchvision==0.24.0

# 5. 安装其余依赖
pip install -r requirements-npu.txt
```

### 1.4 环境验证

```bash
# 设置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证 PyTorch + NPU
python -c "import torch; import torch_npu; print(f'PyTorch: {torch.__version__}, NPU available: {torch.npu.is_available()}')"

# 验证 MindSpore
python -c "import mindspore as ms; ms.set_context(device_target='Ascend'); print(f'MindSpore: {ms.__version__}')"

# 验证 NPU 硬件
npu-smi info
```

### 1.5 权重和数据准备

```bash
# 1. 下载预训练 DiT 权重
# 放到 checkpoints/ 目录
ls checkpoints/
# DiT-XL-2-256x256.pt  sd-vae-ft-mse/

# 2. 准备 ImageNet 数据集
# 数据目录结构：
# /path/to/imagenet/train/
#   0/  (class 0)
#     imagenet_00000001.JPEG
#     ...
#   1/  (class 1)
#     ...
#   999/
#     ...
```

### 1.6 快速开始

#### PyTorch 版本（支持 GPU 和 NPU）

```bash
# 单卡推理（自动检测设备）
python sample.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --seed 0 \
  --ckpt checkpoints/DiT-XL-2-256x256.pt \
  --vae-path checkpoints/sd-vae-ft-mse

# 指定使用 NPU
python sample.py --device npu --ckpt checkpoints/DiT-XL-2-256x256.pt --vae-path checkpoints/sd-vae-ft-mse

# 指定使用 GPU
python sample.py --device cuda --ckpt checkpoints/DiT-XL-2-256x256.pt --vae-path checkpoints/sd-vae-ft-mse

# 8 卡训练
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --data-path /path/to/imagenet/train \
  --amp --dtype bf16 \
  --max-steps 500000 \
  --ckpt-every 10000 \
  --vae-path checkpoints/sd-vae-ft-mse
```

#### MindSpore 版本（仅支持 NPU）

```bash
# 单卡推理（PyTorch .pt 权重）
python ms_sample.py \
  --checkpoint checkpoints/DiT-XL-2-256x256.pt \
  --image_size 256 \
  --vae_path checkpoints/sd-vae-ft-mse

# 单卡推理（MindSpore .ckpt 权重，MODEL 权重，非 EMA）
python ms_sample.py \
  --checkpoint results/001-DiT-XL-2/checkpoints/0225000.ckpt \
  --image_size 256 \
  --vae_path checkpoints/sd-vae-ft-mse

# 8 卡训练
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_train.py \
    --data-path /path/to/imagenet \
    --model DiT-XL/2 \
    --image-size 256 \
    --global-batch-size 256 \
    --max-steps 500000 \
    --amp --dtype bf16 --amp-level O2 \
    --vae-path checkpoints/sd-vae-ft-mse \
    --exec-mode pynative

# 8 卡分布式推理
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_sample_ddp.py \
    --checkpoint checkpoints/DiT-XL-2-256x256.pt \
    --num-fid-samples 50000 \
    --vae-path checkpoints/sd-vae-ft-mse
```

### 1.7 注意事项

1. **CANN 环境变量**：每次使用 NPU 前需要执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
2. **MindSpore 权重兼容**：
   - **`.pt` 权重**：MindSpore 可直接加载 PyTorch 的 `.pt` 权重文件，使用 `DiT_models` 架构（292 参数），无需转换
   - **`.ckpt` 权重**：MindSpore 训练产生的 `.ckpt` 文件使用 `DiTTransformer2DModel` 架构（539 参数），可自动加载推理
   - **EMA 权重缺陷**：早期训练版本（< 236K 步）因 `update_ema()` 参数名匹配错误，EMA 权重从未被更新（始终为随机初始化），推理时必须选择 MODEL 权重而非 EMA
3. **MindSpore 执行模式**：推荐使用 `pynative` 模式，`graph` 模式编译时间过长（DiT-XL/2 >77分钟）
4. **VAE 路径**：推理和训练都需要指定本地 VAE 路径 `--vae-path`
5. **多卡启动**：PyTorch 使用 `torchrun`，MindSpore 使用 `msrun`
6. **设备选择**：PyTorch 版本通过 `--device` 参数选择 GPU/NPU，MindSpore 版本仅支持 NPU

---

## 2. PyTorch 版本 NPU 适配

### 设计原则

PyTorch 版本同时支持 NVIDIA GPU 和华为 NPU，通过设备抽象层实现自动检测和切换。

### 关键改动

#### 2.1 设备抽象层 (`utils/`)

新增 `utils/__init__.py`，提供统一的设备接口：

```python
from utils import get_device, get_device_str, set_device, synchronize

# 自动检测设备 (优先级: NPU > CUDA > CPU)
device = get_device()

# 获取设备字符串
device_str = get_device_str()  # "npu" 或 "cuda" 或 "cpu"

# 设置设备
set_device(device_id)

# 同步设备
synchronize()
```

#### 2.2 分布式后端适配

```python
from utils import get_distributed_backend

# GPU 返回 "nccl"，NPU 返回 "hccl"
backend = get_distributed_backend()
dist.init_process_group(backend)
```

#### 2.3 AMP 混合精度适配

```python
from utils import get_autocast, get_amp_scaler

# 自动选择 torch.cuda.amp 或 torch.npu.amp
autocast = get_autocast(enabled=True, dtype=torch.bfloat16)
scaler = get_amp_scaler()
```

#### 2.4 命令行参数

所有脚本新增 `--device` 参数：

```bash
# 自动检测 (默认)
python sample.py --device auto

# 指定 NPU
python sample.py --device npu

# 指定 GPU
python sample.py --device cuda
```

### 修改的文件

| 文件 | 改动说明 |
|------|---------|
| `utils/__init__.py` | 新增设备抽象层 |
| `train.py` | 使用设备无关 API，支持 NPU 训练 |
| `sample.py` | 使用设备无关 API，支持 NPU 推理 |
| `sample_ddp.py` | 使用设备无关 API，支持 NPU 分布式推理，新增 `--vae-path` 参数 |

---

## 3. MindSpore 版本实现

### 模型结构

MindSpore 版本支持两种模型架构，对应不同的权重格式：

#### 3.1 PT 兼容架构：`DiT_models`（用于 `.pt` 权重）

使用 `mindone.models.dit.DiT_models`，与 PyTorch 版本模型结构完全一致，含 292 个参数：

| 组件 | PyTorch | MindSpore |
|------|---------|-----------|
| Patch Embed | `timm.PatchEmbed` | `mindone.models.dit.PatchEmbed` |
| Attention | `timm.Attention` | `mindone.models.dit.SelfAttention` |
| MLP | `timm.Mlp` | `mindone.models.dit.Mlp` |
| Timestep Embed | `TimestepEmbedder` | `mindone.models.dit.TimestepEmbedder` |
| Label Embed | `LabelEmbedder` | `mindone.models.dit.LabelEmbedder` |
| DiT Block | `DiTBlock` | `mindone.models.dit.DiTBlock` |
| Final Layer | `FinalLayer` | `mindone.models.dit.FinalLayer` |

直接加载 PyTorch 的 `.pt` 权重即可使用（参数名完全一致）。

#### 3.2 MS 训练架构：`DiTTransformer2DModel`（用于 `.ckpt` 权重）

使用 `mindone.diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel`，含 539 个参数。相比 `DiT_models`，它对每个 block 使用独立的 AdaLN 条件嵌入（独立的 t_embedder/y_embedder），并且 QKV 投影分开：

| 组件 | `DiT_models` | `DiTTransformer2DModel` |
|------|-------------|------------------------|
| 参数数量 | 292 | 539 |
| 权重精度 | 权重值完全对齐 PT | MS 训练产生，权重值与 PT 不同 |
| AdaLN 条件 | 共享 t/y embedder | 每个 block 独立的 t/y embedder（`adaLN_modulation`） |
| QKV | 合并投影 | 独立 Q/K/V 投影 |
| 适用场景 | PT 权重推理 | MS 训练权重推理 |

**参数量对比：**

| 模型 | PyTorch | MindSpore (`DiT_models`) | MindSpore (`DiTTransformer2DModel`) |
|------|---------|--------------------------|-------------------------------------|
| DiT-XL/2 | 675,129,632 | 675,129,632 | 675,129,632 |
| 参数数量 | 292 | 292 | 539 |
| 权重值 | 原始 | 与 PT 完全一致 | MS 独立训练产生 |

### 脚本说明

| 脚本 | 功能 |
|------|------|
| `ms_train.py` | MindSpore 训练脚本，支持 8 卡分布式训练，使用 `DiTTransformer2DModel` |
| `ms_sample.py` | MindSpore 单卡推理脚本，支持 `.pt`（`DiT_models`）和 `.ckpt`（`DiTTransformer2DModel`） |
| `ms_sample_ddp.py` | MindSpore 多卡分布式推理脚本，自动根据扩展名选择架构 |

---

## 4. 训练和推理使用方法

### 4.1 PyTorch 训练 (GPU/NPU)

```bash
# GPU 训练
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --data-path /path/to/imagenet/train \
  --max-steps 500000 \
  --ckpt-every 10000

# NPU 训练 (自动检测)
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --data-path /path/to/imagenet/train \
  --amp --dtype bf16 \
  --max-steps 500000 \
  --ckpt-every 10000 \
  --device npu \
  --vae-path /path/to/sd-vae-ft-mse
```

### 4.2 MindSpore 训练 (NPU)

```bash
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_train.py \
    --data-path /path/to/imagenet \
    --model DiT-XL/2 \
    --image-size 256 \
    --global-batch-size 256 \
    --max-steps 500000 \
    --amp --dtype bf16 --amp-level O2 \
    --vae-path /path/to/sd-vae-ft-mse \
    --num-workers 4 \
    --log-every 10 \
    --ckpt-every 10000 \
    --exec-mode pynative
```

### 4.3 PyTorch 推理 (GPU/NPU)

```bash
# 单卡推理
python sample.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-sampling-steps 250 \
  --cfg-scale 4.0 \
  --seed 0 \
  --ckpt /path/to/DiT-XL-2-256x256.pt \
  --vae-path /path/to/sd-vae-ft-mse \
  --device npu

# 多卡分布式推理 (FID 评估)
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-fid-samples 50000 \
  --cfg-scale 4.0 \
  --num-sampling-steps 250 \
  --ckpt /path/to/DiT-XL-2-256x256.pt \
  --vae-path /path/to/sd-vae-ft-mse \
  --device npu
```

### 4.4 MindSpore 推理 (NPU)

```bash
# 单卡推理（PT 预训练权重，.pt）
python ms_sample.py \
  --checkpoint /path/to/DiT-XL-2-256x256.pt \
  --image_size 256 \
  --num_sampling_steps 250 \
  --cfg_scale 4.0 \
  --seed 0 \
  --vae_path /path/to/sd-vae-ft-mse \
  --device_id 0

# 单卡推理（MS 训练权重，.ckpt，自动选择 MODEL 而非 EMA）
python ms_sample.py \
  --checkpoint /path/to/0225000.ckpt \
  --image_size 256 \
  --num_sampling_steps 250 \
  --cfg_scale 4.0 \
  --seed 0 \
  --vae_path /path/to/sd-vae-ft-mse \
  --device_id 0

# 多卡分布式推理（PT 权重）
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_sample_ddp.py \
    --checkpoint /path/to/DiT-XL-2-256x256.pt \
    --sample-dir ms_samples \
    --num-fid-samples 50000 \
    --per-proc-batch-size 8 \
    --cfg-scale 4.0 \
    --num-sampling-steps 250 \
    --global-seed 0 \
    --vae-path /path/to/sd-vae-ft-mse

# 多卡分布式推理（MS 训练权重）
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_sample_ddp.py \
    --checkpoint /path/to/0225000.ckpt \
    --sample-dir ms_samples \
    --num-fid-samples 50000 \
    --per-proc-batch-size 8 \
    --cfg-scale 4.0 \
    --num-sampling-steps 250 \
    --global-seed 0 \
    --vae-path /path/to/sd-vae-ft-mse
```

### 4.5 权重说明

MindSpore 版本支持两种权重格式：

1. **`.pt` 文件（PyTorch 权重）**：使用 `DiT_models` 架构（292参数），参数名与 PyTorch 完全一致，可直接加载无需转换。适用于 PT 预训练权重或 PT 训练产生的权重。

2. **`.ckpt` 文件（MindSpore 训练权重）**：使用 `DiTTransformer2DModel` 架构（539参数），由 `ms_train.py` 产生。关键注意点：
   - **选择 MODEL 而非 EMA**：由于早期训练（< 236K 步）的 EMA 权重因 `update_ema()` bug 从未被更新（始终为随机初始化），推理时必须加载 MODEL 权重。`ms_sample.py` 和 `ms_sample_ddp.py` 会自动从 `.ckpt` 中提取 `model.model._backbone.*` 前缀的权重。
   - **pos_embed 特殊处理**：`pos_embed.pos_embed` 保存为独立 key（`model.pos_embed.pos_embed`），不在 `_backbone.*` 范围内，推理脚本会自动加载并调用 `assign_value()` 注入模型。

---

## 5. 性能对比

### 5.1 训练性能

#### DiT-XL/2, 256x256, 8x 910B3 NPU

**PyTorch + torch_npu vs MindSpore（相同参数）**

| 配置 | 框架 | Batch Size | Steps/Sec | 单步耗时 | HBM 使用 |
|------|------|-----------|-----------|---------|---------|
| BS=256 | PyTorch + torch_npu | 256 (每卡32) | ~1.07 | 0.93s | ~37GB |
| BS=256 | MindSpore (pynative) | 256 (每卡32) | ~1.07 | 0.93s | ~37GB |
| BS=512 | MindSpore (pynative) | 512 (每卡64) | ~0.76 | 1.32s | ~51GB |

**A100 参考性能**

| 配置 | 设备 | Batch Size | Steps/Sec | 单步耗时 |
|------|------|-----------|-----------|---------|
| BS=256 | 8x A100 (80GB) | 256 | ~1.75 | 0.57s |
| BS=256 | 8x 910B3 (64GB) | 256 | ~1.07 | 0.93s |

> 910B3 比 A100 慢约 40%，主要因为计算架构和软件栈成熟度差异。

#### 训练性能优化

通过增大 Batch Size 和调整训练参数，显著提升了训练吞吐量：

| 优化项 | 优化前 | 优化后 | 效果 |
|--------|--------|--------|------|
| Global Batch Size | 256 | 512 | 每步样本数翻倍 |
| num_workers | 4 | 8 | 数据加载加速 |
| ckpt 保存间隔 | 10,000 步 | 25,000 步 | IO 等待减少 |
| max_steps | 500,000 | 250,000 | 总步数减半（保持相同总样本数）|
| HBM 使用率 | 57% (37GB/64GB) | 78% (51GB/64GB) | 资源利用率提升 |
| 样本吞吐量 | 274 samples/sec | 389 samples/sec | **提升 42%** |

**优化前后总训练时间对比：**

| 配置 | 总步数 | Steps/Sec | 总时间 | 节省 |
|------|--------|-----------|--------|------|
| 优化前 (BS=256) | 500K | 1.07 | ~130 小时 | - |
| 优化后 (BS=512) | 250K | 0.76 | ~91 小时 | **~39 小时** |

#### 单卡训练性能

| 配置 | 设备 | 模型 | Steps/Sec | 备注 |
|------|------|------|-----------|------|
| MindSpore (graph, O1) | 1x 910B3 | DiT-B/2 | 0.02 | 图编译时间长，不实用 |
| MindSpore (pynative) | 1x 910B3 | DiT-B/2 | 3.34 | |

#### 断点续训

训练脚本支持从 checkpoint 恢复，自动恢复模型权重、EMA 权重和训练步数：

```bash
# 从 step 140000 的 checkpoint 恢复训练
msrun --worker_num=8 --local_worker_num=8 \
  --master_addr=127.0.0.1 --master_port=12345 \
  --join=True \
  python ms_train.py \
    --data-path /path/to/imagenet \
    --model DiT-XL/2 --image-size 256 \
    --global-batch-size 512 --max-steps 250000 \
    --amp --dtype bf16 --amp-level O2 \
    --vae-path /path/to/sd-vae-ft-mse \
    --num-workers 8 --ckpt-every 25000 \
    --exec-mode pynative \
    --ckpt results/001-DiT-XL-2/checkpoints/0140000.ckpt
```

### 5.2 推理性能

#### 单卡推理 (DiT-XL/2, 256x256, 250 steps, CFG=4.0)

| 配置 | 设备 | 推理时间 | it/s | 备注 |
|------|------|---------|------|------|
| PyTorch + torch_npu | 1x 910B3 | ~32s | ~7.8 | |
| MindSpore (pynative) | 1x 910B3 | ~24s | ~10.4 | MindSpore 推理更快 |

#### 多卡推理 (DiT-XL/2, 256x256, 250 steps, 8卡)

| 配置 | 设备 | 16张图片耗时 | 备注 |
|------|------|------------|------|
| PyTorch + torch_npu | 8x 910B3 | ~31s | |
| MindSpore (pynative) | 8x 910B3 | ~34s | |

### 5.3 910B3 vs A100 性能对比

| 任务 | 8x A100 (80GB) | 8x 910B3 (64GB) | 差距 |
|------|---------------|----------------|------|
| 训练 (BS=256, steps/sec) | ~1.75 | ~1.07 | 910B3 慢 40% |
| 训练优化后 (BS=512, samples/sec) | TODO | 389 | TODO |
| 推理 (单卡, 250步) | TODO | ~24s (MS) / ~32s (PT) | TODO |
| HBM 使用 (BS=256) | TODO | ~37GB/NPU | TODO |
| HBM 使用 (BS=512) | TODO | ~51GB/NPU | TODO |

### 5.4 PyTorch vs MindSpore 推理输出对比

| 场景 | 指标 | 值 | 说明 |
|------|------|-----|------|
| PT → MS (同权重, `.pt`) | 权重差异 | 0 | 292参数完全一致 |
| PT → MS (同权重, `.pt`) | 模型前向传播差异 | < 0.001 | 数值精度差异 |
| PT → MS (同权重, `.pt`) | 最终图片像素差异 | ~60 | 随机数生成器不同导致 |
| MS 训练权重 (`.ckpt`) | 输出有效性 | ✅ | t=996下std=0.994（匹配PT 0.995）|
| MS 训练权重 vs PT 预训练 | 视觉质量 | 可比 | 训练225K步，loss=0.017 |

> **注：**
> - **PT 权重 + MS 推理**：PyTorch 和 MindSpore 使用不同的随机数生成器，即使设置相同种子，生成的随机数序列也不同。这导致推理输出的图片在像素级别有差异，但视觉上相似。模型权重、前向传播逻辑、CFG 实现、diffusion 采样调度完全一致。
> - **MS 训练权重推理**：MS 训练 225K 步（loss 0.017）的 MODEL 权重可产生正确图像；而 EMA 权重因 `update_ema()` bug 始终为随机初始化，输出为噪声。

---

## 6. 移植过程中的修改和优化

### 6.1 PyTorch NPU 适配

1. **设备抽象层**：新增 `utils/__init__.py`，封装 GPU/NPU 差异
2. **分布式后端**：自动选择 NCCL (GPU) 或 HCCL (NPU)
3. **AMP 适配**：支持 `torch.npu.amp` 自动混合精度
4. **数据加载**：`DataLoader` 兼容 NPU 设备
5. **VAE 路径**：新增 `--vae-path` 参数支持本地 VAE 模型

### 6.2 MindSpore 移植

1. **模型定义**：
   - 推理 `.pt` 权重：使用 `mindone.models.dit.DiT_models`（292参数），与 PyTorch 结构完全一致
   - 训练和 `.ckpt` 推理：使用 `mindone.diffusers.DiTTransformer2DModel`（539参数），每个 block 独立的 AdaLN 条件嵌入
2. **权重加载**：
   - `.pt`：直接加载 PyTorch 文件，无需转换
   - `.ckpt`：提取 `model.model._backbone.*` 前缀的权重（MODEL），同时额外加载 `model.pos_embed.pos_embed`
3. **Diffusion 采样**：使用 PyTorch 的 `create_diffusion` 库，确保调度参数一致
4. **Timestep 映射**：使用 `timestep_map` 将 respaced timestep 映射到原始 timestep
5. **CFG 实现**：与 PyTorch 完全一致，只对前 3 个通道应用 classifier-free guidance
6. **`clip_denoised=False`**：与 PT DDPM 调用保持一致，推理时不将预测的 x_start 裁剪到 [-1,1]
7. **训练脚本**：
   - 支持 pynative 和 graph 两种执行模式
   - 使用 `msrun` 启动分布式训练
   - 支持 AMP O2 混合精度
   - 支持梯度裁剪
   - 支持 EMA 更新（需注意参数名匹配方式）
   - 支持 checkpoint 恢复（自动恢复模型权重、EMA、pos_embed、步数）

### 6.3 已解决的问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| MindSpore 图片乱码 | Diffusion 调度参数计算错误 | 使用 PyTorch 的 `create_diffusion` |
| MindSpore 图片乱码 | Timestep 未映射到原始值 | 使用 `timestep_map` 映射 |
| MindSpore 图片乱码 | CFG 应用到错误通道数 | 只对前 3 个通道应用 CFG |
| **EMA 权重从未更新** | `update_ema()` 中 `p.name`（仅"weight"）与 `parameters_and_names()`（完整路径）不匹配，导致 0 参数匹配 | 改用 `parameters_and_names()` 完整路径匹配 |
| **pos_embed 未加载** | `pos_embed.pos_embed` 保存为独立 key `model.pos_embed.pos_embed`，不在 `_backbone.*` 范围内 | 推理脚本单独提取并调用 `assign_value()` |
| **`clip_denoised` 错误** | MS 推理始终将 `pred_xstart` 裁剪到 [-1,1]，但 PT 使用 `clip_denoised=False` | 移除裁剪逻辑，与 PT DDPM 行为一致 |
| **训练 AllReduce 错误** | `ops.AllReduce(ReduceOp.SUM)(g)` 未除以 world_size，梯度累加 | 修改为 `g / world_size` |
| 图模式编译超时 | DiT-XL/2 图编译时间 >77 分钟 | 使用 pynative 模式 |
| HCCL 初始化失败 | `init()` 顺序问题 | `init()` 必须在模型创建前调用 |
| GeneratorDataset 死锁 | PIL 在 graph 模式不兼容 | 使用 `ImageFolderDataset` |

---

## 7. 优化路线图

### 7.1 MindSpore 版本优化

- [x] **EMA 修复**：`update_ema()` 参数名匹配 bug 已修复（改用完整路径匹配）
- [x] **推理修复**：pos_embed 加载、clip_denoised=False、MODEL 权重选择均已修复
- [x] **断点恢复**：支持从 `.ckpt` 恢复训练（模型、EMA、pos_embed、步数）
- [ ] **图模式优化**：解决 DiT-XL/2 图编译时间过长问题，尝试增量编译或子图拆分
- [ ] **Flash Attention**：启用 MindSpore Flash Attention 加速注意力计算
- [ ] **梯度检查点**：使用梯度检查点减少内存占用，支持更大 batch size
- [ ] **VAE 预编码**：预计算 VAE latents，避免每步重复编码
- [ ] **数据加载优化**：使用 MindSpore 原生数据管线，避免 PIL 依赖
- [ ] **编译缓存**：启用图编译缓存，减少重复编译时间
- [ ] **混合精度优化**：探索 O3 级别 AMP，进一步加速

### 7.2 PyTorch NPU 版本优化

- [ ] **torch.compile**：在 NPU 上启用 `torch.compile` 加速
- [ ] **Flash Attention**：启用 NPU Flash Attention
- [ ] **梯度检查点**：减少内存占用
- [ ] **TF32 等效**：探索 NPU 上的 TF32 替代方案

### 7.3 通用优化

- [x] **FID 评估流程**：`ms_sample_ddp.py` 支持分布式生成用于 FID 评估的样本
- [x] **EMA 采样**：EMA 修复后可从 EMA 生成样本（需使用 >=236K 步训练的新 checkpoint）
- [ ] **训练监控**：添加 TensorBoard / MindInsight 日志
- [ ] **分布式数据并行优化**：优化 AllReduce 通信效率
- [ ] **GitHub 推送**：解决网络问题完成远程仓库推送
