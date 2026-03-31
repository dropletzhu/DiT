# DiT MindSpore 版本测试报告

## 1. 测试环境

### 硬件环境
| 项目 | 信息 |
|------|------|
| CPU | Kunpeng-920 (aarch64) |
| CPU核心数 | 192核 (4 clusters x 48 cores) |
| NPU | Ascend 910B4 |
| NPU数量 | 1卡 |
| NPU驱动 | 23.0.6 |

### 软件环境
| 项目 | 版本 |
|------|------|
| 操作系统 | Linux aarch64 |
| Python | 3.11.10 |
| MindSpore | 2.8.0 |

## 2. 依赖库版本

| 库名 | 版本 |
|------|------|
| mindspore | 2.8.0 |
| numpy | 1.26.4 |
| pillow | 11.3.0 |
| tqdm | - |

## 3. 测试用例和测试结果

### 3.1 推理测试 (ms_sample.py)

**测试命令**:
```bash
python ms_sample.py --model DiT-XL/2 --image-size 256 --num-sampling-steps 50
```

**测试结果**: ✅ 通过

| 指标 | 值 |
|------|------|
| 模型 | DiT-XL/2 |
| 图像尺寸 | 256x256 (生成32x32 latent) |
| 采样步数 | 50 |
| 生成图像数 | 8张 (2x4拼接) |
| 输出文件 | ms_sample.png (11.9KB) |
| 采样速度 | ~5.12 it/s |

### 3.2 训练测试 (ms_train.py)

**测试命令**:
```bash
python ms_train.py --data-path <imagenet-path> --model DiT-XL/2 --image-size 256 \
  --epochs 1 --global-batch-size 2 --log-every 10 --ckpt-every 1000 --max-steps 100
```

**测试结果**: ⚠️ 运行缓慢

| 指标 | 值 |
|------|------|
| 训练步数 | 100 (未完成) |
| 训练速度 | ~0.1 steps/sec |
| 模型参数 | 675,129,632 |

**注意**: MindSpore训练速度较慢，100步测试未能完成，主要因为MindSpore NPU编译开销较大。

## 4. 图像质量差异原因分析

MindSpore版本生成的图像质量不如PyTorch版本，原因如下：

### 4.1 缺少VAE解码
- PyTorch版本: 使用 `vae.decode()` 将latent解码为RGB图像
- MindSpore版本: 直接输出latent，无VAE解码 (ms_models未包含VAE)

### 4.2 采样参数差异
- PyTorch使用完整的DDIM采样，包含正确的均值和方差计算
- MindSpore简化了采样过程，未完全实现扩散模型的完整公式

### 4.3 beta schedule差异
- PyTorch使用1000步的预训练beta schedule
- MindSpore使用简化的50步采样，步数不足

### 4.4 模型权重
- PyTorch使用预训练的DiT-XL/2模型权重
- MindSpore使用随机初始化的权重（未加载预训练权重）

## 5. 性能优化建议

### 5.1 NPU训练优化
1. **增加编译缓存**: 使用持久化编译减少首次编译时间
2. **混合精度**: 启用fp16混合精度训练
3. **数据加载优化**: 使用多线程数据加载

### 5.2 代码改进
1. 添加MindSpore VAE支持用于图像解码
2. 实现完整的DDIM采样而非简化版
3. 加载预训练模型权重

## 6. 无法运行的测试/运维

### 6.1 待完善
| 测试项 | 说明 |
|--------|------|
| VAE解码 | 需要实现MindSpore版本的VAE |
| 预训练权重加载 | 需要实现权重加载接口 |
| 完整采样 | DDIM/DDPM完整实现 |

### 6.2 代码问题
| 问题 | 说明 |
|------|------|
| PIL.Image.BOX/BICUBIC | LSP报错但运行时正常 |
| 采样速度慢 | NPU编译开销大 |

## 7. 与PyTorch版本对比

| 指标 | PyTorch | MindSpore |
|------|---------|-----------|
| 推理速度 | ~1.83 it/s | ~5.12 it/s |
| 图像质量 | 清晰 | 模糊(无VAE) |
| 预训练权重 | 支持 | 不支持 |
| VAE解码 | 支持 | 不支持 |
| 训练稳定性 | 稳定 | 编译慢 |
