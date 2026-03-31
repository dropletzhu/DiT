# DiT PyTorch 版本测试报告

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
| Conda环境 | PyTorch-2.7.1 |

## 2. 依赖库版本

| 库名 | 版本 |
|------|------|
| torch | 2.7.1+cpu (with NPU support) |
| torch_npu | 2.7.1 |
| diffusers | 0.37.1 |
| peft | 0.18.1 |
| timm | 1.0.9 |
| numpy | 1.26.4 |
| pillow | 11.3.0 |

## 3. 测试用例和测试结果

### 3.1 推理测试 (sample.py)

**测试命令**:
```bash
python sample.py --model DiT-XL/2 --image-size 256 --seed 42 --vae-path /home/ma-user/work/temp/sd-vae-ft-mse --num-sampling-steps 50
```

**测试结果**: ✅ 通过

| 指标 | 值 |
|------|------|
| 模型 | DiT-XL/2 |
| 图像尺寸 | 256x256 |
| 采样步数 | 50 |
| 生成图像数 | 8张 |
| 输出文件 | sample.png (832KB) |
| 采样速度 | ~1.83 it/s |

### 3.2 训练测试 (train.py)

**测试命令**:
```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
  --data-path /home/ma-user/work/temp/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini/train \
  --model DiT-XL/2 --image-size 256 --epochs 1 \
  --global-batch-size 2 --vae-path /home/ma-user/work/temp/sd-vae-ft-mse \
  --log-every 10 --ckpt-every 10000 --num-workers 4 --max-steps 100
```

**测试结果**: ✅ 通过

| 指标 | 值 |
|------|------|
| 训练步数 | 100 |
| 初始loss | 0.9292 |
| 最终loss | 0.2425 |
| 训练速度 | ~3.3 steps/sec |
| 模型参数 | 675,129,632 |
| 数据集 | ImageNet-mini (34,745张) |
| AMP | fp16 |

## 4. 性能优化建议

### 4.1 NPU训练优化
1. **增加batch size**: 当前global-batch-size=2，可增大以充分利用NPU算力
2. **多卡分布式训练**: 使用 `--nproc_per_node=N` 启用多卡DDP
3. **数据加载优化**: 增加 `--num-workers` 减少数据加载瓶颈
4. **开启持久化编译**: 首次编译后复用可提速

### 4.2 推理优化
1. **减少采样步数**: 从50步降至25步可提速约2倍
2. **使用DDIM采样**: 比DDPM更快
3. **启用TF32**: 代码已启用

### 4.3 注意事项
- 当前NPU驱动版本23.0.6，存在警告但不影响运行
- Non-finite check和unscale警告可忽略
- Driver Version警告不影响功能

## 5. 无法运行的测试/运维

### 5.1 待验证
| 测试项 | 说明 |
|--------|------|
| 多GPU分布式训练 | 需要多卡环境 |
| 多节点训练 | 需要多台机器 |
| FID评估 | 需要50000张采样和预训练模型 |
| Checkpoint保存/加载 | 需训练更多步验证 |
| 完整epoch训练 | 当前仅测试100步 |

### 5.2 代码问题(警告)
| 问题 | 说明 |
|------|------|
| PIL.Image.BOX/BICUBIC | LSP报错但运行时正常(需改为Image.Resampling) |
| diffusers导入 | LSP报错但运行时正常 |

### 5.3 运维建议
1. 定期监控NPU内存使用: `npu-smi info`
2. 检查训练日志中的警告级别
3. 长期训练建议设置checkpoint保存
