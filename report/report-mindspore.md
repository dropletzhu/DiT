# MindSpore DiT Performance Report

## 测试环境

- **硬件**: Ascend NPU (单卡)
- **MindSpore 版本**: 2.8.0
- **mindone 版本**: 最新
- **模型**: DiT-XL/2 (256x256)
- **VAE**: sd-vae-ft-mse
- **测试日期**: 2026-04-05

## 推理性能

### 测试配置
- 模型: DiT-XL/2
- 图像尺寸: 256x256
- 采样步数: 10
- CFG Scale: 4.0
- Seed: 42

### 测试结果 (PyNative Mode, 3次运行)

| Run | 总时间 (秒) | 平均时间 (秒) |
|-----|------------|--------------|
| 1   | 81         |               |
| 2   | 84         |               |
| 3   | 83         | 82.7         |

### 性能分析
- 平均推理时间: 82.7 秒/张
- 采样速度: 约 0.012 images/sec

## 训练性能

### 测试配置
- 模型: DiT-XL/2
- 图像尺寸: 256x256
- Batch Size: 4
- 训练步数: 100
- 数据集: ImageNet-mini (5,050 images)
- AMP: 关闭
- 执行模式: PyNative

### 测试结果 (PyNative Mode, 3次运行)

| Run | 总时间 (秒) | 平均时间 (秒) | 平均 Loss |
|-----|------------|--------------|-----------|
| 1   | 150        |               |           |
| 2   | 159        |               |           |
| 3   | 159        | 156           | 1.44      |

### 性能分析
- 平均训练时间: 156 秒/100 steps
- 平均训练速度: 0.64 steps/sec
- 每 step 耗时: 1.56 秒

## Graph Mode 测试

### 结果
- **推理**: 编译失败 (RuntimeError: compile graph kernel_graph0 failed)
- **训练**: 编译失败 (RuntimeError: compile graph kernel_graph0 failed)

### 错误分析
Graph mode 在当前 mindone 版本上存在兼容性问题:
1. 动态控制流 (如 if/for) 与静态图编译不兼容
2. 时序嵌入 (timestep embedding) 编译失败
3. 需要修改模型代码以适配 graph mode

### PyNative vs Graph 模式对比

| 模式 | 推理性能 | 训练性能 | 可用性 |
|------|---------|---------|--------|
| PyNative | 82.7秒/张 | 1.56秒/step | ✅ 可用 |
| Graph | 编译失败 | 编译失败 | ❌ 不可用 |

## 总结

MindSpore 在 Ascend NPU 上运行 DiT-XL/2:
- 推理: 约 83 秒/张 (PyNative)
- 训练: 约 1.56 秒/step (PyNative)
- Graph mode 当前不可用，需进一步优化