# PyTorch vs MindSpore DiT 性能对比报告

## 测试环境

- **硬件**: Ascend NPU (单卡)
- **PyTorch 版本**: 2.7.1
- **MindSpore 版本**: 2.8.0
- **模型**: DiT-XL/2 (256x256)
- **测试日期**: 2026-04-05

## 推理性能对比

### 测试配置
- 模型: DiT-XL/2
- 图像尺寸: 256x256
- 采样步数: 10
- CFG Scale: 4.0

### 结果对比

| 指标 | PyTorch | MindSpore (PyNative) | 差异 |
|------|---------|-----------|------|
| 平均推理时间 | 76 秒/张 | 82.7 秒/张 | MindSpore 慢 9% |
| 采样速度 | 0.013 images/sec | 0.012 images/sec | PyTorch 快 8% |

## 训练性能对比

### 测试配置
- 模型: DiT-XL/2
- 图像尺寸: 256x256
- Batch Size: 4
- 训练步数: 100

### 结果对比

| 指标 | PyTorch | MindSpore (PyNative) | 差异 |
|------|---------|-----------|------|
| 平均训练时间 | 110 秒/100步 | 156 秒/100步 | MindSpore 慢 42% |
| 训练速度 | 0.91 steps/sec | 0.64 steps/sec | PyTorch 快 42% |
| 每 step 耗时 | 1.1 秒 | 1.56 秒 | MindSpore 慢 42% |
| 训练 Loss | ~0.98 | 1.44 | - |

## MindSpore Graph Mode 测试

### 结果
| 模式 | 推理 | 训练 | 状态 |
|------|------|------|------|
| PyNative | 82.7秒/张 | 1.56秒/step | ✅ 正常 |
| Graph | 编译失败 | 编译失败 | ❌ 不可用 |

### 问题分析
1. 动态控制流与静态图编译不兼容
2. 时序嵌入 (AdaLN) 编译失败
3. 需要修改 mindone 模型代码以适配 graph mode

## 性能分析

### 推理
- PyTorch 和 MindSpore (PyNative) 推理性能接近
- MindSpore 略慢约 9%
- Graph mode 存在编译问题，无法使用

### 训练
- PyTorch 训练速度比 MindSpore 快 42%
- MindSpore 训练 Loss 较高，可能与:
  - 数据集大小不同 (PyTorch: 38668, MindSpore: 5050)
  - 优化器实现差异
  - 训练流程细节不同

## 结论

1. **推理**: PyTorch 和 MindSpore (PyNative) 性能接近，差异约 9%
2. **训练**: PyTorch 明显快于 MindSpore，差异约 42%
3. **Graph Mode**: 当前不可用，存在编译问题
4. **原因分析**:
   - MindSpore 在 NPU 上的优化可能不如 PyTorch 成熟
   - mindone 框架层可能有额外开销
   - Graph mode 需要模型代码适配

## 建议

1. 对于推理，两者都可接受
2. 对于训练，PyTorch 在 NPU 上性能更好
3. Graph mode 需要进一步优化 mindone 模型代码