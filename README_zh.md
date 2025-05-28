# 适用于 DETR 训练的 GPU 批量并行加速的匈牙利算法

本项目提供了一个匈牙利算法的 **CUDA/C++ 高效实现方案**。它能与 PyTorch 无缝集成，专为批量的 300 × N（N ≤ 300）指派问题而优化设计的——这类问题常见于 **DETR** 及相关模型的训练过程中。

相比广泛使用的基于 CPU 的 SciPy `linear_sum_assignment`，本实现方案在随机输入下可实现 **8~160 倍的加速比**，在实际 DETR 训练场景中实现 **3~10 倍的加速比**（测试设备为 **NVIDIA RTX 4090 GPU**）。

---

## 🚀 功能特性

- 支持形状为 [B × 300 × N] 的批量指派问题
  - 每个代价矩阵的最大尺寸可通过编译前修改 CUDA 源码中的 `MAX_ROWS` 和 `MAX_COLS` 进行调整
  - 每个代价矩阵可以有不同尺寸；在堆叠前用 INF 值填充至 N（N ≤ 300）
  - 代价矩阵的数据类型为 `torch.float32`

- 匈牙利算法的 GPU 并行实现包括：
  - 对偶变量初始化
  - 使用 warp 级规约的 Δ 最小值搜索
  - 潜在值更新（u, v, minv）
  - 最终匹配及结果写出

- 在以下方面与 SciPy `linear_sum_assignment` 的进行对比验证：
  - 正确性
  - 运行效率

---

## 📦 环境依赖

- Python 3.x
- PyTorch ≥ 2.0（需支持 `torch.utils.cpp_extension`）
- NVIDIA 驱动 ≥ 535
- CUDA ≥ 12.2（早期版本可能也可运行）
- 支持 C++17 的编译器及 NVCC

> 测试平台：NVIDIA RTX 4090 GPU

---

## ⚙️ 使用方法

**1. 运行示例脚本**

```bash
python hungarian_gpu_batch.py
```

该脚本将：
- 在线编译 CUDA/C++ 扩展
- 使用随机代价矩阵进行测试
- 对比 GPU 和 SciPy CPU 的结果
- 输出运行时间和加速比统计

你将看到每次测试的性能日志，随后是最后 10 次测试的平均统计结果。例如输出：

```log
Mean valid ncols in cost matrix: 203.75
GPU runtime      :    3.85 ms
SciPy CPU runtime:   32.69 ms
Speed-up         :    8.50 x

Mean valid ncols in cost matrix: 164.62
GPU runtime      :    3.23 ms
SciPy CPU runtime:   29.52 ms
Speed-up         :    9.13 x

...
...

Average of the last 10 Loops
GPU runtime      :    1.49 ms
SciPy CPU runtime:   24.67 ms
Speed-up         :   20.67 x
```

**2. 在你自己的代码中使用**

简单示例：

```python
import torch
from hungarian_gpu_batch import hungarian_gpu

cost = torch.rand((16, 300, 300), device="cuda", dtype=torch.float32)  # 批量代价矩阵
Ns = torch.randint(1, 301, (16,), device="cuda", dtype=torch.int32)    # 批量实际任务数，每个 Ni 对应一个代价矩阵（超过 Ni 的条目将被忽略）

output = hungarian_gpu(cost, Ns)
```

---

## 📊 性能表现

**1. 随机测试**

我们展示了在不同批量大小 (B) 下的加速曲线，输入为随机填充的 [B × 300 × 300] 代价矩阵。实验均在单张 NVIDIA RTX 4090 GPU 上进行。

> 对比基准: SciPy `linear_sum_assignment`

<img src="https://github.com/linfeng93/HA4DETR/blob/main/speedup.png" style="width:70%; height:auto;">

**2. 实际应用**

我们将该 GPU 版匈牙利算法集成至 DETR 训练流程，替换原有 SciPy 的 `linear_sum_assignment`。

在每张 GPU 上的 batch size 为 16（每张 GPU 处理自己的样本）时，该实现带来了 **3 倍加速**，整体训练开销减少了约 **10%**。随着 batch size 的增大，加速效果会更为显著，这与随机测试中观察到的趋势一致。

---

## 📜 许可证

本仓库基于 [Apache-2.0 协议](https://github.com/linfeng93/HA4DETR/blob/main/LICENSE) 进行授权。

---

## 📚 引用方式

如果你觉得这个仓库有用，请使用以下 BibTeX 进行引用：

```bibtex
@misc{ha4detr,
    title = {GPU-Accelerated Batched Hungarian Algorithm for DETR},
    author = {Feng Lin, Xiaotian Yu, Rong Xiao},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/linfeng93/HA4DETR},
}
```

---

## 💼 鸣谢

本开源仓库起源于 **Intellifusion Inc.** 的视频目标检测项目，由 Feng Lin、Xiaotian Yu 和 Rong Xiao 共同开发。

---

## 📮 联系方式

欢迎通过 issue 或 PR 参与讨论、提出建议或提问。
