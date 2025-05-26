# GPU 批处理匈牙利算法加速器

本项目是一个基于 **CUDA** 和 **PyTorch C++ 扩展机制** 的高性能 GPU 实现，用于求解最大 300×300 的指派问题（Hungarian Algorithm）。支持批处理输入，运行速度远超 CPU 实现，在 NVIDIA RTX 4090 上可获得超过 30 倍的加速效果。

---

## 🚀 功能特点

- 支持最大 300×300 的代价矩阵
- 支持批量处理（batched assignment）
- CUDA 内核中并行处理包括：
  - 初始潜力初始化
  - 最小增量搜索（Δ min-search）
  - 潜力值 u/v/minv 更新
  - 匹配路径更新与结果写出
- 自动与 CPU 版本结果比对，确保准确性
- 随机数据与真实数据双测试流程

---

## 📦 环境要求

- CUDA ≥ 12.2
- NVIDIA 驱动版本 ≥ 535
- Python ≥ 3.8
- PyTorch ≥ 2.1（需支持 `torch.utils.cpp_extension`）
- SciPy ≥ 1.10
- 支持 C++17 的编译器 + NVCC

> 推荐硬件环境：NVIDIA RTX 4090

---

## ⚙️ 使用方法

无需安装，通过命令直接运行主程序即可：

```bash
python hungarian_gpu_batch_human.py
```

执行内容包括：
- 自动编译 CUDA/C++ 扩展
- 对随机和实际 cost 矩阵执行匈牙利匹配
- 与 SciPy 的 CPU 实现结果对比，验证正确性
- 输出 GPU 加速效果统计信息

---

## 📁 输入数据格式（真实数据）

- 输入为 `.npy` 文件，形状为 `(B, 300, ≤300)` 的三维张量。
- 内部自动处理 padding，超过 1e5 的值将视为 ∞（不可匹配列）。

---

## 📊 性能示例

RTX 4090，batch size = 16，矩阵大小 300×300：

```
平均有效列数       : 276.19
GPU 匹配时间        : 2.13 ms
SciPy CPU 匹配时间 : 60.72 ms
加速比             : 28.52 倍
```

---

## 📜 许可证

本项目采用 MIT License。

---

## 📮 联系作者

如有问题或建议，欢迎通过 GitHub Issue 提出或 PR 交流。
