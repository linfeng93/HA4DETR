# GPU-Accelerated Batched Hungarian Algorithm for DETR

This repository offers an efficient **CUDA + PyTorch C++ extension** implementation of the **Hungarian Algorithm** on GPU, optimized for batched 300×N (N ≤ 300) assignment problems—commonly encountered during the training of **DETR** and related models.

This implementation achieves a **8~150× speedup** on random inputs and a **3~10×** speedup in practical DETR training scenarios, compared to the CPU-based `linear_sum_assignment` from SciPy, tested on **a single NVIDIA RTX 4090**.

---

## 🚀 Features

- Supports batched assignment problems of shape (B × 300 × N)
  - The maximum size of each cost matrix can be adjusted by updating `MAX_ROWS` and `MAX_COLS` in the CUDA source code before compiling
  - Each cost matrix can have a different size; pad to size N with INF values before stacking (N ≤ 300)
  - Uses `torch.float32` as the data type for cost matrices

- GPU parallelization of the Hungarian Algorithm includes:
  - Initialization of dual variables
  - Δ-minimum search with warp-level reduction
  - Potentials update (u, v, minv)
  - Final assignment and write-out

- Fully validated against SciPy's `linear_sum_assignment` in terms of:
  - **Correctness**
  - **Runtime performance**

---

## 📦 Requirements

- Python 3.x
- PyTorch ≥ 2.0 (with `torch.utils.cpp_extension`)
- NVIDIA driver ≥ 535
- CUDA ≥ 12.2 (earlier versions may work)
- C++17-compatible compiler and NVCC
  
> Tested on: NVIDIA RTX 4090 GPU

---

## ⚙️ How to Use

**1. Run the demo script**

```bash
python hungarian_gpu_batch.py
```

This will:
- Compile the CUDA/C++ extension inline
- Run tests using random cost matrices
- Compare GPU results with SciPy CPU results
- Report runtime speed-up statistics

**2. Use in your own code:**

A simple example:

```python
import torch
from hungarian_gpu_batch import hungarian_gpu

cost = torch.rand((16, 300, 300), device="cuda", dtype=torch.float32)  # Batched cost matrices
Ns = torch.randint(1, 301, (16,), device="cuda", dtype=torch.int32)    # Batched actual task numbers, each Ni corresponds to one cost matrix (entries beyond Ni are ignored)

output = hungarian_gpu(cost, Ns)
```

---

## 📁 Input Format for Real Data

- Accepts `.npy` files containing cost tensors of shape `(B, 300, ≤300)`
- Values > 1e5 are treated as ∞ and used for padding invalid columns

---

## 📊 Performance Example

Tested on RTX 4090 with batch size = 16 and 300×300 cost matrices:

```
Average valid cols : 276.19
GPU runtime         : 2.13 ms
SciPy CPU runtime   : 60.72 ms
Speed-up            : 28.52×
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 📮 Contact

Feel free to open an issue or PR for discussions, improvements, or questions.
