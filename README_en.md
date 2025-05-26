# GPU-Batched Hungarian Algorithm Accelerator

This repository provides a **CUDA + PyTorch C++ extension** implementation of the **Hungarian Algorithm**, optimized for solving batched 300×300 assignment problems efficiently on GPU. It achieves up to **30× speed-up** over CPU (SciPy) versions on NVIDIA RTX 4090.

---

## 🚀 Features

- Supports up to 300×300 cost matrices
- Handles batched assignment problems (B × 300 × 300)
- Parallelization includes:
  - Initialization of dual variables
  - Δ-minimum search with warp-level reduction
  - Potentials update (u/v/minv)
  - Final assignment and write-out
- Fully validated against SciPy implementation
- Includes tests with both synthetic and real datasets

---

## 📦 Requirements

- CUDA ≥ 12.2
- NVIDIA driver ≥ 535
- Python ≥ 3.8
- PyTorch ≥ 2.1 (with `torch.utils.cpp_extension`)
- SciPy ≥ 1.10
- C++17-capable compiler and NVCC

> Recommended GPU: NVIDIA RTX 4090

---

## ⚙️ How to Run

No installation required. Just run the main Python script:

```bash
python hungarian_gpu_batch_human.py
```

This will:
- Compile the CUDA/C++ extension inline
- Run tests with random and real cost matrices
- Compare GPU results with SciPy CPU results
- Report speed-up statistics

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
