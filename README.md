# FlashAttention from Scratch (CUDA)

A from-scratch CUDA implementation of the **FlashAttention forward pass**: tiled attention with **online softmax**, so the full `N x N` score matrix is never stored. It is benchmarked against PyTorch attention for **speed** and **GPU memory**.

## Why this matters

Standard attention computes `softmax(Q Kᵀ / √d) V` by building an `N x N` score matrix. Memory grows with the **square** of the sequence length, which is what runs a GPU out of memory on long sequences. FlashAttention computes the exact same result in small tiles, so the extra memory grows only with `N`.

## How it works

1. **Tiling** — each block owns a group of query rows and walks over K and V in small tiles held in fast shared memory.
2. **Online softmax** — for each query row the kernel keeps a running maximum `m` and running sum `l`. When a new tile arrives it rescales the partial output by `exp(m_old − m_new)`. The final answer is exact, not an approximation.
3. **No score matrix** — only the output `O` is allocated, so extra memory is `B·H·N·D·4` bytes.
4. **Causal mask** (optional) — keys after the query are masked, and tiles beyond the diagonal are skipped.

Mapping: one thread owns one query row (its `q` and output accumulators stay in registers). Block sizes are chosen per head size.

| Head size D | Threads per block | K/V tile rows | Shared memory |
|---|---|---|---|
| 32 | 128 | 32 | 8 KB |
| 64 | 128 | 32 | 16 KB |
| 128 | 64 | 16 | 16 KB |

## Memory: what to expect

The score matrix alone, for batch 2 and 8 heads in float32, is `2·8·N²·4` bytes. Our kernel only allocates the output, `2·8·N·64·4` bytes (head size 64).

| N | Naive attention: one N×N matrix | This kernel: output only |
|---|---|---|
| 1024 | 64 MiB | 4 MiB |
| 2048 | 256 MiB | 8 MiB |
| 4096 | 1,024 MiB | 16 MiB |
| 8192 | 4,096 MiB | 32 MiB |

Naive attention creates several matrices of this size (scores, masked scores, softmax output), so its real peak is higher. The benchmark measures the actual peak for all three methods.

## Run it (Google Colab, no setup)

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pick `flash_attention_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.**

The first run compiles the kernel (1–2 minutes). The benchmark then prints a table and saves `results.png`.

Or from any machine with PyTorch and CUDA:

```bash
python benchmark.py                 # default sizes 512 ... 8192
python benchmark.py --causal        # causal mask
python benchmark.py --dim 128 --sizes 1024 2048
```

## What the benchmark reports

For each sequence length it prints time per call and extra GPU memory for three methods:

- **naive** — textbook attention (builds the full score matrix)
- **SDPA** — PyTorch's `scaled_dot_product_attention`
- **ours** — this CUDA kernel

It also prints the maximum difference to PyTorch's result and marks it `OK` when below `1e-3`. If a method runs out of GPU memory it prints `OOM`.

## Honest notes

- This is an educational **FP32** kernel with no Tensor Cores and no heavy tuning. PyTorch's built-in kernels are highly optimised, so the goal here is **correctness and memory scaling**, not beating them on speed.
- Forward pass only (no backward pass).
- Supported: `float32`, head size 32 / 64 / 128, shape `(B, H, N, D)`. Head size 128 needs more registers per thread than the GPU offers, so it likely spills to slower memory and runs slower than 32 and 64.
- Tested on a Tesla T4 (Google Colab).

## Files

| File | Purpose |
|---|---|
| `flash.cu` | The CUDA kernel and its host launcher |
| `main.cpp` | PyTorch (pybind11) binding |
| `benchmark.py` | Correctness check, timing and memory benchmark, chart |
| `flash_attention_colab.ipynb` | Ready-to-run Colab notebook |

## Possible improvements

- Use Tensor Cores (WMMA) for the `QKᵀ` and `PV` products
- FP16 / BF16 inputs
- Split the K/V loop across warps for better occupancy at small batch sizes
- Backward pass

## Author

Syed Tarteel Ejaz — Lahore, Pakistan
Email: syedtarteelejaz@gmail.com
GitHub: [19YearsOlds](https://github.com/19YearsOlds)
