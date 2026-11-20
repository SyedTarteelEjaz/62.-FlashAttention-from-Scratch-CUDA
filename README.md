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

## Results (Tesla T4, Google Colab)

Batch 2, 8 heads, head size 64, float32, no mask. Time is the average of 20 calls. "Extra memory" is the peak GPU memory used on top of the inputs. Maximum difference to PyTorch's result is under 1e-6 at every size (marked `OK` by the benchmark).

| N | Naive ms | PyTorch SDPA ms | **Ours ms** | Naive extra MB | SDPA extra MB | **Ours extra MB** | Max error |
|---|---|---|---|---|---|---|---|
| 512 | 0.705 | 0.813 | **1.094** | 34 | 2 | **2** | 3.6e-07 |
| 1024 | 2.962 | 2.069 | **2.899** | 132 | 4 | **4** | 3.6e-07 |
| 2048 | 9.402 | 5.716 | **6.376** | 520 | 8 | **8** | 3.5e-07 |
| 4096 | 37.124 | 21.436 | **23.011** | 2,064 | 16 | **16** | 6.6e-07 |

### What the numbers show

**Memory.** Naive attention uses memory that grows with the square of the sequence length, from 34 MB at N=512 to 2,064 MB at N=4096. This kernel uses only the output size (2 MB to 16 MB), so it needs **17x less memory at N=512 and 129x less at N=4096**, and the gap keeps widening as N grows.

**Speed against naive attention.** At short sequences the naive version is faster (1.55x at N=512), because the whole score matrix is small and PyTorch's matrix multiplies are very efficient. The two are about equal at N=1024. From N=2048 on, this kernel is faster: **1.47x at N=2048 and 1.61x at N=4096**.

**Speed against PyTorch SDPA.** PyTorch's built-in attention is faster at every size. This kernel runs at about 74%, 71%, 90% and 93% of SDPA speed for N = 512, 1024, 2048 and 4096, so it gets closer as sequences get longer.

**Memory against PyTorch SDPA.** The extra memory is identical (2, 4, 8, 16 MB), because SDPA also uses a fused attention kernel that never builds the full score matrix. The memory saving in this project is therefore against *naive* attention, not against SDPA.

Results come from a single Colab session on one T4, so treat small differences as approximate. N = 8192 was not part of this run.

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

- This is an educational **FP32** kernel with no Tensor Cores and no heavy tuning. PyTorch's built-in kernels are highly optimised, so the goal here is **correctness and memory scaling**. It matches the numerical result of PyTorch and reaches about 71-93% of SDPA speed in the measured range, but it does not beat it.
- Forward pass only (no backward pass).
- Supported: `float32`, head size 32 / 64 / 128, shape `(B, H, N, D)`. Head size 128 needs more registers per thread than the GPU offers, so it likely spills to slower memory and runs slower than 32 and 64.
- Results above are from a Tesla T4 on Google Colab, non-causal, single run. The causal mask is implemented but its timings are not reported here.

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
GitHub: [SyedTarteelEjaz](https://github.com/SyedTarteelEjaz)
