"""Benchmark: our CUDA FlashAttention forward vs PyTorch attention (time + extra memory)."""
import argparse, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument("--causal", action="store_true", help="use a causal mask")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--dim", type=int, default=64, choices=[32, 64, 128])
p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
print("Compiling CUDA extension (first time takes ~1-2 minutes)...")
flash = load(name="flash_ext", sources=[os.path.join(here, "main.cpp"), os.path.join(here, "flash.cu")],
             extra_cuda_cflags=["-O3"], verbose=False)
dev = "cuda"
print("GPU:", torch.cuda.get_device_name(0))
print(f"B={args.batch} H={args.heads} D={args.dim} causal={args.causal} dtype=float32\n")


def naive_attention(q, k, v, causal):
    # the textbook version: builds the full N x N score matrix
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def ours(q, k, v, causal):
    return flash.forward(q, k, v, causal)


def measure(fn, q, k, v, causal):
    """returns (ms per call, extra peak memory in MB) or (None, None) if out of memory"""
    try:
        for _ in range(2):
            fn(q, k, v, causal)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn(q, k, v, causal); torch.cuda.synchronize()
        peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            fn(q, k, v, causal)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / args.iters, peak_mb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def fmt(x, spec):
    return "   OOM" if x is None else format(x, spec)


rows = []
hdr = f"{'N':>6} | {'naive ms':>9} {'SDPA ms':>9} {'ours ms':>9} | {'naive MB':>9} {'SDPA MB':>9} {'ours MB':>9} | {'max err':>9}"
print(hdr); print("-" * len(hdr))
torch.manual_seed(0)
for n in args.sizes:
    q, k, v = [torch.randn(args.batch, args.heads, n, args.dim, device=dev) for _ in range(3)]
    t_n, m_n = measure(naive_attention, q, k, v, args.causal)
    t_s, m_s = measure(sdpa, q, k, v, args.causal)
    t_o, m_o = measure(ours, q, k, v, args.causal)
    # correctness: compare to SDPA (always fits) in float32
    err = (ours(q, k, v, args.causal) - sdpa(q, k, v, args.causal)).abs().max().item()
    print(f"{n:>6} | {fmt(t_n,'9.3f')} {fmt(t_s,'9.3f')} {fmt(t_o,'9.3f')} | "
          f"{fmt(m_n,'9.1f')} {fmt(m_s,'9.1f')} {fmt(m_o,'9.1f')} | {err:9.2e} {'OK' if err < 1e-3 else 'MISMATCH'}")
    rows.append((n, t_n, t_s, t_o, m_n, m_s, m_o))
    del q, k, v; torch.cuda.empty_cache()

# chart
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = [r[0] for r in rows]
    for idx, name in [(1, "naive PyTorch"), (2, "PyTorch SDPA"), (3, "ours (CUDA)")]:
        pts = [(r[0], r[idx]) for r in rows if r[idx] is not None]
        ax[0].plot(*zip(*pts), marker="o", label=name)
    for idx, name in [(4, "naive PyTorch"), (5, "PyTorch SDPA"), (6, "ours (CUDA)")]:
        pts = [(r[0], r[idx]) for r in rows if r[idx] is not None]
        ax[1].plot(*zip(*pts), marker="o", label=name)
    ax[0].set_title("Time per call (ms)"); ax[1].set_title("Extra GPU memory (MB)")
    for a in ax:
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("sequence length N"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("results.png", dpi=150)
    print("\nSaved chart: results.png")
except Exception as ex:
    print("chart skipped:", ex)
