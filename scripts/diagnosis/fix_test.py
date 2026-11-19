"""Fix test for the cuBLAS diagnosis: if the slow FP16 GEMM kernel (chosen for M>=2) is the cause,
then avoiding that kernel for the SAME math must restore speed.
Shape: Qwen2.5-1.5B LM head, W=[151936,1536] FP16. Variants for M = 1, 2, 4, 8, 16:
  base   : torch linear FP16 (what vLLM does)          -> expected slow for M>=2
  rows   : M separate M=1 linears (gemv path)          -> expected ~M x 8.5 ms, much faster than base
  fp32   : x and W in FP32 (no FP16 Tensor-Core kernel) -> expected fast
Also checks results match (max abs diff) so the variants compute the same thing.
usage: python scripts/diagnosis/fix_test.py
"""
import time, torch
from torch.profiler import profile, ProfilerActivity


def timeit(f, iters=10):
    for _ in range(2): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def kernels(f):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    return ",".join(sorted({e.key[:40] for e in p.key_averages() if e.device_time_total > 0}))


torch.manual_seed(0)
W = torch.randn(151936, 1536, device="cuda", dtype=torch.half) * 0.02
W32 = W.float()
lin = torch.nn.functional.linear
print(torch.cuda.get_device_name(0), "torch", torch.__version__)
print(f"{'M':>3} {'base fp16':>10} {'rows(M x gemv)':>15} {'fp32':>8}   maxdiff(rows,fp32 vs base)   base kernel")
for M in (1, 2, 4, 8, 16):
    x = torch.randn(M, 1536, device="cuda", dtype=torch.half)
    x32 = x.float()
    base = lambda: lin(x, W)
    rows = lambda: torch.cat([lin(x[i:i + 1], W) for i in range(M)])
    f32 = lambda: lin(x32, W32)
    tb, tr, tf = timeit(base), timeit(rows), timeit(f32)
    d1 = (rows().float() - base().float()).abs().max().item()
    d2 = (f32() - base().float()).abs().max().item()
    print(f"{M:3d} {tb:10.2f} {tr:15.2f} {tf:8.2f}   {d1:.4f} / {d2:.4f}   {kernels(base)}", flush=True)
