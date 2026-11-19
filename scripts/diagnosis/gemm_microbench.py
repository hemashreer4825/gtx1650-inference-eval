"""Isolate the LM-head matmul outside vLLM: y = x @ W^T, FP16, W = [vocab, hidden].
Times batch M = 1..16 with cuBLAS (default) and cuBLASLt, and shows which kernel each picks.
usage: python scripts/diagnosis/gemm_microbench.py
"""
import torch, time
from torch.profiler import profile, ProfilerActivity

SHAPES = {"Qwen2.5-1.5B lm_head": (151936, 1536), "Qwen3.5-0.8B lm_head": (248320, 1024),
          "Qwen2.5 MLP up (FP16, if unquantised)": (8960, 1536)}


def bench(W, M, iters=20):
    x = torch.randn(M, W.shape[1], device="cuda", dtype=torch.half)
    for _ in range(3): torch.nn.functional.linear(x, W)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): torch.nn.functional.linear(x, W)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def kernel(W, M):
    x = torch.randn(M, W.shape[1], device="cuda", dtype=torch.half)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        torch.nn.functional.linear(x, W); torch.cuda.synchronize()
    ks = [e.key for e in p.key_averages() if e.device_time_total > 0]
    return ks[0][:60] if ks else "?"


print(torch.cuda.get_device_name(0), "torch", torch.__version__)
for name, (n, k) in SHAPES.items():
    W = torch.randn(n, k, device="cuda", dtype=torch.half) * 0.02
    print(f"\n== {name}  W=[{n},{k}]")
    for lib in ("cublas", "cublaslt"):
        torch.backends.cuda.preferred_blas_library(lib)
        for M in (1, 2, 4, 8, 16):
            print(f"  {lib:8s} M={M:2d}  {bench(W, M):8.2f} ms   {kernel(W, M)}")
    del W; torch.cuda.empty_cache()
