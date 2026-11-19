# Inference Framework Evaluation on 4 GB VRAM — Lab Notes

Study: throughput and memory footprint of llama.cpp vs vLLM, 0.8B–2B models on a GTX 1650 (4 GB):
three 4-bit quantised models (1.5B–2B) plus one unquantised FP16 baseline (Qwen3.5-0.8B).

## Hardware
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Mobile (TU117M, PCI 0x1F9D), Turing |
| Compute capability | 7.5 (sm_75) |
| CUDA cores / Tensor cores | 1024 / 0 |
| VRAM | 4 GB GDDR6, 128-bit, ~192 GB/s |
| Peak FP16 / FP32 | ~7.3 / ~3.7 TFLOPS (at 1785 MHz max boost) |
| BF16 | not supported -> use --dtype half |
| Power limit | 50 W max (observed 30 W cap at one point -> keep power mode fixed) |
| CPU | Intel i5-11400H, 6C/12T |
| RAM | 7.6 GB (WSL limited to 6 GB + 12 GB swap on D:) |

## Software
| Item | Value |
|---|---|
| Windows | 11 Home 10.0.26200 |
| NVIDIA driver | 581.83 (CUDA 13.0 capable) |
| WSL | 2.6.3, kernel 6.6.87.2-1 |
| Distro | Ubuntu 22.04.5 LTS (moved to D:\WSL\Ubuntu-22.04 on 2026-11-17) |
| Python | 3.10.12 (venv: ~/infer-eval/.venv-vllm) |
| gcc / cmake | system gcc, cmake 3.22.1 |
| vLLM | 0.31.0 |
| torch | 2.13.0+cu130 (arch list includes sm_75) |
| transformers / triton | 5.17.0 / 3.7.1 |
| CUDA toolkit (nvcc) | 12.8 (apt cuda-toolkit-12-8, for llama.cpp build; vLLM uses its own pip CUDA 13.0 runtime) |
| llama.cpp commit | d3c3e24 (2026-11-16), built with CUDA 12.8, CMAKE_CUDA_ARCHITECTURES=75 |

## Models (updated 2026-11-17: 4 models)
| # | Model | Precision | llama.cpp (GGUF) | vLLM |
|---|---|---|---|---|
| 1 | Qwen3.5-0.8B | FP16 (unquantised baseline) | F16 GGUF converted locally from Qwen/Qwen3.5-0.8B (convert_hf_to_gguf.py) | Qwen/Qwen3.5-0.8B, --dtype half, --language-model-only |
| 2 | Qwen2.5-1.5B-Instruct | 4-bit | Qwen/Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M) | Qwen/Qwen2.5-1.5B-Instruct-AWQ |
| 3 | Qwen3-1.7B | 4-bit | bartowski/Qwen_Qwen3-1.7B-GGUF (Q4_K_M) | JunHowie/Qwen3-1.7B-GPTQ-Int4 |
| 4 | MiniCPM5-2B | 4-bit | openbmb/MiniCPM5-2B-GGUF (Q4_K_M) | openbmb/MiniCPM5-2B-GPTQ (AWQ-packed) |

Qwen3.5-0.8B notes: hybrid architecture (Gated DeltaNet linear attention + full attention),
shipped as a vision-language model (Qwen3_5ForConditionalGeneration); only the text path is benchmarked.
Weights are BF16 on the Hub -> cast to FP16 (Turing has no BF16). Same source weights for both frameworks.
Earlier it was dropped over sm_75 kernel risk; re-added as the FP16 reference point — results below decide.

## Decisions
- 4-bit for models 2–4 (no quantisation sweep); model 1 (Qwen3.5-0.8B) runs unquantised FP16 as a baseline.
- Models/HF cache live inside WSL ext4 at ~/infer-eval/models (HF_HOME set in ~/infer-eval/env.sh), not /mnt/d (9P is slow and would distort load times).
- vLLM scripts need an `if __name__ == "__main__":` guard (WSL forces the spawn start method).
- No Docker: native install in WSL; reproducibility via pinned requirements + commit hashes.
- Fixed context 2048 for every model/framework; all layers on GPU (-ngl 99).
- --dtype half everywhere (no BF16 on Turing).
- Qwen3 thinking mode disabled during benchmarks.

## Log
- 2026-11-17: Moved Ubuntu WSL C: -> D: (C: free 2.2 GB -> 13.6 GB). .wslconfig: memory=6GB, swap=12GB, swapFile on D:.
- 2026-11-17: Created ~/infer-eval/{models,scripts,results/raw,results/plots,logs}; vLLM venv created; vLLM install started.
- 2026-11-17: vLLM 0.31.0 installed OK; torch sees GTX 1650, FP16 matmul OK.
- 2026-11-17: Cleanup: removed old ~/llm-inference venv + pip cache (~10 GB). HF_HOME -> ~/infer-eval/models/hf via env.sh.
- 2026-11-17: Plan change: added Qwen3.5-0.8B unquantised FP16 as 4th model (baseline).
- 2026-11-17: vLLM smoke test Qwen3.5-0.8B FP16 (scripts/vllm_smoke.py, logs/vllm_smoke_qwen3.5-0.8b_fp16*.log):
  - Run 1 (default: torch.compile + CUDA graphs, max_num_batched_tokens=8192): FAILED, CUDA OOM in profile_cudagraph_memory.
  - Run 2 (--enforce-eager, max_num_batched_tokens=2048, max_num_seqs=32, gpu_mem_util=0.80): WORKS. Output coherent in FP16 (17*23=391 correct, no NaN).
    - Attention backend: TRITON_ATTN (FlashAttention-2 needs sm_80; FlashInfer sampler unavailable on sm_75 -> fallback).
    - Weights 1.53 GiB; KV cache 1.07 GiB (35,020 tokens, 17x concurrency at 2048 ctx); total VRAM ~3.46 GiB of 4 GiB (baseline 0.41 GiB).
    - Engine init 414 s (first-run Triton JIT of Gated DeltaNet kernels; should be cached on later runs).
    - Rough speed (NOT a benchmark, no proper warm-up): ~14.6 tok/s single stream, eager mode.
  - TODO: try CUDA graphs with smaller capture sizes (cudagraph_capture_sizes up to ~16) to recover decode speed.
- 2026-11-17: Cloned llama.cpp @ d3c3e24 (2026-11-16, has qwen35 arch). CUDA toolkit 12.8 installed (apt, NVIDIA wsl-ubuntu repo).
- 2026-11-17: Downloaded all models (scripts/download_models.sh): HF cache 6.5 GB, GGUF 5.2 GB.
  Qwen/Qwen3-1.7B-GGUF has only Q8_0 -> using bartowski Q4_K_M. MiniCPM5-2B-GPTQ is AWQ-packed (vLLM loads it as awq).
- 2026-11-17: Converted Qwen3.5-0.8B -> models/gguf/Qwen3.5-0.8B-F16.gguf (1.56 GB, 299 tensors, text only) with
  llama.cpp/convert_hf_to_gguf.py in a separate CPU venv .venv-tools (torch 2.11 cpu, transformers 4.57.6).
- 2026-11-17: vLLM smoke tests, 4-bit models (eager, gpu_util 0.80, 2048 batched tokens) — all OK, coherent output:
  | Model | quant (auto) | linear kernel | attn | load s | VRAM after load | ~single tok/s |
  |---|---|---|---|---|---|---|
  | Qwen2.5-1.5B-AWQ | awq | Marlin | TRITON_ATTN | 165 | 3485 MiB | 21.2 |
  | Qwen3-1.7B-GPTQ | gptq | Marlin | TRITON_ATTN | 93 | 3509 MiB | 18.0 |
  | MiniCPM5-2B (AWQ) | awq | Marlin | TRITON_ATTN | 91 | 3522 MiB | 14.0 |
  Marlin 4-bit kernels DO run on sm_75 in vLLM 0.31 (expected fallback, did not happen).
  vLLM VRAM ~3.5 GB for every model = pre-allocation (gpu_memory_utilization), NOT model need ->
  benchmark vLLM twice: (a) default gpu_util 0.80, (b) fixed minimal KV cache (--kv-cache-gb) for a fair footprint.
- 2026-11-17: WSL became unresponsive for ~1 min twice during vLLM runs (host RAM pressure: 7.6 GB total, WSL 6 GB).
  Rule: run benchmarks strictly one at a time, close other apps; consider memory=5GB in .wslconfig if it recurs.
- 2026-11-17: Benchmark scripts written: gpu_sampler.sh, bench_vllm.py/.sh, bench_llamacpp.sh, build_llamacpp.sh, run_all.sh.
- 2026-11-17: llama.cpp built OK. Smoke (llama-bench pp128/tg32, 1 rep, all layers on GPU) — all 4 GGUFs run:
  Qwen3.5-0.8B F16 274/34 t/s; Qwen2.5-1.5B Q4_K_M 344/49; Qwen3-1.7B Q4_K_M 302/44; MiniCPM5-2B Q4_K_M 234/34.
  (llama-cli -no-cnv printed nothing in this build — text sanity for GGUFs still TODO, e.g. via llama-server.)
- 2026-11-17: Started run_all.sh phase A (llama.cpp full grid). Phase A done 13:47. Phase B (vLLM default, CUDA graphs ok) done 16:02.
- 2026-11-17: Phase C first attempt failed for all models: with kv_cache_memory_bytes set, vLLM still checks the default
  gpu_memory_utilization=0.92 (> free 3.2 GiB) at startup. Fixed in bench_vllm.py (always pass 0.80). Phase C rerun started 16:31.
- 2026-11-17: DIAGNOSIS of vLLM decode slowdown at >=2 sequences (step 49 ms @1 -> ~145 ms @2..8, Qwen2.5-AWQ):
  1. min_tokens on/off: identical (49.0/145.1 vs 50.5/146.7 ms/step) -> not the benchmark method.
  2. vllm bench latency (official tool), in 128 / out 128: bs1 6.89 s, bs2 19.76 s; own script 6.82 / 19.66 s -> method validated.
  3. torch profiler (logs/profile, scripts/analyze_trace.py): at bs2 a new kernel volta_fp16_s884gemm_fp16_64x64 appears,
     ~98 ms per call, 1 call per step = the LM head (FP16, unquantised). At bs1 the same matmul uses cuBLAS gemv2T (3.4 ms).
     Marlin 4-bit kernels only +32% bs1->bs2 (normal).
  4. Pure PyTorch microbenchmark (scripts/gemm_microbench.py, logs/gemm_microbench.log): ANY FP16 linear with M>=2 rows
     selects volta_fp16_s884gemm (Tensor-Core HMMA kernel) on the GTX 1650 and is 10-30x slower than M=1:
     Qwen2.5 lm_head 8.5 ms (M=1) -> 92 ms (M=2..16); generic 8960x1536 layer 0.19 -> 5.7 ms. cuBLASLt picks the same kernel.
  ROOT CAUSE: GTX 1650 (TU117) has no Tensor Cores, but reports sm_75 like RTX 20xx; cuBLAS heuristics choose a
  Tensor-Core FP16 GEMM for M>=2, which runs pathologically slowly on this chip. vLLM runs FP16 matmuls through
  PyTorch/cuBLAS (LM head always; every layer for unquantised Qwen3.5 -> 337 ms/step). llama.cpp uses its own CUDA
  kernels for small batches, so it does not hit this. Benchmarks reflect real out-of-the-box behaviour; keep them.
- 2026-11-17: Phase C (vLLM, KV cache matched to llama.cpp x1.1) done 18:28, all 40 cells OK. Footprints (MiB):
  Qwen3.5 2699, Qwen2.5 2591, Qwen3 3095, MiniCPM5 2991 (default: 3091/3212/2912/3533). Throughput identical to default
  -> KV size is not the cause of the slowdown. Qwen3 matched-KV > default because llama.cpp's KV for Qwen3 (1344 MiB,
  112 KB/token x 12288 ctx) is bigger than vLLM's default auto-sized KV (11,504 tokens).
- 2026-11-17: FIX TEST (scripts/fix_test.py, logs/fix_test.log), Qwen2.5 LM head [151936x1536]:
  M=2: FP16 cuBLAS (s884gemm) 91.0 ms; same math as 2x gemv 23.1 ms; FP32 10.9 ms (results equal, maxdiff 0.002).
  FP32 is normally the SLOWER path, yet 8x faster -> the FP16 s884gemm kernel choice is confirmed as the cause.
- 2026-11-17: analyze.py run -> results/summary.md, summary.csv, all_rows.csv, plots/*.png.
- 2026-11-17: VERIFICATION (scripts/verify_results.py -> logs/verify_results.log):
  - 12 result files x 10 cells, full grid, no NaN/zero, tg=128 everywhere; all vLLM runs used CUDA graphs (no eager fallback).
  - speed = pl*tokens/time recomputed from raw timings: matches stored values (<0.1% error).
  - 'FAILED' lines in run_all_BC.log = first Phase C attempt (gpu_memory_utilization bug); those files were deleted and rerun OK.
  - llama-batched-bench pl=1 tg vs llama-bench tg128: batched 4-15% higher (32.4/31.2, 42.7/37.5, 37.7/33.5, 29.5/25.6);
    pp512 within 5%. -> report each metric from ONE tool only (batched-bench); run-to-run/tool variation ~+-10-15%.
  - Power cap identical for both frameworks (median ~29 W, all runs). SM clock median: llama.cpp ~1410 MHz, vLLM ~1575-1590 MHz
    (vLLM does less work per cycle, so it fits more clock under the same power cap) -> vLLM was NOT disadvantaged by clocks.
    Max temp: llama.cpp 77-81 C, vLLM 83-84 C (longer runs).
  - Text sanity (logs/gguf_text_check.log): all 4 GGUFs give coherent answers via llama-server.
    Qwen3.5 converted GGUF vs vLLM HF weights (logs/qwen35_compare.log, qwen35_compare_vllm.log): IDENTICAL greedy outputs
    on 5 prompts (incl. same wrong '501' for 17*23) -> conversion is correct; the error is the 0.8B model itself.
- 2026-11-18: REORGANISED project (see README.md): results/{throughput/rep1, diagnosis, checks, final}; logs/ and
  scripts/ mirror it; scripts/common.sh holds the model list. Final tables/charts regenerated into results/final/.
- 2026-11-18: Extra tests (3x repeats, perplexity, server latency, long prompts, cuBLAS workaround) were prepared and
  started, then CANCELLED by decision (laptop heat + disk filled up). All their scripts, partial outputs and downloads
  were deleted; nothing from them is used. The paper is based on the first benchmark (rep1) only.
  Lesson recorded: under WSL the NVIDIA driver can silently spill VRAM overflow into system RAM, and the laptop GPU
  throttles hard when hot (85-87 C, 300 MHz) -> any future runs need a cool-down between tests.
