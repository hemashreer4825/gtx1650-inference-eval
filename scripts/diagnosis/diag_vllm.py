"""Diagnose the vLLM decode slowdown at >1 sequence (step time 49 ms @1 -> 143 ms @2..8 for Qwen2.5-AWQ).

Loads the model ONCE with the same engine settings as bench_vllm.py, then times decode for
batch 1/2/4 under different SamplingParams variants:
  A  min_tokens=128 + ignore_eos        (what bench_vllm.py used)
  B  ignore_eos only                      (no min_tokens logits processor)
  C  ignore_eos, separate "decode-only" timing via per-token streaming is not available offline,
     so decode = t(128 tokens) - t(1 token), same as the benchmark.
Then, if VLLM_TORCH_PROFILER_DIR is set, profiles one batch-1 and one batch-2 run (32 tokens).

usage: VLLM_TORCH_PROFILER_DIR=... python scripts/diagnosis/diag_vllm.py <model> [--eager] [--lm-only]
"""
import argparse, os, random, statistics, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--lm-only", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile-only", action="store_true")
    args = ap.parse_args()
    prof_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=args.model, dtype="half", max_model_len=2048, seed=0, enforce_eager=args.eager,
              enable_prefix_caching=False, max_num_batched_tokens=2048, max_num_seqs=16,
              gpu_memory_utilization=0.80)
    if not args.eager:
        kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16]}
    if args.lm_only:
        kw["language_model_only"] = True
    if prof_dir:  # vLLM 0.31: profiling must be enabled via profiler_config
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": prof_dir}
    llm = LLM(**kw)
    vocab = min(llm.get_tokenizer().vocab_size, 100_000)
    rng = random.Random(0)
    P = lambda n: [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab) for _ in range(128)]) for _ in range(n)]

    variants = {
        "A_min_tokens": dict(max_tokens=128, min_tokens=128, ignore_eos=True, temperature=0.0),
        "B_ignore_eos": dict(max_tokens=128, ignore_eos=True, temperature=0.0),
    }
    one = SamplingParams(max_tokens=1, temperature=0.0)
    for n in (1, 2, 4):  # warm-up all shapes
        llm.generate(P(n), SamplingParams(**variants["B_ignore_eos"]), use_tqdm=False)

    print("\nvariant        batch  t_1tok(s)  t_128(s)  ms/step  decode tok/s")
    for name, sp in ({} if args.profile_only else variants).items():
        for n in (1, 2, 4):
            t1, t128 = [], []
            for _ in range(args.reps):
                p = P(n); t0 = time.perf_counter(); llm.generate(p, one, use_tqdm=False); t1.append(time.perf_counter() - t0)
                p = P(n); t0 = time.perf_counter(); llm.generate(p, SamplingParams(**sp), use_tqdm=False); t128.append(time.perf_counter() - t0)
            a, b = statistics.median(t1), statistics.median(t128)
            step = (b - a) / 127
            print(f"{name:14s} {n:5d}  {a:9.3f}  {b:8.3f}  {step*1000:7.1f}  {n/step:11.1f}", flush=True)

    if prof_dir:
        sp = SamplingParams(max_tokens=32, ignore_eos=True, temperature=0.0)
        for n in (1, 2):
            d = os.path.join(prof_dir, f"bs{n}")
            os.makedirs(d, exist_ok=True)
            before = set(os.listdir(prof_dir))
            llm.start_profile()
            llm.generate(P(n), sp, use_tqdm=False)
            llm.stop_profile()
            time.sleep(10)  # let the trace finish writing
            for f in set(os.listdir(prof_dir)) - before:
                if f != f"bs{n}":
                    os.rename(os.path.join(prof_dir, f), os.path.join(d, f))
            print(f"profiled batch {n} -> {d}: {os.listdir(d)}", flush=True)


if __name__ == "__main__":
    main()
