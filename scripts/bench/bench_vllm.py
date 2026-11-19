"""vLLM throughput benchmark mirroring llama.cpp's `llama-batched-bench`.

For every (pp, pl) pair:  pl independent random prompts of pp tokens, each generating tg tokens.
  t_pp  = time for prefill only (max_tokens=1)           speed_pp = pp*pl / t_pp
  t     = time for prefill + tg tokens (ignore_eos)      speed    = (pp+tg)*pl / t
  t_tg  = t - t_pp                                       speed_tg = tg*pl / t_tg
Prefix caching is OFF and every request gets fresh random tokens, so nothing is reused.
Each measurement is the median of --reps runs. One JSON line per (pp, pl) -> --out.

usage: python scripts/bench/bench_vllm.py <model> --tag NAME --out results/throughput/rep1/x.jsonl [--lm-only] [--eager]
Rows carry t_start/t_end (unix time of the cell) so energy can be matched to the GPU power log.
"""
import argparse, json, random, statistics, subprocess, time


def smi_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pp", type=ints, default=[128, 512])
    ap.add_argument("--pl", type=ints, default=[1, 2, 4, 8, 16])
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--kv-cache-gb", type=float, default=None, help="fixed KV cache size instead of gpu-util")
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--lm-only", action="store_true")
    ap.add_argument("--cudagraph-max", type=int, default=16, help="largest CUDA-graph capture size (non-eager)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    vram_base = smi_used_mib()
    kw = dict(model=args.model, dtype="half", max_model_len=args.max_len, seed=args.seed,
              enforce_eager=args.eager, enable_prefix_caching=False,
              max_num_batched_tokens=args.max_batched_tokens, max_num_seqs=max(args.pl))
    # gpu_memory_utilization is always passed: vLLM checks it against free memory at startup even when
    # kv_cache_memory_bytes is set (default 0.92 > free 3.2 GiB on this GPU -> startup error).
    kw["gpu_memory_utilization"] = args.gpu_util
    if args.kv_cache_gb:
        kw["kv_cache_memory_bytes"] = int(args.kv_cache_gb * 1024**3)
    if not args.eager:
        kw["compilation_config"] = {"cudagraph_capture_sizes": [s for s in [1, 2, 4, 8, 16, 32] if s <= args.cudagraph_max]}
    if args.lm_only:
        kw["language_model_only"] = True

    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    vram_loaded = smi_used_mib()

    tok = llm.get_tokenizer()
    vocab = min(tok.vocab_size, 100_000)
    rng = random.Random(args.seed)

    def prompts(pp, pl):  # fresh random tokens every call -> no cache hits
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab) for _ in range(pp)]) for _ in range(pl)]

    sp_pp = SamplingParams(max_tokens=1, temperature=0.0)
    sp_all = SamplingParams(max_tokens=args.tg, min_tokens=args.tg, ignore_eos=True, temperature=0.0)

    # warm-up: every shape once
    for pl in args.pl:
        llm.generate(prompts(max(args.pp), pl), sp_all, use_tqdm=False)

    common = dict(framework="vllm", tag=args.tag, model=args.model, eager=args.eager,
                  load_s=round(load_s, 2), vram_base_mib=vram_base, vram_loaded_mib=vram_loaded,
                  kv_cache_gb=args.kv_cache_gb, gpu_util=None if args.kv_cache_gb else args.gpu_util)
    with open(args.out, "a") as f:
        for pp in args.pp:
            for pl in args.pl:
                tpp, tall = [], []
                t_cell_start = time.time()  # wall clock, for matching the GPU power log (energy)
                for _ in range(args.reps):
                    p = prompts(pp, pl); t0 = time.perf_counter(); llm.generate(p, sp_pp, use_tqdm=False); tpp.append(time.perf_counter() - t0)
                    p = prompts(pp, pl); t0 = time.perf_counter(); o = llm.generate(p, sp_all, use_tqdm=False); tall.append(time.perf_counter() - t0)
                    assert all(len(x.outputs[0].token_ids) == args.tg for x in o)
                t_pp, t = statistics.median(tpp), statistics.median(tall)
                t_tg = t - t_pp
                row = dict(common, pp=pp, tg=args.tg, pl=pl, n_kv=pl * (pp + args.tg),
                           t_pp=round(t_pp, 4), speed_pp=round(pp * pl / t_pp, 2),
                           t_tg=round(t_tg, 4), speed_tg=round(args.tg * pl / t_tg, 2),
                           t=round(t, 4), speed=round((pp + args.tg) * pl / t, 2),
                           vram_now_mib=smi_used_mib(),
                           t_start=round(t_cell_start, 3), t_end=round(time.time(), 3))
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"pp={pp:4d} pl={pl:2d}  S_PP={row['speed_pp']:8.1f} t/s  S_TG={row['speed_tg']:7.1f} t/s  VRAM={row['vram_now_mib']} MiB", flush=True)


if __name__ == "__main__":
    main()
