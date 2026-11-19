"""vLLM smoke test: load one model on the GTX 1650, generate, report speed + VRAM.

usage: python scripts/setup/vllm_smoke.py <hf_repo_or_path> [--quant awq|gptq] [--eager] [--gpu-util 0.80] [--lm-only]
"""
import argparse, json, subprocess, time


def smi_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lm-only", action="store_true", help="skip vision tower (Qwen3.5)")
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-seqs", type=int, default=32)
    args = ap.parse_args()

    base = smi_used_mib()
    from vllm import LLM, SamplingParams

    kw = dict(model=args.model, dtype="half", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_util, enforce_eager=args.eager, seed=0,
              max_num_batched_tokens=args.max_batched_tokens, max_num_seqs=args.max_seqs)
    if args.quant:
        kw["quantization"] = args.quant
    if args.lm_only:
        kw["language_model_only"] = True

    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    after_load = smi_used_mib()

    msgs = [[{"role": "user", "content": p}] for p in [
        "Explain what a GPU is in two sentences.",
        "Write a haiku about memory bandwidth.",
        "What is 17 * 23? Answer with just the number.",
        "List three uses of the Python `zip` function.",
    ]]
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    chat_kw = {"chat_template_kwargs": {"enable_thinking": False}}

    llm.chat(msgs[:1], SamplingParams(max_tokens=8), use_tqdm=False, **chat_kw)  # warm-up

    t0 = time.time()
    outs = llm.chat(msgs, sp, use_tqdm=False, **chat_kw)
    gen_s = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)

    t0 = time.time()
    llm.chat(msgs[:1], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True), use_tqdm=False, **chat_kw)
    single_s = time.time() - t0

    for o in outs:
        print("-" * 60)
        print(repr(o.outputs[0].text[:300]))

    res = dict(model=args.model, quant=args.quant, eager=args.eager, load_s=round(load_s, 1),
               vram_baseline_mib=base, vram_after_load_mib=after_load, vram_now_mib=smi_used_mib(),
               batch4_tokens=ntok, batch4_tok_s=round(ntok / gen_s, 1),
               single_stream_tok_s=round(128 / single_s, 1))
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
