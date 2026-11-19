"""vLLM side of the Qwen3.5 text comparison (see qwen35_compare.sh). Greedy, thinking off."""
import json


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3.5-0.8B", dtype="half", max_model_len=2048, enforce_eager=True, gpu_memory_utilization=0.8,
              max_num_batched_tokens=2048, max_num_seqs=8, language_model_only=True)
    qs = json.load(open("/tmp/q35_prompts.json"))
    sp = SamplingParams(max_tokens=60, temperature=0)
    for q in qs:  # one at a time (batch 1) to avoid any batching effects
        o = llm.chat([{"role": "user", "content": q}], sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        print("  ", repr(q[:30]), "->", repr(o[0].outputs[0].text.strip()[:120]), flush=True)


if __name__ == "__main__":
    main()
