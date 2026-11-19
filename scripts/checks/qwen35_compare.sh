#!/bin/bash
# Same prompts, greedy, thinking off: llama.cpp (converted F16 GGUF) vs vLLM (HF weights, FP16).
# Checks that the locally converted Qwen3.5 GGUF behaves like the original model.
cd ~/infer-eval
cat > /tmp/q35_prompts.json <<'EOF'
["What is 17*23? Answer with just the number.",
 "What is 12*12? Answer with just the number.",
 "Write one sentence about the moon.",
 "List three primary colors.",
 "Translate 'good morning' to French."]
EOF
echo "===== llama.cpp (Qwen3.5-0.8B-F16.gguf)"
llama.cpp/build/bin/llama-server -m models/gguf/Qwen3.5-0.8B-F16.gguf -ngl 99 -c 2048 --port 8099 > logs/gguf_server_q35cmp.log 2>&1 &
pid=$!
for i in $(seq 120); do curl -sf localhost:8099/health >/dev/null && break; sleep 1; done
python3 - <<'EOF'
import json, urllib.request
for q in json.load(open("/tmp/q35_prompts.json")):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": 60, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", body, {"Content-Type": "application/json"}))
    print("  ", repr(q[:30]), "->", repr(json.load(r)["choices"][0]["message"]["content"].strip()[:120]))
EOF
kill $pid; wait $pid 2>/dev/null; sleep 3

echo "===== vLLM (Qwen/Qwen3.5-0.8B, FP16)"
. env.sh; . .venv-vllm/bin/activate
python - <<'EOF' > logs/qwen35_compare_vllm.log 2>&1
import json
def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3.5-0.8B", dtype="half", max_model_len=2048, enforce_eager=True, gpu_memory_utilization=0.8,
              max_num_batched_tokens=2048, max_num_seqs=8, language_model_only=True)
    qs = json.load(open("/tmp/q35_prompts.json"))
    outs = llm.chat([[{"role": "user", "content": q}] for q in qs], SamplingParams(max_tokens=60, temperature=0),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    for q, o in zip(qs, outs):
        print("  ", repr(q[:30]), "->", repr(o.outputs[0].text.strip()[:120]))
if __name__ == "__main__":
    main()
EOF
