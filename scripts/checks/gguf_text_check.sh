#!/bin/bash
# Text sanity check of each GGUF via llama-server (chat template, greedy). Verifies llama.cpp outputs are coherent,
# i.e. the speed numbers come from a correctly working model (esp. the locally converted Qwen3.5 F16 GGUF).
cd ~/infer-eval
BIN=llama.cpp/build/bin/llama-server
for g in Qwen3.5-0.8B-F16.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf Qwen_Qwen3-1.7B-Q4_K_M.gguf MiniCPM5-2B-Q4_K_M.gguf; do
  echo "===== $g"
  $BIN -m models/gguf/$g -ngl 99 -c 2048 --port 8099 > logs/gguf_server_$g.log 2>&1 &
  pid=$!
  for i in $(seq 120); do curl -sf localhost:8099/health >/dev/null && break; sleep 1; done
  for q in "What is 17*23? Answer with just the number." "What is the capital of France? One word."; do
    curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":200,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  ->", repr(d["choices"][0]["message"]["content"].strip()[:200]))'
  done
  kill $pid; wait $pid 2>/dev/null; sleep 3
done
