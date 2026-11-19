#!/bin/bash
# Quick GPU check of every GGUF: loads fully on GPU, short bench, short generation.
. ~/infer-eval/env.sh
cd "$PROJ"
BIN=llama.cpp/build/bin
for g in models/gguf/Qwen3.5-0.8B-F16.gguf models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
         models/gguf/Qwen_Qwen3-1.7B-Q4_K_M.gguf models/gguf/MiniCPM5-2B-Q4_K_M.gguf; do
  echo "=== $g"
  $BIN/llama-bench -m "$g" -ngl 99 -p 128 -n 32 -r 1 -o md 2>&1 | grep -E "^\||error|failed" | tail -3
  timeout 120 $BIN/llama-cli -m "$g" -ngl 99 -c 2048 --temp 0 -n 40 -no-cnv \
     -p "Question: What is 17 * 23? Answer:" 2>/dev/null | tail -c 300
  echo
done
echo SMOKE_DONE
