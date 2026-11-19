#!/bin/bash
# Shared settings for every test script. Source it:  . ~/infer-eval/scripts/common.sh
. ~/infer-eval/env.sh
cd "$PROJ"
BIN=$PROJ/llama.cpp/build/bin

# One row per model:  tag | GGUF (llama.cpp) | HF repo (vLLM) | extra vLLM args | FP16 original HF repo (quality reference)
MODELS=(
  "qwen35-0.8b-fp16|models/gguf/Qwen3.5-0.8B-F16.gguf|Qwen/Qwen3.5-0.8B|--lm-only|Qwen/Qwen3.5-0.8B"
  "qwen25-1.5b-q4|models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf|Qwen/Qwen2.5-1.5B-Instruct-AWQ||Qwen/Qwen2.5-1.5B-Instruct"
  "qwen3-1.7b-q4|models/gguf/Qwen_Qwen3-1.7B-Q4_K_M.gguf|JunHowie/Qwen3-1.7B-GPTQ-Int4||Qwen/Qwen3-1.7B"
  "minicpm5-2b-q4|models/gguf/MiniCPM5-2B-Q4_K_M.gguf|openbmb/MiniCPM5-2B-GPTQ||openbmb/MiniCPM5-2B"
)
# usage:  for row in "${MODELS[@]}"; do model_fields "$row"; ...; done   -> $tag $gguf $hf $extra $fp16
model_fields() { IFS='|' read -r tag gguf hf extra fp16 <<< "$1"; }
# ONLY="tag1 tag2" restricts any loop:   skip_model && continue
skip_model() { [[ -n $ONLY && " $ONLY " != *" $tag "* ]]; }

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
# Thermal gate between runs: wait until the GPU is <= COOL_TEMP C (max COOL_MAX s), so every run starts in the same
# thermal state (rep1 runs started at ~63 C). Without it, a hot laptop throttles to 300 MHz and runs get 2x slower.
COOL_TEMP=${COOL_TEMP:-65}; COOL_MAX=${COOL_MAX:-600}
cool() {
  sleep 20
  local t0=$SECONDS
  while (( $(gpu_temp) > COOL_TEMP && SECONDS - t0 < COOL_MAX )); do sleep 10; done
  echo "    [cool] GPU $(gpu_temp) C after $((SECONDS - t0 + 20)) s"
}
