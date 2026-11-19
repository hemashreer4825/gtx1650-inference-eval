#!/bin/bash
# Throughput benchmark, strictly sequential (one framework/model on the GPU at a time).
#   Phase A: llama.cpp   (batched grid + llama-bench) for all 4 GGUFs
#   Phase B: vLLM default  (gpu_memory_utilization 0.80) — how vLLM is normally run
#   Phase C: vLLM matched-KV (KV cache sized to what llama.cpp allocated for the same grid) — fair footprint
# usage: REP=n run_all.sh [A] [B] [C]   (default: all phases)   ONLY="tag ..." restricts models
. ~/infer-eval/scripts/common.sh
: "${REP:?set REP (1, 2, 3)}"; export REP
PHASES=${*:-A B C}
RES=results/throughput/rep$REP
# vLLM: try CUDA graphs (capture sizes 1..16) first; if the engine fails (e.g. OOM), retry in eager mode.
vllm_run() {  # vllm_run <hf> <tag> [args...]
  local hf=$1 t=$2; shift 2
  bash scripts/bench/bench_vllm.sh "$hf" "$t" "$@" && return 0
  echo "!!! $t failed with CUDA graphs -> retrying --eager"
  rm -f "$RES/vllm_${t}.jsonl"
  bash scripts/bench/bench_vllm.sh "$hf" "$t" "$@" --eager
}

for row in "${MODELS[@]}"; do
  model_fields "$row"; skip_model && continue
  if [[ $PHASES == *A* ]]; then
    echo "### [rep$REP A] llama.cpp $tag  $(date +%T)"
    bash scripts/bench/bench_llamacpp.sh "$gguf" "$tag" || echo "!!! llama.cpp $tag FAILED"
    cool
  fi
done

for row in "${MODELS[@]}"; do
  model_fields "$row"; skip_model && continue
  if [[ $PHASES == *B* ]]; then
    echo "### [rep$REP B] vLLM default $tag  $(date +%T)"
    vllm_run "$hf" "${tag}_default" $extra || echo "!!! vLLM default $tag FAILED"
    cool
  fi
  if [[ $PHASES == *C* ]]; then
    # KV (+ recurrent state) MiB that llama.cpp allocated for the same 16 x 640-token grid
    mib=$(grep -hoE "(KV|RS) buffer size = +[0-9.]+ MiB" "logs/throughput/kv_buffers/llamacpp_buffers_${tag}.log" 2>/dev/null \
          | grep -oE "[0-9.]+" | awk '{s+=$1} END {print s+0}')
    if [[ -z $mib || $mib == 0 ]]; then echo "!!! no llama.cpp KV size for $tag, skipping C"; continue; fi
    gb=$(awk -v m="$mib" 'BEGIN {printf "%.3f", m*1.10/1024}')   # +10% for vLLM block granularity
    echo "### [rep$REP C] vLLM matched-KV $tag  kv=${gb} GiB (llama.cpp ${mib} MiB)  $(date +%T)"
    vllm_run "$hf" "${tag}_matchkv" $extra --kv-cache-gb "$gb" || echo "!!! vLLM matched-KV $tag FAILED"
    cool
  fi
done
echo "### rep$REP ALL DONE $(date +%T)"
