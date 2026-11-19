#!/bin/bash
# Print llama.cpp's GPU buffer breakdown (model / KV / recurrent state / compute) for the benchmark grid's
# context (same -c, -b, -ub, n_seq_max=16 as bench_llamacpp.sh), using a 1-token run with verbose logs.
# -> logs/throughput/kv_buffers/llamacpp_buffers_<tag>.log  (read by run_all.sh phase C)
. ~/infer-eval/scripts/common.sh
D=logs/throughput/kv_buffers; mkdir -p $D
for row in "${MODELS[@]}"; do
  model_fields "$row"; skip_model && continue
  "$BIN/llama-batched-bench" -m "$gguf" -ngl 99 -c $((16*640)) -b 2048 -ub 512 \
    -npp 8 -ntg 1 -npl 16 -lv 4 > "$D/llamacpp_buffers_${tag}.log" 2>&1
  echo "== $tag"
  grep -iE "(model|KV|RS|compute) buffer size|n_ctx_seq|n_seq_max|flash_attn" "$D/llamacpp_buffers_${tag}.log" \
    | sed -E 's/^[0-9. ]+[A-Z] //' | sort -u
done
