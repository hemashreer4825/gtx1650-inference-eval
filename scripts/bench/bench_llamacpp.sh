#!/bin/bash
# llama.cpp throughput for one GGUF: same (pp, pl, tg) grid as bench_vllm.py, plus llama-bench single-stream numbers.
# usage: REP=n bench_llamacpp.sh <gguf> <tag>      -> results/throughput/rep$REP/, logs/throughput/rep$REP/
#   RES/LOGD/PP/PL/TG can be overridden (the long-prompt test reuses this script).
# Each jsonl row gets t_end (unix time when the cell finished) so energy can be matched to the GPU power log.
set -e
. ~/infer-eval/scripts/common.sh
GGUF=$1; TAG=$2
RES=${RES:-results/throughput/rep${REP:?set REP}}; LOGD=${LOGD:-logs/throughput/rep$REP}
mkdir -p "$RES" "$LOGD"
PP=${PP:-128,512}; PL=${PL:-1,2,4,8,16}; TG=${TG:-128}
PPMAX=$(tr , '\n' <<< "$PP" | sort -n | tail -1); PLMAX=$(tr , '\n' <<< "$PL" | sort -n | tail -1)
CTX=$(( PLMAX * (PPMAX + TG) ))   # enough KV for the largest cell

base=$(gpu_mem)
bash scripts/bench/gpu_sampler.sh "$RES/gpu_llamacpp_${TAG}.csv" & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT

# 1) batched throughput grid (mirrors bench_vllm.py); stdbuf -> each line is timestamped as it is printed
t_start=$(date +%s.%N)
stdbuf -oL "$BIN/llama-batched-bench" -m "$GGUF" -ngl 99 -c $CTX -b 2048 -ub 512 \
  -npp $PP -ntg $TG -npl $PL --output-format jsonl \
  2> "$LOGD/llamacpp_batched_${TAG}.log" \
  | python3 -u -c "
import sys, json, time
for l in sys.stdin:
    if not l.startswith('{'): continue
    d = json.loads(l); d.update(framework='llama.cpp', tag='$TAG', model='$GGUF', vram_base_mib=$base,
                               t_run_start=round($t_start, 3), t_end=round(time.time(), 3))
    print(json.dumps(d), flush=True)" > "$RES/llamacpp_batched_${TAG}.jsonl"

# 2) single-stream standard numbers (pp512 / tg128, 5 reps) — main grid only
if [[ -z $SKIP_LLAMA_BENCH ]]; then
  "$BIN/llama-bench" -m "$GGUF" -ngl 99 -p 512 -n 128 -r 5 -o jsonl \
    > "$RES/llamacpp_bench_${TAG}.jsonl" 2> "$LOGD/llamacpp_bench_${TAG}.log"
fi

kill $SAMP 2>/dev/null || true
peak=$(tail -n +2 "$RES/gpu_llamacpp_${TAG}.csv" | cut -d, -f2 | sort -n | tail -1)
echo "$TAG: baseline ${base} MiB, peak ${peak} MiB, footprint $((peak - base)) MiB"
