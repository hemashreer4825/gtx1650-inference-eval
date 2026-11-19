#!/bin/bash
# Run bench_vllm.py for one model with the GPU sampler alongside.
# usage: REP=n bench_vllm.sh <hf_model> <tag> [extra bench_vllm.py args, e.g. --lm-only --eager]
#   RES/LOGD can be overridden (e.g. the cuBLAS-fix and long-prompt tests write elsewhere).
. ~/infer-eval/scripts/common.sh
. .venv-vllm/bin/activate
MODEL=$1; TAG=$2; shift 2
RES=${RES:-results/throughput/rep${REP:?set REP}}; LOGD=${LOGD:-logs/throughput/rep$REP}
mkdir -p "$RES" "$LOGD"
base=$(gpu_mem)
bash scripts/bench/gpu_sampler.sh "$RES/gpu_vllm_${TAG}.csv" & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
python scripts/bench/bench_vllm.py "$MODEL" --tag "$TAG" --out "$RES/vllm_${TAG}.jsonl" "$@" > "$LOGD/vllm_bench_${TAG}.log" 2>&1
rc=$?
kill $SAMP 2>/dev/null
peak=$(tail -n +2 "$RES/gpu_vllm_${TAG}.csv" | cut -d, -f2 | sort -n | tail -1)
echo "$TAG: exit=$rc baseline ${base} MiB, peak ${peak} MiB, footprint $((peak - base)) MiB"
grep -E "^pp=" "$LOGD/vllm_bench_${TAG}.log"
exit $rc
