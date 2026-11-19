#!/bin/bash
# Sample GPU state every 200 ms until killed.
# usage: gpu_sampler.sh <out.csv>   (run in background, kill when done)
# NOTE: under WSL nvidia-smi reports only device-total memory (no per-process), so
# footprint = peak(memory.used) - baseline(memory.used before model load).
OUT=$1
echo "timestamp,memory_used_mib,power_w,sm_clock_mhz,mem_clock_mhz,temp_c,gpu_util_pct" > "$OUT"
exec nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 >> "$OUT"
