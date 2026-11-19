#!/bin/bash
# Smoke-test the three 4-bit models in vLLM (eager, 2048 batched tokens). Quant method auto-detected from config.
. ~/infer-eval/env.sh
cd "$PROJ" && . .venv-vllm/bin/activate
for m in Qwen/Qwen2.5-1.5B-Instruct-AWQ JunHowie/Qwen3-1.7B-GPTQ-Int4 openbmb/MiniCPM5-2B-GPTQ; do
  tag=$(basename "$m")
  log=logs/vllm_smoke_${tag}_eager.log
  echo "=== $m"
  python scripts/setup/vllm_smoke.py "$m" --eager > "$log" 2>&1
  echo "exit=$?"
  grep -E "RESULT" "$log" || grep -E "Error|error:" "$log" | grep -v CUDACaching | tail -3
  grep -oE "(Using [A-Za-z_]+ attention backend|quantization=[a-z_]+|[A-Za-z]*(Marlin|Exllama|AWQ|GPTQ|Machete|Conch)[A-Za-z]*(Kernel|LinearKernel)[A-Za-z]*)" "$log" | sort -u
done
echo SMOKE_ALL_DONE
