#!/bin/bash
# Download all benchmark models. vLLM weights -> HF cache ($HF_HOME); GGUF -> models/gguf/
set -e
. ~/infer-eval/env.sh
cd "$PROJ" && . .venv-vllm/bin/activate

# vLLM checkpoints
hf download Qwen/Qwen3.5-0.8B
hf download Qwen/Qwen2.5-1.5B-Instruct-AWQ
hf download JunHowie/Qwen3-1.7B-GPTQ-Int4
hf download openbmb/MiniCPM5-2B-GPTQ

# llama.cpp GGUF (Q4_K_M). Qwen3.5-0.8B F16 GGUF is converted locally (scripts/convert_qwen35_f16.sh)
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/gguf
hf download bartowski/Qwen_Qwen3-1.7B-GGUF Qwen_Qwen3-1.7B-Q4_K_M.gguf --local-dir models/gguf
hf download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q4_K_M.gguf --local-dir models/gguf

ls -la models/gguf; du -sh models/hf models/gguf
echo DOWNLOADS_DONE
