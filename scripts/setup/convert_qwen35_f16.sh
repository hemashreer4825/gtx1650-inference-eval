#!/bin/bash
# Convert Qwen/Qwen3.5-0.8B (BF16 safetensors) -> F16 GGUF, text model only.
# Same source weights vLLM loads, so both frameworks run identical FP16 weights.
set -e
. ~/infer-eval/env.sh
cd "$PROJ" && . .venv-tools/bin/activate
SRC=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/ | head -1)
echo "source: $SRC"
python llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype f16 --outfile models/gguf/Qwen3.5-0.8B-F16.gguf
ls -la models/gguf/Qwen3.5-0.8B-F16.gguf
