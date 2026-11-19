#!/bin/bash
# CPU-only venv for llama.cpp's convert_hf_to_gguf.py (kept separate from .venv-vllm)
set -e
cd ~/infer-eval
python3 -m venv .venv-tools
. .venv-tools/bin/activate
pip install -q --upgrade pip
pip install -q --no-cache-dir -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install -q --no-cache-dir -e llama.cpp/gguf-py
python -c "import torch, transformers, gguf; print('tools venv OK', torch.__version__, transformers.__version__)"
