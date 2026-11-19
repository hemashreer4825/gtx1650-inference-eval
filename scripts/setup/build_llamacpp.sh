#!/bin/bash
# Build llama.cpp with CUDA for the GTX 1650 only (sm_75). Needs cuda-toolkit-12-8 (apt).
set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH
cd ~/infer-eval/llama.cpp
nvcc --version | tail -2
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --config Release -j 4 --target llama-bench llama-batched-bench llama-cli llama-server
git log -1 --format='llama.cpp commit: %H (%cd)'
ls build/bin
