# Experiment record

llama.cpp vs vLLM on a GTX 1650 (4 GB). Written from the files in this folder, not from memory.
Paths are relative to `~/infer-eval/`. Sections are added one at a time, when reviewed.

---

## 1. Start-up (smoke) tests

Date: 2026-11-17.

### Purpose

Before spending hours on benchmarks, confirm that each of the 4 models:
1. loads fully on the 4 GB GPU (every layer on the GPU),
2. runs without errors or out-of-memory,
3. produces speed numbers,
4. produces sensible text.

There was **no automatic scoring**. A model passed if all four points held; point 4 was judged by reading the output.

**These are not benchmark results.** No number from the smoke tests is used in the final tables, charts or paper.
The analysis script (`scripts/analysis/analyze.py`) reads only `results/throughput/rep1/`, never these logs.

### 1a. llama.cpp smoke test

| Item | Value |
|---|---|
| Script | `scripts/setup/llamacpp_smoke.sh` |
| Output | `logs/setup/llamacpp_smoke.log` |
| Time | 2026-11-17 13:31 |
| Tools | `llama-bench` and `llama-cli`, both built into llama.cpp (not written by us; llama.cpp source unmodified) |

What each tool did:
- **`llama-bench`** (llama.cpp's speed tool): 128-token prompt (`-p 128`), 32 generated tokens (`-n 32`),
  1 repetition (`-r 1`), all layers on GPU (`-ngl 99`). Reports prompt speed (pp128) and generation speed (tg32).
- **`llama-cli`** (llama.cpp's command-line text generator): greedy (`--temp 0`), 40 tokens, context 2048,
  all layers on GPU.

Question asked (**1 question**):
```
Question: What is 17 * 23? Answer:
```

Results:

| Model (GGUF) | Size | Prompt speed pp128 | Generation speed tg32 | Text answer |
|---|---|---|---|---|
| Qwen3.5-0.8B F16 | 1.44 GiB | 273.5 tok/s | 34.2 tok/s | not captured |
| Qwen2.5-1.5B Q4_K_M | 1.04 GiB | 343.7 tok/s | 49.0 tok/s | not captured |
| Qwen3-1.7B Q4_K_M | 1.19 GiB | 302.5 tok/s | 43.7 tok/s | not captured |
| MiniCPM5-2B Q4_K_M | 1.45 GiB | 234.3 tok/s | 34.4 tok/s | not captured |

Outcome:
- All 4 models load and run entirely on the GPU. Points 1–3 passed.
- **`llama-cli` printed no text in this llama.cpp build**, so point 4 could not be judged here and no answer exists.
  The text check was therefore repeated with `llama-server` (section 1c).
- Speeds are from 1 repetition with no warm-up control: a sanity check only.

### 1b. vLLM smoke test

| Item | Value |
|---|---|
| Script | `scripts/setup/vllm_smoke.py`; the three 4-bit models were run by `scripts/setup/run_vllm_smoke_all.sh` |
| Outputs | `logs/setup/vllm_smoke_qwen3.5-0.8b_fp16.log` (1st try), `logs/setup/vllm_smoke_qwen3.5-0.8b_fp16_eager.log`, `logs/setup/vllm_smoke_Qwen2.5-1.5B-Instruct-AWQ_eager.log`, `logs/setup/vllm_smoke_Qwen3-1.7B-GPTQ-Int4_eager.log`, `logs/setup/vllm_smoke_MiniCPM5-2B-GPTQ_eager.log`, summary `logs/setup/vllm_smoke_all_summary.log` |
| Time | 2026-11-17 12:12–13:03 |

Settings: `dtype=half`, `max_model_len=2048`, `gpu_memory_utilization=0.80`, `max_num_batched_tokens=2048`,
`max_num_seqs=32`, eager mode (`enforce_eager`, no CUDA graphs), `seed=0`, chat template, Qwen thinking mode off,
greedy (`temperature=0`), max 128 tokens per answer. One short warm-up generation first.
**The 4 questions were sent together as one batch.**

Questions asked (**4 questions**):
1. `Explain what a GPU is in two sentences.`
2. `Write a haiku about memory bandwidth.`
3. `What is 17 * 23? Answer with just the number.`
4. ``List three uses of the Python `zip` function.``

Metrics recorded (the `RESULT` line of each log):
- `load_s`: time to load the model
- `vram_baseline_mib` / `vram_after_load_mib`: GPU memory before and after loading (`nvidia-smi`)
- `batch4_tok_s`: tokens per second while answering the 4 questions together
- `single_stream_tok_s`: tokens per second for one forced 128-token answer

Results:

| Model | Load (s) | VRAM before → after load | Batch-of-4 tok/s | Single tok/s |
|---|---|---|---|---|
| Qwen3.5-0.8B FP16 | 472.0 | 416 → 3,459 MiB | 8.3 | 14.6 |
| Qwen2.5-1.5B AWQ | 164.6 | 282 → 3,485 MiB | 15.1 | 21.2 |
| Qwen3-1.7B GPTQ | 93.2 | 267 → 3,509 MiB | 13.3 | 18.0 |
| MiniCPM5-2B GPTQ (AWQ-packed) | 91.1 | 281 → 3,522 MiB | 7.5 | 14.0 |

Answers (first 300 characters are saved in each log, after each `-----` line):

| Model | Q1 GPU | Q2 haiku | Q3 17 × 23 | Q4 `zip` |
|---|---|---|---|---|
| Qwen3.5-0.8B FP16 | "A GPU is a specialized graphics processing unit that significantly accelerates the rendering of 3D graphics …" | "Silent streams of data, / A river of pixels, / Where time slows to a crawl." | **391** ✓ | lists uses (pairing elements, …) |
| Qwen2.5-1.5B AWQ | "A GPU, or Graphics Processing Unit, is a specialized electronic circuit designed to rapidly manipulate and alter memory …" | "Memory bandwidth, / A river of data flowing, / Past the CPU's shore." | **391** ✓ | lists uses (combining lists, …) |
| Qwen3-1.7B GPTQ | "A GPU, or Graphics Processing Unit, is a specialized electronic circuit designed to handle complex visual computations …" | "Cold data streams through the core, / Echoes of past in swift, unyielding flow. / Memory's pulse, a fleeting shore." | **351** ✗ | lists uses (combining multiple lists, …) |
| MiniCPM5-2B GPTQ | "A GPU is a specialized processor designed to handle complex mathematical calculations quickly …" | "The data flows, / A quiet stream, / Memory holds." | **391** ✓ | "I can't solve this." ✗ |

Outcome:
- All 4 models load and answer. Points 1–4 passed; two wrong answers (Qwen3 on 17 × 23, MiniCPM5 on `zip`)
  show these small 4-bit models make mistakes, which a smoke test does not judge.
- **Qwen3.5 first try failed**: with CUDA graphs on (vLLM's default), it ran out of GPU memory
  (`CUDA error: out of memory` in `vllm_smoke_qwen3.5-0.8b_fp16.log`). Eager mode worked. The benchmark later used
  CUDA graphs limited to batch sizes 1–16, which fit.
- vLLM ran all three 4-bit models with **Marlin** kernels, attention backend **TRITON_ATTN** (`vllm_smoke_all_summary.log`).
- vLLM used about **3.5 GB for every model**, whatever its size. That is vLLM's up-front memory reservation
  (`gpu_memory_utilization=0.80`), not what the model needs. This finding led to the benchmark's "vLLM matched-KV"
  configuration.
- Speeds here are eager mode, with no CUDA graphs and no warm-up control: a sanity check only.

### 1c. llama.cpp text check via `llama-server`

Since `llama-cli` returned no text, the text check for llama.cpp was repeated with `llama-server`.

| Item | Value |
|---|---|
| Script | `scripts/checks/gguf_text_check.sh` |
| Output | `results/checks/gguf_text_check.log` (answers); server logs `logs/checks/gguf_server_<model>.log` |
| Tool | `llama-server`, built into llama.cpp: runs the model as a chat server with an OpenAI-style API (`/v1/chat/completions`) |
| Model files | the same 4 GGUF files as in 1a (`models/gguf/`) |

Why `llama-server`: it gave text output where `llama-cli` did not, and it formats each question as a proper chat
message using the model's own chat template — the same way vLLM does in 1b. (`llama-cli` in 1a was given raw text
with no chat formatting.)

Settings: all layers on GPU (`-ngl 99`), context 2048 (`-c 2048`), greedy (`temperature=0`), max 200 tokens,
Qwen thinking mode off (`enable_thinking: false`). One model at a time: start server → wait until `/health` is OK →
ask the questions → stop server.

Questions asked (**2 questions**):
1. `What is 17*23? Answer with just the number.`
2. `What is the capital of France? One word.`

Results:

| Model (GGUF) | Q1 17 × 23 | Q2 capital |
|---|---|---|
| Qwen3.5-0.8B F16 | 501 ✗ (correct: 391) | Paris |
| Qwen2.5-1.5B Q4_K_M | 391 ✓ | Paris. |
| Qwen3-1.7B Q4_K_M | 391 ✓ | Paris |
| MiniCPM5-2B Q4_K_M | 391 ✓ | Paris |

Outcome:
- All 4 llama.cpp models produce sensible text. Point 4 passed for llama.cpp.
- Qwen3.5-0.8B got 17 × 23 wrong. Because its GGUF was converted locally (not downloaded), this raised the question of
  whether the conversion was faulty — checked separately (to be documented in its own section).

### Why the two smoke tests differ

There was no plan to make them equal. The llama.cpp test was a quick speed check that also tried one text prompt;
the vLLM script was written to load, answer a small mixed set of questions, and report memory.

| | llama.cpp smoke | vLLM smoke |
|---|---|---|
| Questions | 1 (`llama-cli`), then 2 (`llama-server`, 1c) | 4 (one batch) |
| Answers saved | `llama-cli`: no (printed nothing); `llama-server`: yes | yes, first 300 characters |
| Speed numbers | yes (llama-bench, 1 repetition) | yes (rough, eager mode) |
| Memory numbers | no | yes |

### Not to be confused with

- The **benchmark** (random-token input, 120 settings): separate test, separate files (`results/throughput/rep1/`).
- The **Qwen3.5 conversion check** (`results/checks/qwen35_text_*.log` and the weight comparison): to be documented in its own section.

---

## 2. vLLM memory: default vs KV cache matched to llama.cpp

### Background

While generating, a model keeps a working memory of every token seen so far: the **KV cache** (key-value cache).
More users and longer text need a bigger KV cache. Both frameworks keep it in GPU memory but size it differently:

- **llama.cpp** allocates the model plus exactly the KV cache requested with `-c` (context size), nothing more.
- **vLLM** claims a fixed share of total GPU memory at start-up (`gpu_memory_utilization`), and after loading the model
  fills all remaining space with KV cache, stored in fixed-size pages, whether the workload needs it or not.

So vLLM's memory use mainly reflects its reservation, not what the test needs. Comparing only "as normally used"
would make vLLM look memory-hungry for that reason alone. A third configuration was added to separate the two.

### The three configurations

| Configuration | Memory setting | Purpose |
|---|---|---|
| llama.cpp | `-c 10240` requested (16 users × 640 tokens = largest cell); llama.cpp rounds each sequence's share up to a multiple of 256, giving 12,288 tokens | framework as normally used |
| vLLM default | `gpu_memory_utilization=0.80` (vLLM's own default 0.92 did not start on this card) | framework as normally used |
| vLLM matched-KV | `kv_cache_memory_bytes` = llama.cpp's KV memory × 1.1 | same KV memory as llama.cpp: tests whether memory size explains the differences |

Both vLLM settings are **official vLLM options** (documented, user-settable), not something written for this study.

### How matched-KV was set (automatically, in `scripts/bench/run_all.sh`, phase C)

1. Read the KV memory llama.cpp allocated for each model from its log (`logs/throughput/kv_buffers/`).
2. Multiply by 1.1 — a 10% margin, because vLLM allocates in fixed-size pages, so vLLM gets **at least** as much room
   as llama.cpp, never less.
3. Start vLLM with that fixed KV size (`--kv-cache-gb` in our script → vLLM's `kv_cache_memory_bytes`) instead of the
   80% rule, all other settings unchanged.
4. Run the same 10-cell grid (2 prompt lengths × 5 user counts).

| Model | llama.cpp KV memory | × 1.1 | vLLM matched-KV cache |
|---|---|---|---|
| Qwen3.5-0.8B | 144 MiB KV + 308.25 MiB recurrent state = 452.25 MiB | 497 MiB | 22,820 tokens* |
| Qwen2.5-1.5B | 336 MiB (12,288 tokens) | 370 MiB | 13,504 tokens (= 1.1 × 12,288) |
| Qwen3-1.7B | 1,344 MiB | 1,478 MiB | 13,504 tokens |
| MiniCPM5-2B | 504 MiB | 554 MiB | 13,504 tokens |

\*Qwen3.5 is a hybrid model: half its layers use linear attention, which keeps a "recurrent state" instead of a normal
KV cache. vLLM counts that memory differently, so its token number is not comparable to the others; the memory amount
was matched correctly.

### Result: before vs after matching

**Memory — changed.**

| Model | KV cache: default → matched | GPU memory footprint: default → matched | Change |
|---|---|---|---|
| Qwen3.5-0.8B | 38,912 → 22,820 tokens | 3,091 → 2,699 MiB | −392 MiB |
| Qwen2.5-1.5B | 58,096 → 13,504 tokens | 3,212 → 2,591 MiB | −621 MiB |
| Qwen3-1.7B | 11,504 → 13,504 tokens | 2,912 → 3,095 MiB | **+183 MiB** |
| MiniCPM5-2B | 14,096 → 13,504 tokens | 3,533 → 2,991 MiB | −542 MiB |

- 3 of 4 models used 392–621 MiB less: by default vLLM had reserved far more KV cache than needed
  (Qwen2.5: 58,096 tokens when ~12,000 were enough).
- Qwen3-1.7B used more: vLLM's own default (11,504 tokens) was already smaller than llama.cpp's 12,288, so matching
  raised it.
- Even matched, vLLM used more memory than llama.cpp for 3 of 4 models (e.g. Qwen2.5: 2,591 vs 1,661 MiB). The rest
  is vLLM's own overhead (software, CUDA graphs), not KV cache.

**Speed — not changed.** (tokens/s, default → matched)

| Model | Decode, 1 user | Decode, 16 users | Prefill, 1 user (512-token prompt) |
|---|---|---|---|
| Qwen3.5-0.8B | 27.7 → 25.2 | 44.8 → 45.0 | 238 → 238 |
| Qwen2.5-1.5B | 20.5 → 20.2 | 84.7 → 85.1 | 215 → 216 |
| Qwen3-1.7B | 17.7 → 17.6 | 64.5 → 64.5 | 201 → 201 |
| MiniCPM5-2B | 13.9 → 13.8 | 64.9 → 65.5 | 141 → 143 |

Across all 40 cells: 36 within 2%, median difference 0.7%. Largest: two Qwen3.5 cells, −9% (1 user, 128-token prompt)
and −12% (16 users, 512-token prompt) — single runs, no consistent direction.

### What it means

1. vLLM's extra memory use is mostly its up-front reservation; sizing the KV cache to the workload saves memory with
   no measurable speed cost.
2. Memory size is **not** why vLLM is slower than llama.cpp in this study.
3. Practical point: on a 4 GB GPU, 0.4–0.6 GB is a large saving, obtained with an existing vLLM setting.

### Limits (must be stated with the finding)

- "No speed cost" holds **for this workload**: at most 16 users × 640 tokens at once. A smaller KV cache holds fewer
  tokens; with more users or longer texts it fills up, vLLM queues requests, and throughput drops. Size the KV cache to
  the expected workload, not as small as possible.
- Only one reduced size per model was tested, not a range of sizes.
- The setting itself is known (vLLM documentation); the contribution is the measurement on a 4 GB GPU.
- One run per configuration.

Evidence: `results/throughput/rep1/vllm_<model>_default.jsonl` and `vllm_<model>_matchkv.jsonl` (speeds),
`results/throughput/rep1/gpu_vllm_*.csv` (memory), `logs/throughput/rep1/vllm_bench_*.log` ("GPU KV cache size"),
`logs/throughput/kv_buffers/` (llama.cpp KV sizes), `logs/throughput/rep1/run_all_C.log` (phase C).
