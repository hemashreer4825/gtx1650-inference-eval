# llama.cpp vs vLLM on a 4 GB GTX 1650 — project layout

Lab notebook (dated log of every step and decision): `notes.md`.
Each part has the same folder name in `scripts/`, `results/` and `logs/`.
`results/` = measured data (what goes in the paper). `logs/` = console output (for debugging only).
Nothing is stored twice; tables and charts in `results/final/` are regenerated from the raw data by `scripts/analysis/`.

| Part | Script | Results | What it is |
|---|---|---|---|
| Throughput benchmark | `scripts/bench/run_all.sh` (REP=1) | `results/throughput/rep1/` | tokens/s for prompt 128/512, 1-16 users; GPU memory, power, clocks (every 200 ms) |
| vLLM slowdown diagnosis | `scripts/diagnosis/` | `results/diagnosis/` | why vLLM slows down at >= 2 users (steps 1-5) |
| Checks | `scripts/checks/` | `results/checks/` | data completeness, speed formula, fairness, text sanity |
| Final tables + charts | `scripts/analysis/analyze.py` | `results/final/` | summary.md, CSV tables, 3 charts |

Other folders: `scripts/setup/` (install, download, build, convert, smoke tests -> `logs/setup/`),
`scripts/common.sh` (model list + shared settings), `models/` (weights), `llama.cpp/` (source + build).
Configurations in rep1: llama.cpp, vLLM default (80% VRAM), vLLM with KV cache matched to llama.cpp.
