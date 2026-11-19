"""Sanity-check all throughput outputs (every repeat + the cuBLAS-fix runs) before reporting.
usage: .venv-tools/bin/python scripts/checks/verify_results.py  > results/checks/verify_results.log
"""
import glob, json, os
import pandas as pd

P = os.path.expanduser("~/infer-eval")
TAGS = ["qwen35-0.8b-fp16", "qwen25-1.5b-q4", "qwen3-1.7b-q4", "minicpm5-2b-q4"]
ok = lambda c: "OK  " if c else "FAIL"
GRID = {(pp, pl) for pp in (128, 512) for pl in (1, 2, 4, 8, 16)}
load = lambda f: pd.DataFrame([json.loads(l) for l in open(f) if l.strip()])

for R in sorted(glob.glob(f"{P}/results/throughput/rep*")) + [f"{P}/results/cublas_fix"]:
    if not os.path.isdir(R):
        continue
    L = R.replace("/results/", "/logs/")
    print(f"\n################ {os.path.relpath(R, P)}")
    files = sorted(glob.glob(f"{R}/llamacpp_batched_*.jsonl") + glob.glob(f"{R}/vllm_*.jsonl"))

    print("== 1. raw files: rows, grid coverage, NaN/zero values")
    for f in files:
        df = load(f)
        grid = set(zip(df.pp, df.pl)) == GRID
        bad = df[["speed_pp", "speed_tg"]].isna().any().any() or (df[["speed_pp", "speed_tg"]] <= 0).any().any()
        print(f"  {ok(len(df) == 10 and grid and not bad and set(df.tg) == {128})} {os.path.basename(f):45s} rows={len(df)} tg={set(df.tg)}")

    print("== 2. speed formula: speed_tg == pl*tg/t_tg, speed_pp == pl*pp/t_pp")
    for f in files:
        df = load(f)
        if "t_tg" in df:
            err = ((df.pl * df.tg / df.t_tg - df.speed_tg).abs() / df.speed_tg).max()
            errp = ((df.pl * df.pp / df.t_pp - df.speed_pp).abs() / df.speed_pp).max()
            print(f"  {ok(err < 0.02 and errp < 0.02)} {os.path.basename(f):45s} max rel err tg={err:.3%} pp={errp:.3%}")

    print("== 3. llama-batched-bench pl=1 vs llama-bench (independent tool)")
    for t in TAGS:
        fb, fl = f"{R}/llamacpp_batched_{t}.jsonl", f"{R}/llamacpp_bench_{t}.jsonl"
        if not (os.path.exists(fb) and os.path.exists(fl)):
            continue
        b, lb = load(fb), [json.loads(l) for l in open(fl) if l.strip()]
        tg_lb = [x["avg_ts"] for x in lb if x.get("n_gen", 0) > 0][0]
        pp_lb = [x["avg_ts"] for x in lb if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0][0]
        tg_b = b[(b.pp == 128) & (b.pl == 1)].speed_tg.item()
        pp_b = b[(b.pp == 512) & (b.pl == 1)].speed_pp.item()
        print(f"  {ok(abs(tg_b / tg_lb - 1) < 0.15)} {t:18s} tg: batched {tg_b:6.1f} vs bench {tg_lb:6.1f} | pp512: batched {pp_b:6.1f} vs bench {pp_lb:6.1f}")

    print("== 4. GPU sampler: power / clocks / temperature per run (throttling check)")
    for f in sorted(glob.glob(f"{R}/gpu_*.csv")):
        g = pd.read_csv(f, skipinitialspace=True)
        busy = g[g.gpu_util_pct > 50] if "gpu_util_pct" in g else g
        busy = g if busy.empty else busy
        print(f"  {ok(busy.sm_clock_mhz.median() > 1000)} {os.path.basename(f):38s} n={len(g):5d}  power med {busy.power_w.median():5.1f} W"
              f"  SM clk med {busy.sm_clock_mhz.median():6.0f} MHz  temp max {g.temp_c.max():3.0f} C")

    print("== 5. errors / fallbacks in logs")
    for f in sorted(glob.glob(f"{L}/*.log")):
        txt = open(f, errors="ignore").read()
        hits = [k for k in ("Traceback", "OutOfMemory", "out of memory", "CUDA error", "retry with --eager", "FAILED") if k in txt]
        print(f"  {ok(not hits)} {os.path.basename(f):45s} {hits}")
