"""Build every final table and chart from the raw results -> results/final/ (one place, no copies elsewhere).

usage: .venv-tools/bin/python scripts/analysis/analyze.py

Input:  results/throughput/rep1/  (llama.cpp, vLLM default, vLLM matched-KV; raw rows + GPU sampler CSVs)
Output: results/final/summary.md, *.csv (tables), *.png (charts)
If more runs are ever added as results/throughput/rep2, rep3 ..., tables show mean ± std automatically.
"""
import glob, json, os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = os.path.expanduser("~/infer-eval")
R, OUT = f"{P}/results", f"{P}/results/final"
os.makedirs(OUT, exist_ok=True)

MODELS = [("qwen35-0.8b-fp16", "Qwen3.5-0.8B (FP16)"), ("qwen25-1.5b-q4", "Qwen2.5-1.5B (4-bit)"),
          ("qwen3-1.7b-q4", "Qwen3-1.7B (4-bit)"), ("minicpm5-2b-q4", "MiniCPM5-2B (4-bit)")]
NAME = dict(MODELS)
# fixed categorical order (reference palette slots 1-3, validated all-pairs incl. colour-blindness)
CONFIGS = [  # key, label, colour, marker, linestyle, filled
    ("llama.cpp", "llama.cpp (GGUF)", "#2a78d6", "o", "-", True),
    ("vllm_default", "vLLM (default, 80% VRAM)", "#eb6834", "s", "-", True),
    ("vllm_matchkv", "vLLM (KV matched to llama.cpp)", "#1baf7a", "^", "-", True),
]
CLABEL = {c[0]: c[1] for c in CONFIGS}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
md = []  # summary.md sections


def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def split_tag(d):
    if d["framework"] == "llama.cpp":
        return d["tag"], "llama.cpp"
    t, mode = d["tag"].rsplit("_", 1)
    return t, f"vllm_{mode}"


def power_trace(csv):
    """GPU sampler CSV -> DataFrame with unix time column 'ts' (nvidia-smi prints local time)."""
    g = pd.read_csv(csv, skipinitialspace=True)
    off = datetime.now().astimezone().utcoffset().total_seconds()
    dt = pd.to_datetime(g["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    g["ts"] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds() - off
    return g


def load_grid(folder, rep):
    rows = []
    for f in glob.glob(f"{folder}/llamacpp_batched_*.jsonl") + glob.glob(f"{folder}/vllm_*.jsonl"):
        rs = jl(f)
        if not rs:
            continue
        tag, cfg = split_tag(rs[0])
        csv = f"{folder}/gpu_llamacpp_{tag}.csv" if cfg == "llama.cpp" else f"{folder}/gpu_vllm_{rs[0]['tag']}.csv"
        g = power_trace(csv) if os.path.exists(csv) else None
        peak = int(g["memory_used_mib"].max()) if g is not None else None
        for d in rs:
            d.update(model_tag=tag, config=cfg, rep=rep, vram_footprint_mib=(peak - d["vram_base_mib"]) if peak else None)
            rows.append(d)
    return rows


def pm(s, fmt="{:.1f}"):
    s = s.dropna()
    if s.empty:
        return ""
    return fmt.format(s.mean()) + (f" ± {fmt.format(s.std())}" if len(s) > 1 else "")


def table(df, name, title, note=""):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    md.append(f"## {title}\n\n{note}\n\n" + df.to_markdown(index=False) + "\n")
    print(f"\n== {title}\n{df.to_string(index=False)}")


# ------------------------------------------------------------------ 1. throughput grid (all repeats)
rows = []
for d in sorted(glob.glob(f"{R}/throughput/rep*")):
    rows += load_grid(d, int(d.rsplit("rep", 1)[1]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/all_throughput_rows.csv", index=False)
main = df

summ = []
for tag, name in MODELS:
    for key, *_ in CONFIGS:
        s = main[(main.model_tag == tag) & (main.config == key)]
        if s.empty:
            continue
        c = lambda pp, pl, col: s[(s.pp == pp) & (s.pl == pl)][col]
        summ.append({"model": name, "config": CLABEL[key], "runs": s.rep.nunique(),
                     "decode tok/s, 1 user": pm(c(128, 1, "speed_tg")),
                     "decode tok/s, 16 users": pm(c(128, 16, "speed_tg")),
                     "prefill tok/s, 1×512": pm(c(512, 1, "speed_pp"), "{:.0f}"),
                     "prefill tok/s, 16×512": pm(c(512, 16, "speed_pp"), "{:.0f}"),
                     "VRAM footprint MiB": pm(s.groupby("rep").vram_footprint_mib.first(), "{:.0f}")})
table(pd.DataFrame(summ), "throughput", "Throughput and VRAM footprint",
      "One full run per configuration (vLLM cells = median of 3 repetitions inside the run). Decode = S_TG = "
      "users×128 / decode time (128-token prompts); prefill = S_PP = users×512 / prefill time. VRAM footprint = peak "
      "nvidia-smi memory.used minus idle baseline.")

# full grid, mean ± std per cell (decode + prefill)
grid = (main.groupby(["model_tag", "config", "pp", "pl"])
        .agg(runs=("rep", "nunique"), decode_mean=("speed_tg", "mean"), decode_std=("speed_tg", "std"),
             prefill_mean=("speed_pp", "mean"), prefill_std=("speed_pp", "std")).reset_index())
grid["cv_decode_%"] = 100 * grid.decode_std / grid.decode_mean
grid.round(2).to_csv(f"{OUT}/throughput_grid_mean_std.csv", index=False)
cv = grid[grid.runs > 1]["cv_decode_%"]
if not cv.empty:
    md.append(f"Run-to-run variation (decode, cells with ≥2 runs): median CV {cv.median():.1f}%, max {cv.max():.1f}%. "
              "Full grid: `throughput_grid_mean_std.csv`.\n")

# ------------------------------------------------------------------ charts
def lines(metric, pp, title, fname, ylabel, configs, data):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4.2))
    for ax, (tag, name) in zip(axes, MODELS):
        for key, label, col, mk, ls, filled in configs:
            s = data[(data.model_tag == tag) & (data.config == key) & (data.pp == pp)]
            if s.empty:
                continue
            g = s.groupby("pl")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.pl, g["mean"], yerr=g["std"].fillna(0), color=col, lw=2, ls=ls, marker=mk, ms=8,
                        mfc=col if filled else SURF, mec=col if not filled else SURF, mew=2, capsize=3, label=label)
            ax.annotate(f"{g['mean'].iloc[-1]:.0f}", (g.pl.iloc[-1], g["mean"].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels([1, 2, 4, 8, 16])
        ax.set_xlim(0.8, 24); ax.set_ylim(bottom=0); ax.set_title(name); ax.set_xlabel("parallel sequences")
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.09, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150, bbox_inches="tight"); plt.close(fig)


main_cfg = CONFIGS[:3]
lines("speed_tg", 128, "Decode throughput vs concurrency — GTX 1650 (4 GB)",
      "decode_tps_vs_concurrency.png", "decode tokens/s", main_cfg, main)
lines("speed_pp", 512, "Prefill throughput vs concurrency (512-token prompts) — GTX 1650 (4 GB)",
      "prefill_tps_vs_concurrency.png", "prefill tokens/s", main_cfg, main)

fig, ax = plt.subplots(figsize=(10, 4.6))
n = len(main_cfg); h = 0.8 / n
for i, (key, label, col, *_) in enumerate(main_cfg):
    ys, vs = [], []
    for j, (tag, _) in enumerate(MODELS):
        s = main[(main.model_tag == tag) & (main.config == key)].groupby("rep").vram_footprint_mib.first().dropna()
        if not s.empty:
            ys.append(j + (i - (n - 1) / 2) * h); vs.append(s.mean())
    ax.barh(ys, vs, height=h - 0.03, color=col, label=label, edgecolor=SURF, linewidth=2)
    for y, v in zip(ys, vs):
        ax.text(v + 40, y, f"{v:,.0f} MiB", va="center", fontsize=9, color=INK2)
ax.axvline(4096 - 262, color=INK2, lw=1, ls="--")
ax.text(4096 - 262, len(MODELS) - 0.45, " usable VRAM", color=INK2, fontsize=9, va="bottom")
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m[1] for m in MODELS]); ax.invert_yaxis()
ax.set_xlabel("GPU memory used above idle baseline (MiB, peak)")
ax.set_xlim(0, 4300); ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
ax.set_title("VRAM footprint during the benchmark grid", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/vram_footprint.png", dpi=150); plt.close(fig)

with open(f"{OUT}/summary.md", "w") as fh:
    fh.write("# Results — llama.cpp vs vLLM on a GTX 1650 (4 GB)\n\n"
             f"Generated by scripts/analysis/analyze.py on {datetime.now():%Y-%m-%d %H:%M}. "
             "Charts: decode_tps_vs_concurrency.png, prefill_tps_vs_concurrency.png, vram_footprint.png.\n\n"
             + "\n".join(md))
print("\n->", OUT)
