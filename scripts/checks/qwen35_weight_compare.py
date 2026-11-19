"""Compare every weight tensor of the converted Qwen3.5-0.8B F16 GGUF with the original HF safetensors (BF16).

usage: python compare_weights.py <hf_snapshot_dir> <gguf_file> <out_log>
Read-only. CPU only. For each GGUF tensor: find the HF tensor it came from (llama.cpp's own name map),
convert the HF BF16 values to FP16 exactly as the converter does, and compare element by element.
If values are not identical as stored, a few known converter transforms are tried and reported by name.
"""
import json, sys, os, glob
import numpy as np
import torch
from safetensors import safe_open
import gguf

hf_dir, gguf_path, out_path = sys.argv[1:4]
out = open(out_path, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

reader = gguf.GGUFReader(gguf_path)
arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
n_blocks = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
log(f"GGUF: {gguf_path}\narch={arch} blocks={n_blocks} tensors={len(reader.tensors)}")
g = {t.name: t for t in reader.tensors}

# HF tensors (lazy)
files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
hf = {}
for f in files:
    h = safe_open(f, "pt")
    for k in h.keys():
        hf[k] = (h, k)
log(f"HF: {hf_dir}\nsafetensors files={len(files)} tensors={len(hf)}")

tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH[arch.upper()] if arch.upper() in gguf.MODEL_ARCH.__members__ else
                                next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch), n_blocks)

def hf_to_gguf_name(name):
    for strip in ("model.language_model.", "language_model.model.", "model."):
        pass
    cands = [name, name.replace("model.language_model.", "model."), name.replace("language_model.", "")]
    for c in cands:
        n = tmap.get_name(c, try_suffixes=(".weight", ".bias"))
        if n:
            return n
    return None

def gguf_array(t):
    a = np.array(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.F16:
        a = a.view(np.float16)
    elif t.tensor_type == gguf.GGMLQuantizationType.F32:
        a = a.view(np.float32)
    else:
        return None
    return a.astype(np.float64).reshape(-1)

# map GGUF name -> list of HF names
by_gguf = {}
unmapped_hf = []
for k in hf:
    n = hf_to_gguf_name(k)
    if n is None:
        unmapped_hf.append(k)
    else:
        by_gguf.setdefault(n, []).append(k)

TRANSFORMS = {
    "identical": lambda x: x,
    "plus_1 (norm stored as 1+w)": lambda x: x + 1,
    "minus_exp (A_log -> -exp(A))": lambda x: -np.exp(x),
}

counts = {"exact": 0, "transform": 0, "concat/permuted (elementwise multiset equal)": 0, "MISMATCH": 0,
          "not_float": 0, "no_hf_source": 0}
total_elems = 0
for name, t in g.items():
    a = gguf_array(t)
    if a is None:
        counts["not_float"] += 1; log(f"  [not F16/F32] {name} type={t.tensor_type.name}"); continue
    total_elems += a.size
    srcs = by_gguf.get(name, [])
    if not srcs:
        counts["no_hf_source"] += 1; log(f"  [no HF source found] {name} shape={list(t.shape)} n={a.size}"); continue
    h_parts = [hf[s][0].get_tensor(hf[s][1]) for s in srcs]
    hcat = torch.cat([p.reshape(-1) for p in h_parts]).to(torch.float32)
    # converter path: BF16 -> F32 -> F16 (for F16 outputs); F32 outputs keep F32
    ref16 = hcat.to(torch.float16).to(torch.float64).numpy() if a.dtype == np.float64 and t.tensor_type == gguf.GGMLQuantizationType.F16 else hcat.to(torch.float64).numpy()
    ref32 = hcat.to(torch.float64).numpy()
    if ref16.size != a.size:
        counts["MISMATCH"] += 1; log(f"  [SIZE MISMATCH] {name} gguf={a.size} hf={ref16.size} from {srcs}"); continue
    done = False
    for tn, fn in TRANSFORMS.items():
        r = fn(ref32)
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            r = r.astype(np.float16).astype(np.float64)
        else:
            r = r.astype(np.float32).astype(np.float64)
        if np.array_equal(r, a):
            if tn == "identical":
                counts["exact"] += 1
            else:
                counts["transform"] += 1; log(f"  [match after {tn}] {name}")
            done = True; break
    if not done:
        r = ref16 if t.tensor_type == gguf.GGMLQuantizationType.F16 else ref32
        if np.array_equal(np.sort(r), np.sort(a)):
            counts["concat/permuted (elementwise multiset equal)"] += 1
            log(f"  [same values, different order (reshaped/permuted by converter)] {name} from {srcs}")
        else:
            d = np.abs(r - a); counts["MISMATCH"] += 1
            log(f"  [MISMATCH] {name} from {srcs} max_abs_diff={d.max():.3e} mean_abs_diff={d.mean():.3e}")

used = {s for v in by_gguf.values() for s in v if v and any(True for _ in [0])}
hf_not_in_gguf = [k for k in hf if hf_to_gguf_name(k) not in g]
log("\n===== SUMMARY")
log(f"GGUF tensors: {len(g)}  elements: {total_elems:,}")
for k, v in counts.items():
    log(f"  {k:48s} {v}")
log(f"HF tensors with no GGUF counterpart: {len(hf_not_in_gguf)}")
groups = {}
for k in hf_not_in_gguf:
    p = k.split(".")[0] + "." + (k.split(".")[1] if len(k.split(".")) > 1 else "")
    groups[p] = groups.get(p, 0) + 1
for p, c in sorted(groups.items()):
    log(f"    {p}*  ({c} tensors)")
out.close()
