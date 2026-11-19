"""Second pass: resolve GGUF tensors the first pass could not pair, and re-check transform mismatches with FP16 tolerance.

usage: python compare_weights2.py <hf_snapshot_dir> <gguf_file> <first_pass_log> <out_log>
For each unresolved GGUF tensor in layer N (or the MTP layer), search the HF tensors of the same layer for one tensor or a
combination of tensors (concatenated, possibly reordered) whose values equal the GGUF tensor's values after BF16->FP16.
Equality is tested on sorted values (order-independent), so concatenation and row reordering are covered; a value change is not.
"""
import glob, itertools, os, re, sys
import numpy as np
import torch
from safetensors import safe_open
import gguf

hf_dir, gguf_path, first_log, out_path = sys.argv[1:5]
out = open(out_path, "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

reader = gguf.GGUFReader(gguf_path)
g = {t.name: t for t in reader.tensors}
hf = {}
for f in sorted(glob.glob(os.path.join(hf_dir, "*.safetensors"))):
    h = safe_open(f, "pt")
    for k in h.keys():
        hf[k] = (h, k)

def garr(t):
    a = np.array(t.data)
    a = a.view(np.float16) if t.tensor_type == gguf.GGMLQuantizationType.F16 else a.view(np.float32)
    return a.reshape(-1), t.tensor_type == gguf.GGMLQuantizationType.F16

def harr(k):
    return hf[k][0].get_tensor(hf[k][1]).reshape(-1).to(torch.float32).numpy().astype(np.float64)

def cast(x, is16):
    return x.astype(np.float16) if is16 else x.astype(np.float32)

todo = []
for line in open(first_log):
    m = re.search(r"\[(no HF source found|MISMATCH)\] (\S+)", line)
    if m:
        todo.append((m.group(1), m.group(2)))
log(f"unresolved from first pass: {len(todo)}")

N_MAIN = 24  # main transformer blocks 0..23; blk.24 = MTP (next-token prediction) layer
res = {"resolved_exact": 0, "resolved_within_1ulp": 0, "UNRESOLVED": 0}
for kind, name in todo:
    a, is16 = garr(g[name])
    m = re.match(r"blk\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    if layer is not None and layer < N_MAIN:
        pool = [k for k in hf if re.search(rf"language_model\.layers\.{layer}\.", k)]
    elif layer == N_MAIN:
        pool = [k for k in hf if k.startswith("mtp.")]
    else:
        pool = [k for k in hf if "visual" not in k and "layers." not in k]

    found = None
    if kind == "MISMATCH":  # A_log -> -exp(A_log): compare with 1-ulp tolerance (exp rounding)
        src = [k for k in pool if k.endswith("A_log")]
        if src:
            r = cast(-np.exp(harr(src[0])), is16).astype(np.float64)
            av = a.astype(np.float64)
            ulp = np.spacing(np.abs(r).astype(np.float16 if is16 else np.float32)).astype(np.float64)
            if np.all(np.abs(r - av) <= ulp):
                found = ("within_1ulp", [src[0]], "-exp(A_log)", float(np.abs(r - av).max()))
    else:
        sizes = {k: hf[k][0].get_slice(hf[k][1]).get_shape() for k in pool}
        nel = {k: int(np.prod(s)) for k, s in sizes.items()}
        target = np.sort(a.astype(np.float64))
        cands = [k for k in pool if nel[k] <= a.size]
        for r_ in (1, 2, 3, 4):
            for combo in itertools.combinations(cands, r_):
                if sum(nel[k] for k in combo) != a.size:
                    continue
                v = np.concatenate([harr(k) for k in combo])
                for tname, fn in (("as is", lambda x: x), ("1+w", lambda x: x + 1)):
                    if np.array_equal(np.sort(cast(fn(v), is16).astype(np.float64)), target):
                        found = ("exact", list(combo), tname, 0.0); break
                if found: break
            if found: break
    if found:
        res["resolved_exact" if found[0] == "exact" else "resolved_within_1ulp"] += 1
        log(f"  OK   {name}  <=  {' + '.join(found[1])}  [{found[2]}{'' if found[0]=='exact' else f', max diff {found[3]:.2e} (<=1 ulp)'}]")
    else:
        res["UNRESOLVED"] += 1
        log(f"  ???  {name}  n={a.size}  (no matching source among {len(pool)} HF tensors of that layer)")

log("\n===== SECOND PASS SUMMARY")
for k, v in res.items():
    log(f"  {k:24s} {v}")
out.close()
