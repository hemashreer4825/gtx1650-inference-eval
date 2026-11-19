"""Summarise torch-profiler traces from diag_vllm.py: GPU kernel time by kernel name, batch 1 vs batch 2.
usage: python scripts/diagnosis/analyze_trace.py logs/profile
"""
import collections, glob, gzip, json, os, sys

root = sys.argv[1]
tables = {}
for bs in ("bs1", "bs2"):
    files = [f for f in glob.glob(os.path.join(root, bs, "**", "*"), recursive=True) if f.endswith((".json", ".json.gz", ".pt.trace.json.gz"))]
    tot = collections.Counter(); cnt = collections.Counter(); cpu_wall = 0
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        try:
            d = json.load(op(f, "rt"))
        except Exception as e:
            print("skip", f, e); continue
        for ev in d.get("traceEvents", []):
            if ev.get("ph") != "X":
                continue
            if ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                name = ev["name"][:90]
                tot[name] += ev.get("dur", 0); cnt[name] += 1
    tables[bs] = (tot, cnt)
    gpu_ms = sum(tot.values()) / 1000
    print(f"\n=== {bs}: {len(files)} trace file(s), total GPU kernel time {gpu_ms:.1f} ms")
    for name, us in tot.most_common(15):
        print(f"  {us/1000:9.1f} ms  {cnt[name]:6d}x  {name}")

if len(tables) == 2:
    (t1, _), (t2, _) = tables["bs1"], tables["bs2"]
    print("\n=== biggest increase bs1 -> bs2 (ms)")
    diff = {k: t2.get(k, 0) - t1.get(k, 0) for k in set(t1) | set(t2)}
    for k, v in sorted(diff.items(), key=lambda x: -x[1])[:12]:
        print(f"  +{v/1000:9.1f} ms  (bs1 {t1.get(k,0)/1000:.1f} -> bs2 {t2.get(k,0)/1000:.1f})  {k}")
