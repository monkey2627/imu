"""C5 (end-to-end) summary: e2e_film vs e2e_b0, 3-seed means and paired identity bootstrap; also C4 if present."""
import json, os, sys
import numpy as np
S = (42, 7, 3)
def load(tag, stem, src):
    fs = [f"results/line14/{tag}/{stem}_s{s}.json" for s in S]
    if not all(os.path.exists(f) for f in fs): return None
    R = [json.load(open(f)) for f in fs]
    for r in R: assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    ids = sorted(R[0]["metrics"][src]["per_identity"])
    return {m: np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R
def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
def cmp(name, A, B):
    print(name)
    for m in ("sip", "ang", "pos", "mesh"):
        d = A[m] - B[m]; mean, lo, hi = boot(d); print(f"   {m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}]  improved {int((d<0).sum())}/{len(d)}")
runs = {"C5 e2e b0": ("C5", "e2e_b0", "b0"), "C5 e2e film": ("C5", "e2e_film", "film"),
        "C4a Poser(train pred_b0) @b0": ("C4a", "joints_pred_b0", "b0"), "C4b Poser(train pred_film) @film": ("C4b", "joints_pred_film", "film"),
        "C4c Poser(train pred_film+2cm) @film": ("C4c", "joints_pred_film", "film"), "C1 Poser @b0": ("C1", "joints", "b0"), "C1 Poser @film": ("C1", "joints", "film")}
L = {k: load(*v) for k, v in runs.items()}
for k, v in L.items():
    if v: print(f"{k:40s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in ("sip", "ang", "pos", "mesh")), "Ang/seed", [round(r["metrics"][runs[k][2]]["ang"], 2) for r in v[1]])
print()
for a, b in [("C5 e2e film", "C5 e2e b0"), ("C4b Poser(train pred_film) @film", "C4a Poser(train pred_b0) @b0"), ("C4c Poser(train pred_film+2cm) @film", "C4a Poser(train pred_b0) @b0"),
             ("C5 e2e b0", "C1 Poser @b0"), ("C5 e2e film", "C1 Poser @film")]:
    if L[a] and L[b]: cmp(f"{a}  -  {b}", L[a][0], L[b][0])
