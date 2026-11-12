"""C10: ADULT population end-to-end chain (AMASS adults, no scaling): B0 / PivotPrint / true height / height+PivotPrint (concat, residual). 3-seed means + paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
RUNS = {"e2e b0": ("C5", "e2e_b0", "b0"), "e2e film (PivotPrint)": ("C5", "e2e_film", "film"), "e2e film (true height)": ("C10", "e2e_film_height", "film"),
        "e2e film (height+PivotPrint concat)": ("C10", "e2e_film_hestp6", "film"), "e2e film (height+PivotPrint residual)": ("C10", "e2e_film_hrestp6", "film"), "e2e film (TRUE full geometry r2, 24-d)": ("C16", "e2e_film_r2", "film")}
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
L = {k: load(*v) for k, v in RUNS.items()}
for k, v in L.items():
    if v: print(f"{k:40s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"][RUNS[k][2]]["ang"], 2) for r in v[1]], " joint_err_cm", [round(r["joint_err_cm_final"], 2) for r in v[1]])
    else: print(f"{k:40s} (not finished)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")
for a, b in [("e2e film (TRUE full geometry r2, 24-d)", "e2e b0"), ("e2e film (TRUE full geometry r2, 24-d)", "e2e film (true height)"), ("e2e film (TRUE full geometry r2, 24-d)", "e2e film (PivotPrint)"), ("e2e film (true height)", "e2e b0"), ("e2e film (PivotPrint)", "e2e film (true height)"), ("e2e film (height+PivotPrint concat)", "e2e film (true height)"),
             ("e2e film (height+PivotPrint residual)", "e2e film (true height)"), ("e2e film (height+PivotPrint residual)", "e2e film (height+PivotPrint concat)"),
             ("e2e film (height+PivotPrint residual)", "e2e film (PivotPrint)"), ("e2e film (height+PivotPrint concat)", "e2e b0")]:
    if L[a] and L[b]:
        row = []
        for m in ("sip", "ang", "pos", "mesh"):
            d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d<0).sum())}/{len(d)}")
        print(f"{a} - {b}\n   " + "  ".join(row))
