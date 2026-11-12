"""Adult population, Joints level: does calibration (PivotPrint) add to a known height?  All Joints are FiLM, same budget; B0 = no geometry.
Metric = mean 24-joint root-relative position error (cm) per validation identity, 3-seed mean, paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
RUNS = {"B0 (no geometry)": ("E5", "r1_b0_s{s}"), "height (true) FiLM": ("C9a", "height_oracle_film_s{s}"), "PivotPrint est pca6 FiLM (E18)": ("E18", "estp6_oracle_film_s{s}"),
        "height + PivotPrint concat": ("C9a", "hestp6_oracle_film_s{s}"), "height + PivotPrint residual": ("C9ar", "hrestp6_oracle_film_s{s}")}
def load(tag, pat):
    fs = [f"results/line14/{tag}/{pat.format(s=s)}.json" for s in S]
    if not all(os.path.exists(f) for f in fs): return None
    R = [json.load(open(f)) for f in fs]
    ids = sorted(R[0]["metrics"]["per_identity"])
    g = lambda r, i, k: r["metrics"]["per_identity"][i]["metrics"][k]["mean"]
    return {k: np.array([np.mean([g(r, i, k) for r in R]) for i in ids]) for k in ("all", "elbow", "knee", "shoulder", "hip")}, R
def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
L = {k: load(*v) for k, v in RUNS.items()}
for k, v in L.items():
    if v: print(f"{k:34s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " all/seed", [round(r['metrics']['aggregate']['all']['mean'], 3) for r in v[1]])
    else: print(f"{k:34s} (not finished)")
print("\npaired differences (A - B), cm, negative = A better; identity bootstrap 95% CI")
for a, b in [("height (true) FiLM", "B0 (no geometry)"), ("PivotPrint est pca6 FiLM (E18)", "height (true) FiLM"), ("height + PivotPrint concat", "height (true) FiLM"),
             ("height + PivotPrint residual", "height (true) FiLM"), ("height + PivotPrint residual", "height + PivotPrint concat"), ("height + PivotPrint residual", "PivotPrint est pca6 FiLM (E18)")]:
    if L[a] and L[b]:
        row = []
        for m in ("all", "elbow", "knee", "shoulder", "hip"):
            d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d<0).sum())}/{len(d)}")
        print(f"{a} - {b}\n   " + "  ".join(row))
