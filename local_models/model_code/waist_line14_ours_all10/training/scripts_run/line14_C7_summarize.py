"""C7 (wide-population chain, scale 0.5-1.2) summary: 3-seed means + paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
def load(tag, stem, src):
    fs = [f"results/line14/{tag}/{stem}_s{s}.json" for s in S]
    R = [json.load(open(f)) for f in fs]
    for r in R: assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    ids = sorted(R[0]["metrics"][src]["per_identity"])
    return {m: np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R
def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
runs = {"imu-only Poser": ("C7", "imu", "none"), "joints@b0": ("C7", "joints", "b0"), "joints@film": ("C7", "joints", "film"), "joints@gt (upper bound)": ("C7", "joints", "gt"),
        "e2e b0": ("C7e", "e2e_b0", "b0"), "e2e film (PivotPrint)": ("C7e", "e2e_film", "film"), "e2e film (true height)": ("C7e", "e2e_film_height", "film"), "e2e film (height+PivotPrint concat)": ("C7e", "e2e_film_hestp6", "film"), "e2e film (height+PivotPrint residual)": ("C7e", "e2e_film_hrestp6", "film")}
L = {k: load(*v) for k, v in runs.items()}
out = {}
for k, (d, R) in L.items():
    print(f"{k:26s}", "  ".join(f"{m} {d[m].mean():.3f}" for m in d), " Ang/seed", [round(r["metrics"][runs[k][2]]["ang"], 2) for r in R],
          (" joint_err_cm/seed " + str([round(r["joint_err_cm_final"], 2) for r in R])) if k.startswith("e2e") else "")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")
for a, b in [("e2e film (height+PivotPrint residual)", "e2e film (true height)"), ("e2e film (height+PivotPrint concat)", "e2e film (true height)"), ("e2e film (height+PivotPrint residual)", "e2e film (PivotPrint)"), ("e2e film (height+PivotPrint concat)", "e2e film (height+PivotPrint residual)"), ("e2e film (PivotPrint)", "e2e film (true height)"), ("e2e film (true height)", "e2e b0"), ("joints@film", "joints@b0"), ("joints@b0", "imu-only Poser"), ("joints@film", "imu-only Poser"), ("e2e film (PivotPrint)", "e2e b0"), ("e2e b0", "joints@b0"), ("e2e film (PivotPrint)", "joints@film"),
             ("joints@gt (upper bound)", "joints@b0"), ("e2e film (PivotPrint)", "imu-only Poser")]:
    row = {}
    for m in ("sip", "ang", "pos", "mesh"):
        dd = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(dd); row[m] = {"diff": mean, "ci": [lo, hi], "improved": int((dd < 0).sum()), "n": len(dd)}
    out[f"{a} - {b}"] = row
    print(f"{a} - {b}\n   " + "  ".join(f"{m} {row[m]['diff']:+.3f} [{row[m]['ci'][0]:+.3f},{row[m]['ci'][1]:+.3f}] {row[m]['improved']}/{row[m]['n']}" for m in row))
json.dump(out, open("results/line14/C7e/summary_3seed.json", "w"), indent=1)
