"""C28: orientation-estimation precision -> downstream accuracy.  True R_off + a fixed per-identity random rotation (mean angle error 1.0 / 1.7 / 2.5 deg), orientation
correction only (no geometry, no lever features), wide-scale + sensor-offset population.  3 seeds, paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {  # label -> (dir, ori, geom, suffix)
    "B0 (no calibration)": ("C21", "none", "none", ""),
    "oracle I2S (0 deg)": ("C21", "true", "none", ""),
    "true R + noise 1.0 deg": ("C28", "true", "none", "n10"),
    "true R + noise 1.7 deg": ("C28", "true", "none", "n17"),
    "true R + noise 2.5 deg": ("C28", "true", "none", "n25"),
    "estimated I2S (rot estimator, 1.7 deg)": ("C21", "est", "none", "rot"),
}


def load(d, ori, geom, suf):
    fs = [f"results/line14/{d}/ori-{ori}_geom-{geom}_s{s}" + (f"_{suf}" if suf else "") + ".json" for s in S]
    if not all(os.path.exists(f) for f in fs):
        return None
    R = [json.load(open(f)) for f in fs]
    for r in R:
        assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    ids = sorted(R[0]["metrics"]["b0"]["per_identity"])
    return {m: np.array([np.mean([r["metrics"]["b0"]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R


def boot(x, n=10000, seed=20261023):
    rng = np.random.default_rng(seed); b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    if v:
        print(f"{k:42s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"]["b0"]["ang"], 2) for r in v[1]])
    else:
        print(f"{k:42s} (not finished)")
b0 = L["B0 (no calibration)"][0]; orc = L["oracle I2S (0 deg)"][0]
print("\nfraction of the oracle gain (B0 -> oracle) retained")
for k in ("true R + noise 1.0 deg", "true R + noise 1.7 deg", "true R + noise 2.5 deg", "estimated I2S (rot estimator, 1.7 deg)"):
    if L[k]:
        print(f"  {k:42s}", "  ".join(f"{m} {(b0[m].mean() - L[k][0][m].mean()) / (b0[m].mean() - orc[m].mean()):.2f}" for m in b0))
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")


def cmp(a, b):
    if not (L[a] and L[b]):
        return
    row = []
    for m in ("sip", "ang", "pos", "mesh"):
        d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d < 0).sum())}/{len(d)}")
    print(f"{a} - {b}\n   " + "  ".join(row))


for a, b in [("true R + noise 1.0 deg", "oracle I2S (0 deg)"), ("true R + noise 1.7 deg", "oracle I2S (0 deg)"), ("true R + noise 2.5 deg", "oracle I2S (0 deg)"),
             ("true R + noise 1.7 deg", "estimated I2S (rot estimator, 1.7 deg)"), ("true R + noise 1.0 deg", "true R + noise 1.7 deg"), ("true R + noise 2.5 deg", "true R + noise 1.7 deg"),
             ("true R + noise 1.0 deg", "estimated I2S (rot estimator, 1.7 deg)")]:
    cmp(a, b)
