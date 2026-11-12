"""C23: PivotPrint used as physics (lever-arm compensated joint accelerations, 36 extra input columns), wide-scale + sensor-offset population.
Reads results/line14/C21 (reference arms) and results/line14/C23.  3 seeds, paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {  # label -> (dir, ori, geom, suffix)
    "I2S only (rot)": ("C21", "est", "none", "rot"),
    "separate (I2S + PivotPrint FiLM)": ("C21", "est", "estp6", "sep"),
    "oracle I2S (true R)": ("C21", "true", "none", ""),
    "phys: mean r": ("C23", "est", "none", "lvmean"),
    "phys: estimated r": ("C23", "est", "none", "lvest"),
    "phys: estimated r + FiLM": ("C23", "est", "estp6", "sep_lvest"),
    "phys: true R + true r (upper bound)": ("C23", "true", "none", "lvtrue"),
}


def load(d, ori, geom, suf):
    fs = [f"results/line14/{d}/ori-{ori}_geom-{geom}_s{s}" + (f"_{suf}" if suf else "") + ".json" for s in S]
    if not all(os.path.exists(f) for f in fs):
        return None
    R = [json.load(open(f)) for f in fs]
    for r in R:
        assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    src = "film" if geom != "none" else "b0"
    ids = sorted(R[0]["metrics"][src]["per_identity"])
    return {m: np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R, src


def boot(x, n=10000, seed=20261022):
    rng = np.random.default_rng(seed); b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    if v:
        print(f"{k:38s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"][v[2]]["ang"], 2) for r in v[1]],
              " joint_err_cm", [round(r["joint_err_cm_final"], 2) for r in v[1]])
    else:
        print(f"{k:38s} (not finished)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")


def cmp(a, b):
    if not (L[a] and L[b]):
        return
    row = []
    for m in ("sip", "ang", "pos", "mesh"):
        d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d < 0).sum())}/{len(d)}")
    print(f"{a} - {b}\n   " + "  ".join(row))


for a, b in [("phys: estimated r", "I2S only (rot)"), ("phys: mean r", "I2S only (rot)"), ("phys: estimated r", "phys: mean r"),
             ("phys: estimated r + FiLM", "separate (I2S + PivotPrint FiLM)"), ("phys: estimated r + FiLM", "phys: estimated r"),
             ("phys: true R + true r (upper bound)", "oracle I2S (true R)")]:
    cmp(a, b)
