"""C24: controls and arms-only variants of the lever-arm-compensation features (PivotPrint as physics).  3 seeds, paired identity bootstrap."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {  # label -> (dir, ori, geom, suffix)
    "I2S only (rot)": ("C21", "est", "none", "rot"),
    "oracle I2S (true R)": ("C21", "true", "none", ""),
    "est R + mean r (all)": ("C23", "est", "none", "lvmean"),
    "est R + est r (all)": ("C23", "est", "none", "lvest"),
    "true R + true r (all)": ("C23", "true", "none", "lvtrue"),
    "true R + mean r (all)": ("C24", "true", "none", "lvmean_trueR"),
    "est R + true r (all)": ("C24", "est", "none", "lvtrue_estR"),
    "est R + mean r (arms only)": ("C24", "est", "none", "lvmean_arms"),
    "est R + est r (arms only)": ("C24", "est", "none", "lvest_arms"),
    "est R + est r (legs only)": ("C24", "est", "none", "lvest_legs"),
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


def boot(x, n=10000, seed=20261023):
    rng = np.random.default_rng(seed); b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    if v:
        print(f"{k:30s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"][v[2]]["ang"], 2) for r in v[1]])
    else:
        print(f"{k:30s} (not finished)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")


def cmp(a, b):
    if not (L[a] and L[b]):
        return
    row = []
    for m in ("sip", "ang", "pos", "mesh"):
        d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d < 0).sum())}/{len(d)}")
    print(f"{a} - {b}\n   " + "  ".join(row))


for a, b in [("true R + mean r (all)", "oracle I2S (true R)"), ("true R + true r (all)", "true R + mean r (all)"), ("est R + true r (all)", "est R + est r (all)"),
             ("est R + est r (all)", "est R + mean r (all)"), ("est R + est r (arms only)", "est R + est r (all)"), ("est R + est r (arms only)", "I2S only (rot)"),
             ("est R + mean r (arms only)", "I2S only (rot)"), ("est R + est r (legs only)", "I2S only (rot)"), ("est R + mean r (arms only)", "est R + est r (arms only)"),
             ("est R + est r (arms only)", "est R + est r (legs only)")]:
    cmp(a, b)
