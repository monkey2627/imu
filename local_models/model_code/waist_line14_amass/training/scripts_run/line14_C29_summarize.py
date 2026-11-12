"""C29: lever-arm compensation features WITHOUT orientation correction (ori=none).  Tests whether orientation correction and lever-arm compensation are coupled.
Reads C21 (B0, rot), C23 (est R + lever), C29 (no R + lever).  3 seeds, paired identity bootstrap, validation identities only."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {
    "B0": ("C21", "none", "none", ""),
    "I2S only (rot)": ("C21", "est", "none", "rot"),
    "noR + est r": ("C29", "none", "none", "lvest_noR"),
    "noR + mean r": ("C29", "none", "none", "lvmean_noR"),
    "estR + est r": ("C23", "est", "none", "lvest"),
    "estR + mean r": ("C23", "est", "none", "lvmean"),
}
M4 = ("sip", "ang", "pos", "mesh")


def load(d, ori, geom, suf):
    fs = [f"results/line14/{d}/ori-{ori}_geom-{geom}_s{s}" + (f"_{suf}" if suf else "") + ".json" for s in S]
    if not all(os.path.exists(f) for f in fs):
        return None
    R = [json.load(open(f)) for f in fs]
    for r in R:
        assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    ids = sorted(R[0]["metrics"]["b0"]["per_identity"])
    return {m: np.array([np.mean([r["metrics"]["b0"]["per_identity"][i][m] for r in R]) for i in ids]) for m in M4}, R


def boot(x, n=10000, seed=20261024):
    rng = np.random.default_rng(seed); b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    print(f"{k:16s}", ("  ".join(f"{m} {v[0][m].mean():.3f}" for m in M4) + "  Ang/seed " + str([round(r['metrics']['b0']['ang'], 2) for r in v[1]])) if v else "(missing)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")


def show(name, d):
    print(f"{name}\n   " + "  ".join(f"{m} {boot(d[m])[0]:+.3f} [{boot(d[m])[1]:+.3f},{boot(d[m])[2]:+.3f}]" + (f" {int((d[m] < 0).sum())}/{len(d[m])}" if True else "") for m in M4))


def diff(a, b):
    return {m: L[a][0][m] - L[b][0][m] for m in M4}


for a, b in [("noR + est r", "B0"), ("noR + mean r", "B0"), ("noR + est r", "noR + mean r"), ("estR + est r", "noR + est r"), ("estR + est r", "I2S only (rot)"), ("estR + mean r", "I2S only (rot)")]:
    show(f"{a} - {b}", diff(a, b))
print("\ndifference of differences (feature gain WITHOUT R correction) - (feature gain WITH R correction); >0 means the feature helps LESS without R")
for tag, noR, withR in (("est r", "noR + est r", "estR + est r"), ("mean r", "noR + mean r", "estR + mean r")):
    g0 = diff(noR, "B0"); g1 = diff(withR, "I2S only (rot)")
    show(f"[{tag}]  (noR+feat - B0) - (estR+feat - I2S only)", {m: g0[m] - g1[m] for m in M4})
