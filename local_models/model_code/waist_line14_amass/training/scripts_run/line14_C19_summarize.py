"""C19: sensor-to-bone connection (orientation offset I2S estimated by calibration + body geometry PivotPrint), end-to-end from scratch, 3 seeds, paired identity bootstrap.
Arms are named ori-{none,est,true}_geom-{none,estp6}."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {"B0 (none)": ("none", "none"), "geometry only": ("none", "estp6"), "I2S only (est)": ("est", "none"), "link (est I2S + geometry)": ("est", "estp6"),
        "oracle I2S": ("true", "none"), "oracle link (true I2S + geometry)": ("true", "estp6")}
def load(ori, geom):
    fs = [f"results/line14/C19/ori-{ori}_geom-{geom}_s{s}.json" for s in S]
    if not all(os.path.exists(f) for f in fs): return None
    R = [json.load(open(f)) for f in fs]
    for r in R: assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    src = "film" if geom == "estp6" else "b0"
    ids = sorted(R[0]["metrics"][src]["per_identity"])
    return {m: np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R, src
def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    if v: print(f"{k:36s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"][v[2]]["ang"], 2) for r in v[1]], " joint_err_cm", [round(r["joint_err_cm_final"], 2) for r in v[1]])
    else: print(f"{k:36s} (not finished)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")
def cmp(a, b):
    if not (L[a] and L[b]): return
    row = []
    for m in ("sip", "ang", "pos", "mesh"):
        d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d<0).sum())}/{len(d)}")
    print(f"{a} - {b}\n   " + "  ".join(row))
for a, b in [("geometry only", "B0 (none)"), ("I2S only (est)", "B0 (none)"), ("link (est I2S + geometry)", "B0 (none)"), ("link (est I2S + geometry)", "I2S only (est)"),
             ("link (est I2S + geometry)", "geometry only"), ("oracle I2S", "B0 (none)"), ("oracle link (true I2S + geometry)", "B0 (none)"),
             ("oracle link (true I2S + geometry)", "oracle I2S"), ("link (est I2S + geometry)", "oracle link (true I2S + geometry)")]:
    cmp(a, b)
print("\nshare of the oracle gain recovered by the calibration estimate (identity-level mean differences)")
for m in ("sip", "ang", "pos", "mesh"):
    if L["I2S only (est)"] and L["oracle I2S"] and L["link (est I2S + geometry)"] and L["oracle link (true I2S + geometry)"]:
        b0 = L["B0 (none)"][0][m].mean()
        print(f"  {m}: I2S only {(b0 - L['I2S only (est)'][0][m].mean()) / (b0 - L['oracle I2S'][0][m].mean()):.2f} of oracle I2S;  "
              f"link {(b0 - L['link (est I2S + geometry)'][0][m].mean()) / (b0 - L['oracle link (true I2S + geometry)'][0][m].mean()):.2f} of oracle link")
