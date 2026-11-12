"""C21: wide-scale population (0.5-1.2) + per-identity sensor orientation offsets (RMS 5 deg) + lever arm (3 cm): downstream rotation estimate with different
calibration-derived 'links'.  3 seeds, paired identity bootstrap.  File stems: ori-<none|est|true>_geom-<none|height|estp6>_s<seed>[_<suffix>]."""
import json, os
import numpy as np
S = (42, 7, 3)
ARMS = {  # label -> (ori, geom, suffix)
    "B0 (no calibration)": ("none", "none", ""),
    "height only (true)": ("none", "height", ""),
    "I2S only (rot estimator)": ("est", "none", "rot"),
    "separate: I2S(rot) + PivotPrint(geo)": ("est", "estp6", "sep"),
    "unified link (uni R + uni geometry)": ("est", "estp6", "uni"),
    "cascade link (rot R + cas geometry)": ("est", "estp6", "cas"),
    "oracle I2S (true R)": ("true", "none", ""),
    "oracle I2S + true height": ("true", "height", ""),
}


def load(ori, geom, suf):
    fs = [f"results/line14/C21/ori-{ori}_geom-{geom}_s{s}" + (f"_{suf}" if suf else "") + ".json" for s in S]
    if not all(os.path.exists(f) for f in fs):
        return None
    R = [json.load(open(f)) for f in fs]
    for r in R:
        assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
    src = "film" if geom != "none" else "b0"
    ids = sorted(R[0]["metrics"][src]["per_identity"])
    return {m: np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R]) for i in ids]) for m in ("sip", "ang", "pos", "mesh")}, R, src


def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed); b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


L = {k: load(*v) for k, v in ARMS.items()}
for k, v in L.items():
    if v:
        print(f"{k:40s}", "  ".join(f"{m} {v[0][m].mean():.3f}" for m in v[0]), " Ang/seed", [round(r["metrics"][v[2]]["ang"], 2) for r in v[1]],
              " joint_err_cm", [round(r["joint_err_cm_final"], 2) for r in v[1]])
    else:
        print(f"{k:40s} (not finished)")
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")


def cmp(a, b):
    if not (L[a] and L[b]):
        return
    row = []
    for m in ("sip", "ang", "pos", "mesh"):
        d = L[a][0][m] - L[b][0][m]; mean, lo, hi = boot(d); row.append(f"{m} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {int((d < 0).sum())}/{len(d)}")
    print(f"{a} - {b}\n   " + "  ".join(row))


B0, H, I, SEP, UNI, CAS = "B0 (no calibration)", "height only (true)", "I2S only (rot estimator)", "separate: I2S(rot) + PivotPrint(geo)", "unified link (uni R + uni geometry)", "cascade link (rot R + cas geometry)"
for a, b in [(H, B0), (I, B0), (SEP, B0), (UNI, B0), (CAS, B0), (SEP, I), (SEP, H), (UNI, SEP), (CAS, SEP), (CAS, UNI), (UNI, I), (CAS, I),
             ("oracle I2S (true R)", B0), ("oracle I2S + true height", "oracle I2S (true R)")]:
    cmp(a, b)
