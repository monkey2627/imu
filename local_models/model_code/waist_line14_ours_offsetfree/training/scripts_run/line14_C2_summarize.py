"""Summary of the chain experiment C1: 3-seed means and paired identity-level bootstrap (SIP / Ang / Pos / Mesh, degrees / cm)."""
import json, os
import numpy as np

S = (42, 7, 3)
RUNS = {  # label -> (file stem, joint source)
    "imu-only Poser": ("imu", "none"),
    "joints(B0 pred)": ("joints", "b0"),
    "joints(FiLM estp6 pred)": ("joints", "film"),
    "joints(GT) upper bound": ("joints", "gt"),
    "geo[estp6] + joints(FiLM estp6)": ("joints_geo_estp6", "film"),
    "geo[estp6] + joints(B0)": ("joints_geo_estp6", "b0"),
    "geo[height,true] + joints(B0)": ("joints_geo_height", "b0"),
    "geo[height,true] + joints(FiLM estp6)": ("joints_geo_height", "film"),
    "geo[estp6] + joints(GT)": ("joints_geo_estp6", "gt"),
}
R = {}
for lab, (stem, src) in RUNS.items():
    fs = [f"results/line14/C2/{stem}_s{s}.json" for s in S]
    if all(os.path.exists(f) for f in fs):
        R[lab] = [json.load(open(f)) for f in fs]
        for r, s in zip(R[lab], S):
            assert r["status"] == "formal_validation_only" and not r["protocol"]["test_accessed"]
ids = sorted(next(iter(R.values()))[0]["metrics"][next(iter(next(iter(R.values()))[0]["metrics"]))]["per_identity"])


def per_id(lab, m):
    stem, src = RUNS[lab]
    return np.array([np.mean([r["metrics"][src]["per_identity"][i][m] for r in R[lab]]) for i in ids])


def boot(d, n=10000, seed=20261021):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


print("3-seed means over validation identities")
for lab in R:
    print(f"{lab:42s}", "  ".join(f"{m} {per_id(lab, m).mean():.3f}" for m in ("sip", "ang", "pos", "mesh")),
          " Ang/seed", [round(r["metrics"][RUNS[lab][1]]["ang"], 2) for r in R[lab]])
pairs = [("joints(FiLM estp6 pred)", "joints(B0 pred)"), ("geo[estp6] + joints(FiLM estp6)", "joints(B0 pred)"),
         ("geo[estp6] + joints(FiLM estp6)", "joints(FiLM estp6 pred)"), ("geo[height,true] + joints(FiLM estp6)", "joints(FiLM estp6 pred)"),
         ("geo[estp6] + joints(FiLM estp6)", "geo[height,true] + joints(FiLM estp6)"), ("joints(B0 pred)", "imu-only Poser"),
         ("joints(GT) upper bound", "joints(B0 pred)")]
out = {}
print("\npaired differences (A - B), negative = A better; identity bootstrap 95% CI")
for a, b in pairs:
    if a in R and b in R:
        row = {}
        for m in ("sip", "ang", "pos", "mesh"):
            d = per_id(a, m) - per_id(b, m)
            mean, lo, hi = boot(d); row[m] = {"diff": mean, "ci": [lo, hi], "improved": int((d < 0).sum()), "n": len(d)}
        out[f"{a} - {b}"] = row
        print(f"{a} - {b}\n   " + "  ".join(f"{m} {row[m]['diff']:+.3f} [{row[m]['ci'][0]:+.3f},{row[m]['ci'][1]:+.3f}] {row[m]['improved']}/{row[m]['n']}" for m in row))
json.dump(out, open("results/line14/C2/summary_3seed.json", "w"), indent=1)
