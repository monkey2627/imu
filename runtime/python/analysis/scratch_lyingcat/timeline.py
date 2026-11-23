import sys, json, numpy as np
from analysis.scratch_lyingcat.signal import pos, elev
from production.arm_prior import _globals

def metrics(L):
    G = np.array([_globals(l.astype(np.float64)) for l in L]); P = np.array([pos(g) for g in G])
    out = {"pelv": elev(G[:, 0, :, 1]), "sp1": elev(G[:, 3, :, 1]), "sp2": elev(G[:, 6, :, 1]), "sp3": elev(G[:, 9, :, 1]),
           "neck": elev(G[:, 12, :, 1])}
    # arch: height of spine2 joint (6) / spine3 joint (9) above the pelvis->neck chord (cm)
    a, b = P[:, 0], P[:, 12]
    for j in (6, 9):
        s = np.clip(((P[:, j] - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1), 0, 1)
        out[f"arch{j}"] = 100 * (P[:, j] - (a + s[:, None] * (b - a)))[:, 1]
    out["trunk"] = out["sp3"] - out["pelv"]       # total spine bend in the sagittal plane
    return out, P

if __name__ == "__main__":
    stem = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "prod"
    d = np.load(f"analysis/scratch_lyingcat/cap_{stem}_{tag}.npz")
    t = d["t"]; W = list(d["labels"]).index("waist")
    wy = elev(d["measured"][:, W, :, 1])
    raw, _ = metrics(d["raw"]); fin, _ = metrics(d["final"])
    print(f"{stem} [{tag}] per second means: waistY | raw pelv sp1 sp3 trunk arch9 | final pelv sp1 sp3 trunk arch9")
    for s in range(0, int(t[-1]) + 1):
        m = (t >= s) & (t < s + 1)
        if not m.any(): continue
        f = lambda o, k: o[k][m].mean()
        print(f"{s:3d} {wy[m].mean():6.1f} | {f(raw,'pelv'):6.1f} {f(raw,'sp1'):6.1f} {f(raw,'sp3'):6.1f} {f(raw,'trunk'):6.1f} {f(raw,'arch9'):5.1f}"
              f" | {f(fin,'pelv'):6.1f} {f(fin,'sp1'):6.1f} {f(fin,'sp3'):6.1f} {f(fin,'trunk'):6.1f} {f(fin,'arch9'):5.1f}")
