import numpy as np
from analysis.scratch_lyingcat.rule_only import run
from analysis.scratch_lyingcat.signal import pos
from analysis.scratch_lyingcat.variant_heights import WIN
from analysis.scratch_lyingcat.timeline import metrics
from production.arm_prior import _globals
def drift(stem, **kw):
    t, L, _ = run(stem, **kw); a, b = WIN[stem]; m = (t > a) & (t < b)
    P = np.array([pos(_globals(l)) for l in L[m]]) * 100
    kn = (P[:, 4, 1] + P[:, 5, 1]) / 2
    out = []
    for j, n in ((20, "wL"), (21, "wR")):
        h = P[:, j, [0, 2]]; d = np.linalg.norm(h - np.median(h, 0), axis=1)
        out.append(f"{n} up p95 {np.percentile(P[:, j, 1] - kn, 95):+.1f} drift p95 {np.percentile(d, 95):.1f}")
    arch = metrics(L[m])[0]["arch9"]
    return " | ".join(out) + f" | arch p5..p95 {np.percentile(arch,5):+.1f}..{np.percentile(arch,95):+.1f}"
