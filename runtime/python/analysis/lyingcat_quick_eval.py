"""Fast joint-only lying-cat metrics (no mesh skinning) for iterating on rule variants.

python -m analysis.lyingcat_quick_eval v3_420afe8,current,v8 [recordings...]
"""
from __future__ import annotations

import sys

import numpy as np

from analysis.compare_video import fk
from analysis.lyingcat_contact_report import (VARIANTS, _offsets, _twist_deg, align_live, capture, mid_window,
                                              run_rule)
from production.arm_prior import _globals

RECS = ["lyingcat_20261120_230042.jsonl", "lyingcat_20261120_230158.jsonl"]


def _amp(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def metrics(local, off, mid):
    G = np.stack([np.stack(_globals(x)) for x in local])
    P = np.stack([fk(x, off)[1] for x in local])
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    m = {}
    knee_plane = P[:, [4, 5], 1].mean(1)
    m["wrist-knee dy min cm"] = 100 * (P[mid][:, [20, 21], 1] - knee_plane[mid, None]).min()
    m["wrist-knee dy max cm"] = 100 * (P[mid][:, [20, 21], 1] - knee_plane[mid, None]).max()
    for name, j in (("wristL", 20), ("wristR", 21), ("kneeL", 4), ("kneeR", 5)):
        q = P[mid][:, j][:, [0, 2]]          # horizontal: Unity/renderer ground the body vertically
        m[f"{name} drift p95 cm"] = 100 * np.percentile(np.linalg.norm(q - np.median(q, 0), axis=1), 95)
        m[f"{name} step max cm"] = 100 * np.linalg.norm(np.diff(P[:, j][:, [0, 2]], axis=0), axis=1)[mid[1:] & mid[:-1]].max()
    ankle_plane = P[mid][:, [7, 8], 1] - knee_plane[mid, None]
    m["ankle-knee dy absmax cm"] = 100 * np.abs(ankle_plane).max()
    pelvis_up = np.degrees(np.arcsin(np.clip(G[:, 0, 1, 1], -1, 1)))
    m["pelvis pitch amp deg"] = _amp(pelvis_up[mid])
    hip = np.degrees(np.arccos(np.clip(np.einsum("pi,pi->p", unit(np.einsum("pij,j->pi", G[:, 1], off[4])),
                                                  -G[:, 0, :, 1]), -1, 1)))
    m["hipFlexL amp deg"] = _amp(hip[mid])
    lumbar = np.degrees(np.arccos(np.clip(np.einsum("pi,pi->p", G[:, 0, :, 1], G[:, 3, :, 1]), -1, 1)))
    m["lumbar(pelvis-spine1) med deg"] = float(np.median(lumbar[mid]))
    thigh = np.degrees(np.arcsin(np.clip(unit(P[:, 4] - P[:, 1])[:, 1], -1, 1)))
    m["thighElevL range"] = f"{np.percentile(thigh[mid], 5):.0f}..{np.percentile(thigh[mid], 95):.0f}"
    flex = np.degrees(np.arccos(np.clip(np.einsum("pi,pi->p", unit(P[:, 4] - P[:, 1]), unit(P[:, 7] - P[:, 4])), -1, 1)))
    m["kneeFlexL range"] = f"{np.percentile(flex[mid], 5):.0f}..{np.percentile(flex[mid], 95):.0f}"
    for side, u, f, w in (("L", 16, 18, 20), ("R", 17, 19, 21)):
        bend = np.degrees(np.arccos(np.clip(np.einsum("pi,pi->p", unit(P[:, f] - P[:, u]), unit(P[:, w] - P[:, f])), -1, 1)))
        m[f"elbowBend{side} p95 deg"] = float(np.percentile(bend[mid], 95))
        tw = _twist_deg(local[:, f], off[w])[mid]
        m[f"elbowTwist{side} absmax deg"] = float(np.abs(tw).max())
    return m


def main():
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["current"]
    recs = sys.argv[2:] or RECS
    off = _offsets()
    for rec in recs:
        d = capture(rec)
        ref_diag = run_rule(d, "v3_420afe8")[1]
        mid = mid_window(d, ref_diag)
        cols = {"live": d["live"][align_live(d)]}
        for v in variants:
            cols[v] = run_rule(d, v)[0]
        ms = {k: metrics(np.asarray(s, np.float64), off, mid) for k, s in cols.items()}
        print(f"\n== {rec} mid {d['times'][mid].min():.1f}-{d['times'][mid].max():.1f}s")
        print(f"{'':32s}" + "".join(f"{k:>16s}" for k in cols))
        for key in ms["live"]:
            vals = [ms[k][key] for k in cols]
            print(f"{key:32s}" + "".join(f"{v:>16s}" if isinstance(v, str) else f"{v:16.1f}" for v in vals))


if __name__ == "__main__":
    main()
