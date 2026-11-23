"""Per cat/cow phase layer table (Q3/Q4).  Phase windows: 230158 from waist-pitch plateaus (confirmed by the
ZUPT signs), 004720 from the waist vertical translation (phases.py), because its pitch hardly moves."""
import json, sys, numpy as np
from analysis.scratch_lyingcat.signal import elev
from analysis.scratch_lyingcat.timeline import metrics
from production.bridge_contact import _rest_offsets
OFF = _rest_offsets()
PHASES = {
    "lyingcat_20261120_230158": [("cow", 11.0, 14.0), ("cat", 15.5, 18.0), ("cow", 21.0, 24.0), ("cat", 25.0, 27.0),
                                 ("cow", 30.0, 33.0), ("cat", 33.5, 36.0), ("cow", 38.0, 41.0), ("cat", 41.5, 43.5),
                                 ("cow", 45.5, 48.0), ("cat", 49.0, 51.0)],
    "lyingcat_20261123_004720": [("cow", 14.0, 17.4), ("cat", 21.2, 23.0), ("cow", 26.3, 28.3), ("cat", 32.1, 33.0),
                                 ("cow", 36.2, 37.2), ("cat", 40.4, 41.6), ("cow", 44.8, 45.4), ("cat", 48.2, 49.7),
                                 ("cow", 52.8, 53.1), ("cat", 55.5, 56.1)],
}
from analysis.scratch_lyingcat.phases import phases as _zupt_phases
for _stem, _t0 in (("lyingcat_20261123_044239", 9.0), ("lyingcat_20261123_044417", 21.0)):
    # pose reached = end of each ZUPT motion segment (up = cat, down = cow); window -0.3..+0.5 s
    PHASES[_stem] = [("cat" if dy > 0 else "cow", e - 0.3, e + 0.5) for _s, e, dy in _zupt_phases(_stem, t0=_t0)[0]
                     if e < (47.0 if "044239" in _stem else 57.0)]
BASE = (4.0, 7.5)     # cat_neutral start hold


def layer_signals(stem, tag="prod"):
    d = np.load(f"analysis/scratch_lyingcat/cap_{stem}_{tag}.npz", allow_pickle=True)
    t, tt = d["t"], d["tick_t"]
    raw, _ = metrics(d["raw"]); fin, P = metrics(d["final"])
    diag = [json.loads(x) if x != "null" else None for x in d["diag"]]
    sp = np.array([dg["spineDeg"][:2] if dg else [np.nan, np.nan] for dg in diag])
    S = {"waist uncal": (tt, elev(d["uncal"][:, 4, :, 1])),
         "waist calib": (t, elev(d["measured"][:, 4, :, 1])),
         "net pelvis": (t, raw["pelv"]), "net sp3": (t, raw["sp3"]), "net trunk": (t, raw["trunk"]), "net arch cm": (t, raw["arch9"]),
         "rule sp2+3 deg": (t, sp.sum(1)),
         "out pelvis": (t, fin["pelv"]), "out sp1": (t, fin["sp1"]), "out sp3": (t, fin["sp3"]), "out trunk": (t, fin["trunk"]), "out arch cm": (t, fin["arch9"]),
         "lw forearm": (t, elev(np.einsum("nij,j->ni", d["measured"][:, 0], OFF[20]))),
         "rw forearm": (t, elev(np.einsum("nij,j->ni", d["measured"][:, 1], OFF[21])))}
    return S


def table(stem, tag="prod", keys=None):
    S = layer_signals(stem, tag); ph = PHASES[stem]
    keys = keys or list(S)
    mean = lambda k, a, b: S[k][1][(S[k][0] >= a) & (S[k][0] < b)].mean()
    print(f"\n== {stem} [{tag}]   values: start-hold | per phase mean ... | cat-cow (mean of adjacent pairs)")
    print(" " * 16 + "hold " + " ".join(f"{p[0]}{p[1]:4.0f}" for p in ph))
    res = {}
    for k in keys:
        v = [mean(k, a, b) for _, a, b in ph]
        diffs = [(v[i] - v[i + 1]) * (1 if ph[i][0] == "cat" else -1) for i in range(len(v) - 1)]
        res[k] = diffs
        print(f"{k:15s} {mean(k, *BASE):5.1f} " + " ".join(f"{x:7.1f}" for x in v) + f" | {np.mean(diffs):+6.1f} (min {np.min(diffs):+5.1f})")
    return res


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "prod"
    for stem in PHASES: table(stem, tag)
