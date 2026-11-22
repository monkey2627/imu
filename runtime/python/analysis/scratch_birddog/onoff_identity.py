import sys, numpy as np
import production.frame_chain as fc
import production.live_replay as lr
recs = sys.argv[1:]
for rec in recs:
    out = {}
    for mode in ("on", "off"):
        fc.BIRDDOG_ACTIONS = frozenset({"birddog"}) if mode == "on" else frozenset()
        r = lr.replay_live(rec, simulate=True)
        out[mode] = r.final_local.copy()
        if mode == "on": print(r.summary())
    same = np.array_equal(out["on"], out["off"])
    diff = np.abs(out["on"] - out["off"]).max()
    print(f"RESULT {rec}: bit-identical={same} maxabs={diff:.3g}", flush=True)
