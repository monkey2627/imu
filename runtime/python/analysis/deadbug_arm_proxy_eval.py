"""Score the dead-bug leg estimator against an arm-reach proxy (no ground truth on real recordings).

In the dead bug the extending leg moves together with the opposite arm reaching overhead: the left leg is
called "extended" by the proxy while the right forearm is within 35 deg of horizontal (and vice versa),
lying supine.  Reports, per side, the share of proxy-extended frames the estimator shows extended/extending
(hit) and the share of proxy-flexed frames it shows extended (false).

    python -m analysis.deadbug_arm_proxy_eval unity_20261116_045157 ...
"""
import json
import sys

import numpy as np
import torch

from production.deadbug_replay import replay
from production.input_recording import resolve_recording_path

MODEL = "waist_imudino_stream_deadbug_simreal"


def score(rec, model=MODEL):
    r = replay(resolve_recording_path(rec), model, leg_mode="auto")
    fr = np.clip(r.represented_frames, 0, r.n_frames - 1)
    L = r.labels
    o = r.calibrated_ori[fr].astype(np.float64)
    supine = o[:, L.index("waist")][:, 1, 2] > 0.7
    arm_elev = {lab: np.degrees(np.arcsin(np.clip((o[:, L.index(lab)] @ axis)[:, 1], -1, 1)))
                for lab, axis in (("lw", np.array([1.0, 0, 0])), ("rw", np.array([-1.0, 0, 0])))}
    out = {}
    for side, arm in (("left", "rw"), ("right", "lw")):
        phase = np.array([d["deadbugLegs"].get(side, {}).get("phase", "") for d in r.diagnostics])
        shown_ext = np.isin(phase, ("extended", "extending"))
        proxy_ext = supine & (np.abs(arm_elev[arm]) < 35)
        proxy_flex = supine & (arm_elev[arm] > 65)
        out[side] = {"proxyExtFrames": int(proxy_ext.sum()),
                     "hit": round(float(shown_ext[proxy_ext].mean()), 3) if proxy_ext.any() else None,
                     "false": round(float(shown_ext[proxy_flex].mean()), 3) if proxy_flex.any() else None}
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    for rec in sys.argv[1:]:
        print(rec, json.dumps(score(rec)), flush=True)
