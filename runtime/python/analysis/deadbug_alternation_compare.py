"""Per-leg agreement between network thigh angle and the motion-evidence estimator state (all confidences)."""
import json, sys
import numpy as np, torch
from analysis.deadbug_model_compare import load, network_legs, MODEL_ID
from production.deadbug_replay import replay
from production.input_recording import resolve_recording_path
torch.set_grad_enabled(False); torch.set_num_threads(8)
CK = "../../local_models/checkpoints/"
models = {"limbmix": CK + "waist_imudino_stream_deadbug_limbmix", "sim1115": CK + "waist_imudino_stream_deadbug_sim",
          "rehabv5": CK + "waist_imudino_stream_deadbug_rehabv5", "simreal": CK + "waist_imudino_stream_deadbug_simreal"}
import sys
recs = sys.argv[1:] or ["unity_20261113_232942", "unity_20261113_233607", "unity_20261113_233649"]
out = {}
for name, path in models.items():
    out[name] = {}
    for rec in recs:
        r = replay(resolve_recording_path(rec), MODEL_ID, leg_mode="auto", loaded=load(path))
        net = network_legs(r)
        row = {}
        for side in ("left", "right"):
            th, kn = net[side]
            ph = np.array([d["deadbugLegs"].get(side, {}).get("phase", "") for d in r.diagnostics])
            cf = np.array([d["deadbugLegs"].get(side, {}).get("confidence", 0) for d in r.diagnostics])
            m = np.isin(ph, ("flexed", "extended")) & (cf >= 0.5)
            ref = (ph[m] == "flexed").astype(float)
            ext_ok = ((th[m] < 45) & (kn[m] < 40))[ref == 0].mean() if (ref == 0).any() else None
            row[side] = dict(frames=int(m.sum()), ext_frames=int((ref == 0).sum()),
                             corr_thigh_vs_flexed=round(float(np.corrcoef(th[m], ref)[0, 1]), 3),
                             thigh_med_flexed=round(float(np.median(th[m][ref == 1])), 1),
                             thigh_med_extended=round(float(np.median(th[m][ref == 0])), 1) if ext_ok is not None else None,
                             knee_med_extended=round(float(np.median(kn[m][ref == 0])), 1) if ext_ok is not None else None,
                             extended_called=round(float(ext_ok), 3) if ext_ok is not None else None)
        out[name][rec] = row
        print(name, rec, json.dumps(row), flush=True)
json.dump(out, open("recordings/deadbug_alternation_" + "_".join(r[-6:] for r in recs) + ".json", "w"), indent=1)
