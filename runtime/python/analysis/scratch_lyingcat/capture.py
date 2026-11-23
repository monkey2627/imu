"""Capture every layer of the live lying-cat chain on a recording (replay_live(simulate=True)).

Saves analysis/scratch_lyingcat/cap_<stem>[_<tag>].npz with per prediction:
  t, raw (network output, 24x3x3), anchored (after calibrator.anchor), final, measured (calibrated sensors,
  n_labels x3x3), uncal (bone_orientation of the tick quats, no calibration), headings, corrections, diag json.
    python -m analysis.scratch_lyingcat.capture <recording> [tag] [variant_module]
"""
import importlib, json, sys
import numpy as np, torch
import production.frame_chain as fc
import production.live_replay as lr
from production.calibration import bone_orientation
from production.paths import paths

torch.set_grad_enabled(False); torch.set_num_threads(8)


def capture(rec, rule=True, variant=None):
    if variant:
        fc.LyingCatContact = importlib.import_module(variant).LyingCatContact
    rows = {k: [] for k in ("raw", "anchored", "final", "measured", "head", "corr", "diag")}
    orig_pp = lr.postprocess_output
    calls = {"n": 0}

    def wrapped(output, calibrator, lag, labels, *a, **kw):
        calls["n"] += 1
        if not rule:
            kw = dict(kw); kw["action_id"] = None if kw.get("action_id") == "lyingcat" else kw.get("action_id")
        res = orig_pp(output, calibrator, lag, labels, *a, **kw)
        if calls["n"] % 2 == 1:
            m = calibrator.measured_frame(lag)
            rows["raw"].append(output.detach().numpy().astype(np.float32).copy())
            rows["anchored"].append(calibrator.anchor(output, lag).numpy().astype(np.float32))
            rows["final"].append(res[0].detach().numpy().astype(np.float32))
            rows["measured"].append(np.full((5, 3, 3), np.nan) if m is None else np.asarray(m).copy())
            rows["head"].append(np.asarray(calibrator.headings).copy())
            rows["corr"].append(np.asarray(calibrator.corrections).copy())
            rows["diag"].append(json.dumps(res[2].get("lyingcat")))
        return res

    lr.postprocess_output = wrapped
    try:
        result = lr.replay_live(paths.recordings / rec if not rec.startswith(("recordings", "E:")) else rec, simulate=True)
    finally:
        lr.postprocess_output = orig_pp
    # uncalibrated waist per tick
    ticks_t, uncal = [], []
    labels = result.labels
    from production.input_recording import _sample_for
    for line in open(result.path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") != "tick":
            continue
        s = {l: _sample_for(r["samples"][l], l) for l in labels}
        q = np.asarray([s[l].quat_xyzw for l in labels], np.float32)
        ticks_t.append(r["t"]); uncal.append(bone_orientation(q, labels).numpy())
    return result, rows, np.asarray(ticks_t), np.asarray(uncal)


if __name__ == "__main__":
    rec = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "prod"
    variant = sys.argv[3] if len(sys.argv) > 3 else None
    result, rows, tt, uncal = capture(rec, rule=(tag != "off"), variant=variant)
    stem = rec.split("/")[-1].replace(".jsonl", "")
    np.savez_compressed(f"analysis/scratch_lyingcat/cap_{stem}_{tag}.npz", t=result.times, labels=np.array(result.labels),
                        tick_t=tt, uncal=uncal, **{k: np.asarray(v) for k, v in rows.items()})
    print(result.summary()); print("saved", stem, tag, len(result.times))
