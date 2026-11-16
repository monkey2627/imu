"""Compare dead-bug stream checkpoints on real recordings, without ground truth.

Reference states come from the motion-evidence leg estimator (production/deadbug_legs.py): frames where it
reports a confirmed "flexed" (tabletop) or "extended" hold with confidence >= 0.9.  For each checkpoint the
*raw network* thigh angle and knee flexion on those frames are summarised (network output before any
post-processing), plus how often the network itself calls the state correctly.

    python -m analysis.deadbug_model_compare unity_20261113_232942 ... \
        --model old=<dir with best.pt + imudino.pt> --model new=<dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.deadbug_diagnostics import globals_of, sagittal
from production.deadbug_replay import replay
from production.input_recording import resolve_recording_path

MODEL_ID = "waist_imudino_stream_deadbug_limbmix"   # same contract / lag; only the weights differ


def load(path: Path):
    from production.model_backends import _imudino_entry
    adapter = _imudino_entry("stream_entry").StreamAdapter(Path(path) / "best.pt")
    return "imudino_stream", Path(path) / "best.pt", adapter


def network_legs(result):
    frames = np.clip(result.represented_frames, 0, result.n_frames - 1)
    waist = result.calibrated_ori[frames, result.labels.index("waist")].astype(np.float64)
    g = globals_of(result.raw_local.astype(np.float64))
    down = np.array([0, -1.0, 0])
    out = {}
    for side, hip, knee in (("left", 1, 4), ("right", 2, 5)):
        thigh, shank = g[:, hip] @ down, g[:, knee] @ down
        out[side] = (sagittal(thigh, waist),
                     np.degrees(np.arccos(np.clip((thigh * shank).sum(-1), -1, 1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--model", action="append", required=True,
                    help="name=checkpoint directory[:accelfix]  (accelfix: feed the sensor-frame-corrected acceleration)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    report = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        accel_fix = path.endswith(":accelfix")
        path = path[:-len(":accelfix")] if accel_fix else path
        pools = {"flexed": {"thigh": [], "knee": [], "called_ok": []},
                 "extended": {"thigh": [], "knee": [], "called_ok": []}}
        for rec in args.recordings:
            result = replay(resolve_recording_path(rec), MODEL_ID, leg_mode="auto", loaded=load(path),
                            model_accel_fix=accel_fix)
            net = network_legs(result)
            for i, diag in enumerate(result.diagnostics):
                legs = diag["deadbugLegs"]
                for side in ("left", "right"):
                    leg = legs.get(side, {})
                    if leg.get("confidence", 0) < 0.9 or leg.get("phase") not in ("flexed", "extended"):
                        continue
                    thigh, knee = net[side][0][i], net[side][1][i]
                    if leg["phase"] == "flexed":
                        ok = thigh > 60 and knee > 50
                    else:
                        ok = thigh < 45 and knee < 40
                    pool = pools[leg["phase"]]
                    pool["thigh"].append(thigh); pool["knee"].append(knee); pool["called_ok"].append(ok)
        report[name] = {}
        for state, pool in pools.items():
            th, kn = np.asarray(pool["thigh"]), np.asarray(pool["knee"])
            report[name][state] = {
                "frames": int(len(th)),
                "network_thigh_p25_50_75": np.percentile(th, (25, 50, 75)).round(1).tolist() if len(th) else None,
                "network_knee_p25_50_75": np.percentile(kn, (25, 50, 75)).round(1).tolist() if len(kn) else None,
                "network_state_correct": round(float(np.mean(pool["called_ok"])), 3) if pool["called_ok"] else None,
            }
        print(name, json.dumps(report[name]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
