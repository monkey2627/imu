"""Replay a recording through the production session path with the IMUDINO stream adapter.

Same preprocessing as run.py (bone_orientation, identity heading -> SMPL basis,
recorded identity calibration), same push/infer cadence as the live prediction
rows. Reports per-call latency and output continuity (no smoothing). Read-only.

    python -m analysis.imudino_stream_replay_check BirdDog_RingSkinC_Recorded_20261111_235722 [--checkpoint path/best.pt]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from analysis.imudino_jitter_stages import LABELS, NAMES, angle, fk, load, resolve_recording_path
from production.calibration import identity_heading
from production.layout import ResolvedLayout
from production.model_backends import (WAIST_CHECKPOINT_ROOT, WAIST_MODEL_FILES, load_waist_adapter,
                                       make_waist_session)

MODEL = "waist_imudino_stream_wristankle"
JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 16, 17, 18, 19]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--checkpoint", type=Path, default=WAIST_CHECKPOINT_ROOT / WAIST_MODEL_FILES[MODEL][1])
    args = ap.parse_args()
    header, ticks, _, quats, acc, pred_tick, _, _, _ = load(resolve_recording_path(args.recording))
    resolved = ResolvedLayout(header["layout"], LABELS, dict(header["labelToTrackerKey"]))
    adapter = load_waist_adapter("imudino_stream", args.checkpoint)
    session = make_waist_session(MODEL, resolved, identity_heading(LABELS),
                                 loaded=("imudino_stream", args.checkpoint, adapter))
    up = header.get("upright")
    if up and up.get("quats"):
        session.prime(np.asarray(up["quats"], np.float32), np.asarray(up["accel"], np.float32))
    outs, frames, cost = [], [], []
    k = 0
    for t in range(len(quats)):
        feature, ori, accel = session.prepare_feature(quats[t], acc[t])
        session.push_observation(feature, ori, accel)
        while k < len(pred_tick) and pred_tick[k] == t:
            t0 = time.perf_counter()
            out = session.infer()
            cost.append(time.perf_counter() - t0)
            outs.append(out.double().numpy())
            frames.append(adapter.decoded)
            k += 1
    seq, frames = np.stack(outs), np.asarray(frames)
    skip = 60  # recurrent start-up
    seq, frames, cost = seq[skip:], frames[skip:], np.asarray(cost[skip:]) * 1000
    two = np.diff(frames) == 2
    loc = angle(seq[1:][two][:, JOINTS], seq[:-1][two][:, JOINTS])
    g = fk(seq)
    glob = angle(g[1:][two][:, JOINTS], g[:-1][two][:, JOINTS])
    rep = {
        "checkpoint": str(args.checkpoint), "predictions": int(len(seq)),
        "inferMs": {q: round(float(np.percentile(cost, q)), 2) for q in (50, 90, 99)} | {"max": round(float(cost.max()), 2)},
        "lagFrames": adapter.lag,
        "local2_p50_p99": [round(float(np.percentile(loc, 50)), 2), round(float(np.percentile(loc, 99)), 2)],
        "global2_p50_p99": [round(float(np.percentile(glob, 50)), 2), round(float(np.percentile(glob, 99)), 2)],
        "perJointLocal2_p99": {NAMES[j]: round(float(np.percentile(loc[:, i], 99)), 2) for i, j in enumerate(JOINTS)},
        "perJointGlobal2_p99": {NAMES[j]: round(float(np.percentile(glob[:, i], 99)), 2) for i, j in enumerate(JOINTS)},
    }
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
