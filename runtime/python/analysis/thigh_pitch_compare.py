"""Thigh/shank/forearm pitch over time: recorded live output vs offline stream-model replay.

Angle of each bone's long axis from straight down (0 = hanging/vertical, 90 = horizontal).
Kneeling on all fours: thigh ~0-20 deg, shank ~90 deg. Bird-dog leg lift: thigh ~70-100 deg.
Read-only. Usage:
    python -m analysis.thigh_pitch_compare BirdDog_RingSkinC_Recorded_20261112_014354
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.imudino_jitter_stages import LABELS, fk, load, quat_xyzw_to_mat, resolve_recording_path, unity_to_smpl
from production.calibration import bone_orientation, identity_heading
from production.layout import ResolvedLayout
from production.model_backends import WAIST_CHECKPOINT_ROOT, WAIST_MODEL_FILES, load_waist_adapter, make_waist_session

DOWN = np.array([0.0, -1.0, 0.0])
# SMPL rest-pose bone directions (child joint - joint), Y-up: legs point down, arms point sideways.
AXIS = {"L_thigh": (1, [0, -1, 0]), "R_thigh": (2, [0, -1, 0]), "L_shank": (4, [0, -1, 0]),
        "R_shank": (5, [0, -1, 0]), "L_forearm": (18, [1, 0, 0]), "R_forearm": (19, [-1, 0, 0])}


def pitch(global_rot):
    out = {}
    for name, (j, axis) in AXIS.items():
        v = global_rot[:, j] @ np.asarray(axis, float)
        out[name] = np.degrees(np.arccos(np.clip(v @ DOWN, -1, 1)))
    return out


def stream_replay(header, quats, acc, pred_tick, checkpoint=None):
    model = "waist_imudino_stream_wristankle"
    ck = Path(checkpoint) if checkpoint else WAIST_CHECKPOINT_ROOT / WAIST_MODEL_FILES[model][1]
    adapter = load_waist_adapter("imudino_stream", ck)
    session = make_waist_session(model, ResolvedLayout(header["layout"], LABELS, dict(header["labelToTrackerKey"])),
                                 identity_heading(LABELS), loaded=("imudino_stream", ck, adapter))
    up = header.get("upright")
    if up and up.get("quats"):
        session.prime(np.asarray(up["quats"], np.float32), np.asarray(up["accel"], np.float32))
    outs, k = [], 0
    for t in range(len(quats)):
        session.push_observation(*session.prepare_feature(quats[t], acc[t]))
        while k < len(pred_tick) and pred_tick[k] == t:
            outs.append(session.infer().double().numpy()); k += 1
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    args = ap.parse_args()
    header, _, preds, quats, acc, pred_tick, _, raw_q, fin_q = load(resolve_recording_path(args.recording))
    live_model = preds[0]["diagnostics"]["model"]
    rec = {"live_raw": fk(unity_to_smpl(quat_xyzw_to_mat(raw_q))), "live_final": fk(unity_to_smpl(quat_xyzw_to_mat(fin_q)))}
    rec["stream"] = fk(stream_replay(header, quats, acc, pred_tick))
    t = np.asarray([p["t"] for p in preds])
    report = {"liveModel": live_model, "seconds": round(float(t[-1]), 1)}
    for src, g in rec.items():
        p = pitch(g)
        report[src] = {name: {"p10": round(float(np.percentile(v, 10)), 1), "p50": round(float(np.percentile(v, 50)), 1),
                              "p90": round(float(np.percentile(v, 90)), 1)} for name, v in p.items()}
    # time series every ~2 s for eyeballing
    rows = []
    pl, ps = pitch(rec["live_raw"]), pitch(rec["stream"])
    for i in range(0, len(t), 60):
        rows.append([round(float(t[i]), 1), round(float(pl["L_forearm"][i])), round(float(pl["R_forearm"][i])),
                     round(float(pl["L_thigh"][i])), round(float(pl["R_thigh"][i])),
                     round(float(ps["L_thigh"][i])), round(float(ps["R_thigh"][i])),
                     round(float(pl["L_shank"][i])), round(float(pl["R_shank"][i]))])
    report["series_cols"] = "t, live L/R forearm, live L/R thigh, stream L/R thigh, live L/R shank"
    report["series"] = rows
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
