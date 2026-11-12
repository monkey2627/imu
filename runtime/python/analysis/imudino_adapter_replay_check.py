"""Drive the production FiveIMUAdapter with a recording's exact live cadence.

Pushes every recorded 60 Hz tick (same preprocessing as run.py with the
recorded identity calibration) and calls infer() exactly where the live bridge
did (prediction rows). With --lag 9 the output must equal the recorded
rawLocalQuaternions; other lags report continuity and distance to the
validation-style stride-240 stitch of the same input. Read-only.

    python -m analysis.imudino_adapter_replay_check BirdDog_RingSkinC_Recorded_20261111_235722 --lags 9 15 23
"""
import argparse
import json

import numpy as np
import torch

from analysis.imudino_jitter_stages import (
    LABELS, NAMES, W, angle, bone_orientation, fk, identity_heading, load, quat_xyzw_to_mat,
    resolve_recording_path, stitch, unity_to_smpl,
)
from production.model_backends import load_waist_adapter, model_lag_frames, WAIST_CHECKPOINT_ROOT, WAIST_MODEL_FILES

JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 16, 17, 18, 19]


def p(x, q):
    return round(float(np.percentile(x, q)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--lags", type=int, nargs="+", default=[9, 15, 23])
    args = ap.parse_args()
    _, ticks, preds, quats, acc_server, pred_tick, _, raw_q, _ = load(resolve_recording_path(args.recording))
    heading = identity_heading(LABELS)
    ori = bone_orientation(quats, LABELS).numpy().astype(np.float32)
    acc = np.stack([heading.apply(x).numpy() for x in acc_server]).astype(np.float32)
    recorded = unity_to_smpl(quat_xyzw_to_mat(raw_q))
    checkpoint = WAIST_CHECKPOINT_ROOT / WAIST_MODEL_FILES["waist_imudino_fiveimu"][1]
    adapter = load_waist_adapter("imudino", checkpoint)
    ref_all = stitch(_StitchRunner(adapter), acc, ori, 1, len(ori) - 1)
    report = {"defaultModelLagFrames": model_lag_frames("waist_imudino_fiveimu")}
    for lag in args.lags:
        adapter.lag = lag
        adapter.orientations.clear(); adapter.accelerations.clear(); adapter.votes.clear(); adapter.pushed = 0
        out, rep_frames, counts = {}, {}, []
        k = 0
        for t in range(len(ori)):
            adapter.push(None, ori[t], acc[t])
            while k < len(pred_tick) and pred_tick[k] == t:
                if t >= W:
                    out[k] = adapter.infer().double().numpy()
                    rep_frames[k] = t - lag
                    counts.append(adapter.last_vote_count)
                k += 1
        ks = np.asarray(sorted(out))
        seq = np.stack([out[i] for i in ks])
        frames = np.asarray([rep_frames[i] for i in ks])
        steps = np.diff(frames)
        two = steps == 2
        local2 = angle(seq[1:][two][:, JOINTS], seq[:-1][two][:, JOINTS])
        g = fk(seq)
        glob2 = angle(g[1:][two][:, JOINTS], g[:-1][two][:, JOINTS])
        ref = ref_all[frames]
        dev = angle(seq[:, 1:], ref[:, 1:])
        entry = {
            "n": int(len(ks)),
            "windowsPerFrame": {"p50": p(counts, 50), "min": int(min(counts)), "max": int(max(counts))},
            "local2_p50": p(local2, 50), "local2_p99": p(local2, 99),
            "global2_p50": p(glob2, 50), "global2_p99": p(glob2, 99),
            "perJointLocal2_p99": {NAMES[j]: p(local2[:, i], 99) for i, j in enumerate(JOINTS)},
            "perJointGlobal2_p99": {NAMES[j]: p(glob2[:, i], 99) for i, j in enumerate(JOINTS)},
            "devFromStride240_p50": p(dev, 50), "devFromStride240_p90": p(dev, 90),
        }
        if lag == 9:
            entry["vsRecordedRaw_deg"] = {"p50": p(angle(seq, recorded[ks]), 50),
                                          "max": round(float(angle(seq, recorded[ks]).max()), 3)}
        report[f"lag{lag}"] = entry
    print(json.dumps(report, indent=1, ensure_ascii=False))


class _StitchRunner:
    def __init__(self, adapter):
        self.adapter = adapter

    def dense(self, imu):
        from waist_imudino_fiveimu.model_entry import sixd_to_matrix
        pred = self.adapter.predict_imu(imu)[0]
        out = np.tile(np.eye(3), (W, 24, 1, 1))
        out[:, 1:] = sixd_to_matrix(pred.reshape(W, 23, 6)).double().numpy()
        return out


if __name__ == "__main__":
    main()
