"""Compare online read-out schemes of the frozen IMUDINO five-IMU model on one recording.

Read-only and offline. Every scheme sees exactly the same 60 Hz input as the
live bridge (tick rows, SMPL-basis acceleration) and the same weights; only
*which window(s)* produce the pose for frame f changes. No temporal filter is
applied to any output sequence: multi-window schemes average predictions of the
SAME frame from different window placements (as validation stitching does),
never neighbouring frames.

    python -m analysis.imudino_readout_sweep BirdDog_RingSkinC_Recorded_20261111_235722 --frames 1200
"""
import argparse
import json

import numpy as np
import torch

from analysis.imudino_jitter_stages import (
    LABELS, NAMES, W, Runner, angle, bone_orientation, identity_heading, load, resolve_recording_path,
    stitch, to_imu6,
)

KEEP = 160  # keep window indices [480-KEEP, 480)
JOINTS = [1, 2, 3, 4, 5, 6, 9, 16, 17, 18, 19]


def mean_rot(stack):
    m = np.mean(stack, axis=0)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    det = np.linalg.det(r)
    if (det < 0).any():
        u[det < 0, :, -1] *= -1
        r = u @ vt
    return r


def summary(seq, ref):
    """seq/ref: [T,24,3,3] local poses for the same frames."""
    step2 = angle(seq[2:, JOINTS], seq[:-2, JOINTS])
    acc2 = angle(seq[2:, JOINTS] @ np.swapaxes(seq[1:-1, JOINTS], -1, -2),
                 seq[1:-1, JOINTS] @ np.swapaxes(seq[:-2, JOINTS], -1, -2))  # change of 1-frame increment
    dev = angle(seq[:, JOINTS], ref[:, JOINTS])
    return {
        "step2_p50": round(float(np.percentile(step2, 50)), 2),
        "step2_p99": round(float(np.percentile(step2, 99)), 2),
        "jerk_p99": round(float(np.percentile(acc2, 99)), 2),
        "devFromStride240_p50": round(float(np.percentile(dev, 50)), 2),
        "devFromStride240_p90": round(float(np.percentile(dev, 90)), 2),
        "perJointStep2_p99": {NAMES[j]: round(float(np.percentile(step2[:, i], 99)), 1)
                              for i, j in enumerate(JOINTS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--start", type=int, default=W)
    args = ap.parse_args()
    _, _, _, quats, acc, _, _, _, _ = load(resolve_recording_path(args.recording))
    ori = bone_orientation(quats, LABELS).numpy().astype(np.float32)
    acc = np.stack([identity_heading(LABELS).apply(x).numpy() for x in acc]).astype(np.float32)
    runner = Runner()
    first_end = args.start
    last_end = min(len(ori) - 2, first_end + args.frames + 64)
    # dense[e]: window built from pushed frames e-480..e; index i == frame e-480+i (frame e = gyro lookahead).
    dense = {}
    for e in range(first_end, last_end + 1):
        dense[e] = runner.window(acc, ori, e)[W - KEEP:].astype(np.float32)

    def pred(e, f):
        idx = f - (e - W)  # window index of frame f
        return dense[e][idx - (W - KEEP)].astype(np.float64)

    f0 = first_end
    f1 = first_end + args.frames - 64
    frames = np.arange(f0, f1)
    ref = stitch(runner, acc, ori, 1, last_end)[frames]  # stitched row i == model frame i
    out = {"frames": [int(f0), int(f1)]}

    def run(name, fn, latency):
        seq = np.stack([fn(f) for f in frames])
        out[name] = {"latencyFrames": latency, **summary(seq, ref)}

    # Current: index 471 (lag 9: 8 plateau + 1 gyro lookahead), recomputed every frame.
    for idx in (471, 463, 447, 431, 399):
        run(f"single_idx{idx}", lambda f, idx=idx: pred(f + W - idx, f), W - idx)
    # Same-frame ensemble across window placements, newest window still index 471.
    for m, s in ((2, 2), (4, 2), (4, 4), (8, 2), (8, 4), (16, 2)):
        lag = 9 + (m - 1) * s
        run(f"avg_M{m}_s{s}", lambda f, m=m, s=s: mean_rot(np.stack(
            [pred(f + 9 + k * s, f) for k in range(m)])), lag)
    # Fixed-phase chunk: one window per 16 frames, its dense trajectory played for the chunk.
    for k in (8, 16):
        def chunk(f, k=k):
            e = -(-(f + 9) // k) * k  # first window end >= f+9 on a fixed k-frame grid
            return pred(e, f)
        run(f"chunk{k}", chunk, 9 + k - 1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
