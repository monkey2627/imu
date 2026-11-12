"""Arm direction and forearm twist: sensor vs. predicted, in the waist frame.

Read-only. Directions are expressed in the waist-sensor frame (SMPL axes: x = body left,
y = up, z = forward). For each forearm: sensor bone axis, sensor palm normal (bone-local -y,
i.e. palm-down in T-pose), predicted forearm axis / palm normal, predicted upper-arm axis.
    python -m analysis.arm_forward_twist <recording> [--replay ckpt] [--step 30]
"""
import argparse
from pathlib import Path

import numpy as np

from analysis.imudino_jitter_stages import LABELS, fk, load, quat_xyzw_to_mat, resolve_recording_path, unity_to_smpl
from production.calibration import bone_orientation

AX = {"L": np.array([1.0, 0, 0]), "R": np.array([-1.0, 0, 0])}
PALM = np.array([0, -1.0, 0])
FORE = {"L": 18, "R": 19}
UPPER = {"L": 16, "R": 17}
SENS = {"L": 0, "R": 1}


def az_el(v):
    """azimuth: 0 = forward, +90 = body left; elevation: + up."""
    return np.degrees(np.arctan2(v[..., 0], v[..., 2])), np.degrees(np.arcsin(np.clip(v[..., 1], -1, 1)))


def fmt(v):
    a, e = az_el(v)
    return f"{a:+4.0f}/{e:+3.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--raw", action="store_true", help="ignore the recording's initialCalibration")
    args = ap.parse_args()
    header, _, preds, quats, acc, pred_tick, _, raw_q, _ = load(resolve_recording_path(args.recording))
    if args.replay is None:
        g = fk(unity_to_smpl(quat_xyzw_to_mat(raw_q)))
    else:
        from analysis.thigh_pitch_compare import stream_replay
        g = fk(stream_replay(header, quats, acc, pred_tick, args.replay))
    ori = bone_orientation(quats, LABELS).numpy().astype(float)
    cal = header.get("initialCalibration")
    if cal and not args.raw:   # same correction the live loop applied: H @ bone @ C
        ori = np.asarray(cal["headings"])[None] @ ori @ np.asarray(cal["corrections"])[None]
    frame = pred_tick - 1
    W = ori[frame, 4]                       # waist sensor
    Wt = np.swapaxes(W, -1, -2)
    t = np.array([p["t"] for p in preds])
    print(f"model={preds[0]['diagnostics']['model']}  cols: t | per side: sensor fore az/el, sensor palm az/el, "
          f"pred fore az/el, pred palm az/el, pred upper az/el, fore dir err, twist err")
    rows = []
    for i in range(0, len(t), args.step):
        line = [f"{t[i]:6.1f}"]
        for s in ("L", "R"):
            S = Wt[i] @ ori[frame[i], SENS[s]]
            P = Wt[i] @ g[i, FORE[s]]
            U = Wt[i] @ g[i, UPPER[s]]
            sf, sp, pf, pp, pu = S @ AX[s], S @ PALM, P @ AX[s], P @ PALM, U @ AX[s]
            derr = np.degrees(np.arccos(np.clip(sf @ pf, -1, 1)))
            # twist error: palm normals after projecting out the forearm axis
            a = sp - (sp @ sf) * sf; b = pp - (pp @ sf) * sf
            terr = np.degrees(np.arctan2(np.cross(b, a) @ sf, a @ b)) if np.linalg.norm(a) > 1e-6 else np.nan
            line.append(f"{s}: {fmt(sf)} {fmt(sp)} | {fmt(pf)} {fmt(pp)} up {fmt(pu)} | d{derr:3.0f} tw{terr:+4.0f}")
        print("  ".join(line))


if __name__ == "__main__":
    main()
