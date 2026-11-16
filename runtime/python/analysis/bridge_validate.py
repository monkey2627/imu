"""Offline validation of the glute-bridge chain (bridge_contact.py) on recordings without a bridge calibration.

The lie-flat calibration is emulated from the first steady rest window (lying, knees bent, feet planted):
A = waist_flat_fit(mean calibrated waist), B = trunk_bias(raw outputs with A).  Then the full chain
(A + B + contact solve) is compared with the recorded output.

    python -m analysis.bridge_validate unity_20261116_022753 ...
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from production.bridge_contact import mean_rotation, trunk_bias, waist_flat_fit
from production.deadbug_replay import replay
from production.input_recording import resolve_recording_path
from production.paths import paths
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw  # noqa: F401

Y = np.array([0, 1.0, 0])
BODY = None


def elevation(v):
    return np.degrees(np.arcsin(np.clip(v[..., 1] / np.linalg.norm(v, axis=-1), -1, 1)))


def rest_window(result, seconds=1.0):
    """Steady 1 s lying window (chest up, shanks down, waist still) with the flattest pelvis, i.e. the
    hips down between repetitions (not a held bridge top)."""
    L = result.labels
    ori = result.calibrated_ori.astype(np.float64)
    waist, shank = ori[:, L.index("waist")], ori[:, L.index("ls")] @ -Y
    chest_up = waist[:, 1, 2]
    sh = elevation(shank)
    n = int(seconds * 60)
    head = elevation(waist @ Y)
    best = None
    for start in range(60, len(ori) - n, 15):
        w = slice(start, start + n)
        if (chest_up[w] > 0.7).all() and ((sh[w] > -80) & (sh[w] < -40)).all():
            mean = mean_rotation(waist[w])
            spread = max(np.degrees(np.arccos(np.clip((np.trace(mean.T @ r) - 1) / 2, -1, 1))) for r in waist[w])
            if spread < 3.0 and (best is None or head[w].mean() > best[0]):
                best = (head[w].mean(), start)
    return None if best is None else (best[1], best[1] + n)


def metrics(result, local):
    global BODY
    BODY = BODY or ParametricModel(str(paths.smpl_file))
    _, joints, verts = BODY.forward_kinematics(torch.from_numpy(local.astype(np.float32)), calc_mesh=True)
    j, v = joints.numpy(), verts.numpy()
    floor = v[..., 1].min(1)
    torso = elevation(j[:, 12] - j[:, 0])
    fr = np.clip(result.represented_frames, 0, result.n_frames - 1)
    L = result.labels
    sh = elevation(result.calibrated_ori[fr, L.index("ls")].astype(np.float64) @ -Y)
    waist = result.calibrated_ori[fr, L.index("waist")].astype(np.float64)
    lying = (waist[:, 1, 2] > 0.7) & (sh < -40)
    head = elevation(waist @ Y)
    if lying.sum() < 10:
        return {}
    hi, lo = np.percentile(head[lying], (70, 20))
    rest, top = lying & (head >= hi), lying & (head <= lo) & (head < hi - 10)
    out = {}
    for name, k in (("rest", rest), ("top", top)):
        if k.sum() < 5:
            continue
        out[name] = {"frames": int(k.sum()), "torsoDeg": round(float(np.median(torso[k])), 1),
                     "pelvisCm": round(float(np.median((j[k, 0, 1] - floor[k]) * 100)), 1),
                     "neckCm": round(float(np.median((j[k, 12, 1] - floor[k]) * 100)), 1),
                     "thighDeg": round(float(np.median(elevation(j[k, 4] - j[k, 1]))), 1)}
    return out


def main():
    torch.set_grad_enabled(False)
    report = {}
    for rec in sys.argv[1:]:
        path = resolve_recording_path(rec)
        base = replay(path, leg_mode="off")
        window = rest_window(base)
        row = {"recorded_chain_old": None}
        if window is None:
            print(rec, "no steady rest window")
            continue
        w = slice(*window)
        waist = mean_rotation(base.calibrated_ori[w, base.labels.index("waist")].astype(np.float64))
        A, tilt = waist_flat_fit(waist)
        r_a = replay(path, leg_mode="off", bridge_override={"bridgeWaist": A.tolist()})
        fr = np.clip(r_a.represented_frames, 0, r_a.n_frames - 1)
        hold = (fr >= window[0]) & (fr < window[1])
        B, spread = trunk_bias(r_a.raw_local[hold])
        r_ab = replay(path, leg_mode="off", bridge_override={
            "bridgeWaist": A.tolist(), "bridgeTrunk": {str(k): v.tolist() for k, v in B.items()}})
        row = {"restWindowS": [round(window[0] / 60, 2), round(window[1] / 60, 2)], "waistTiltDeg": round(tilt, 1),
               "trunkSpreadDeg": round(spread, 1),
               "raw_network": metrics(base, base.raw_local.astype(np.float64)),
               "contact_only": metrics(base, base.final_local.astype(np.float64)),
               "lieflat_A_B_contact": metrics(r_ab, r_ab.final_local.astype(np.float64))}
        report[rec] = row
        print(rec, json.dumps(row, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    out = main()
    open("recordings/bridge_validate_20261116.json", "w", encoding="utf-8").write(json.dumps(out, indent=1, ensure_ascii=False))
