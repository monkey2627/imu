"""Render the glute-bridge chain for a recording without a bridge calibration (lie-flat calibration emulated
from the first steady rest window, as in analysis.bridge_validate): MP4 + SMPL web page.

    python -m analysis.bridge_render unity_20261116_022753
"""
import sys

import numpy as np
import torch

from analysis.bridge_validate import rest_window
from production.bridge_contact import mean_rotation, trunk_bias, waist_flat_fit
from production.deadbug_replay import per_frame_unity_quaternions, replay
from production.input_recording import resolve_recording_path
from production.mesh_replay import write_mesh_replay
from production.mesh_video import write_video

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    path = resolve_recording_path(sys.argv[1])
    base = replay(path, leg_mode="off")
    window = rest_window(base)
    w = slice(*window)
    A, tilt = waist_flat_fit(mean_rotation(base.calibrated_ori[w, base.labels.index("waist")].astype(np.float64)))
    r_a = replay(path, leg_mode="off", bridge_override={"bridgeWaist": A.tolist()})
    fr = np.clip(r_a.represented_frames, 0, r_a.n_frames - 1)
    B, _ = trunk_bias(r_a.raw_local[(fr >= window[0]) & (fr < window[1])])
    r = replay(path, leg_mode="off", bridge_override={"bridgeWaist": A.tolist(),
                                                      "bridgeTrunk": {str(k): v.tolist() for k, v in B.items()}})
    q = per_frame_unity_quaternions(r)
    labels = [""] * r.n_frames
    for frame, diag in zip(r.represented_frames, r.diagnostics):
        b = diag["deadbugLegs"].get("bridge") or {}
        if 0 <= frame < r.n_frames and b:
            labels[frame] = (f"臀桥约束：骨盆高 {100 * b.get('pelvisHeightM', 0):.0f} cm · 左大腿 "
                             f"{b.get('left', {}).get('thighElevTargetDeg', 0):.0f}° · 右大腿 "
                             f"{b.get('right', {}).get('thighElevTargetDeg', 0):.0f}°")
    for i in range(1, len(labels)):
        labels[i] = labels[i] or labels[i - 1]
    note = f"臀桥接触约束 + 平躺校准（离线模拟，腰部 {tilt:.0f}°，取 {window[0] / 60:.1f}-{window[1] / 60:.1f} s）"
    print(write_mesh_replay(path, path.with_name(path.stem + "_bridge_fix.html"), precomputed=(q, labels),
                            title_suffix=note))
    print(write_video(path, path.with_name(path.stem + "_bridge_fix.mp4"), q, f"{path.stem}  bridge contact + lie-flat (emulated)"))
