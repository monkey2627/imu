"""Per arm-reach episode: opposite thigh lift predicted live vs. what the shank sensors show.

Read-only. For each recording (stream-model prediction rows only), finds forearm
reach episodes (forearm pitch > 35 deg) and prints the opposite thigh pitch (0 =
vertical kneeling, 90 = horizontal lift), the opposite shank sensor pitch range and
its vertical acceleration range. Static shank + high thigh = prior-driven leg lift.

    python -m analysis.reach_thigh_episodes BirdDog_RingSkinC_Recorded_20261112_015543 [...]
    python -m analysis.reach_thigh_episodes <recording> --replay path/to/best.pt   # offline re-run, same cadence
"""
import argparse
from pathlib import Path

import numpy as np
from analysis.imudino_jitter_stages import LABELS, fk, load, quat_xyzw_to_mat, resolve_recording_path, unity_to_smpl
from analysis.thigh_pitch_compare import pitch
from production.calibration import bone_orientation, identity_heading
DOWN = np.array([0, -1.0, 0])
ap = argparse.ArgumentParser()
ap.add_argument("recordings", nargs="+")
ap.add_argument("--replay", type=Path, default=None, help="stream checkpoint to re-run offline instead of live output")
args = ap.parse_args()
for name in args.recordings:
    header, _, preds, quats, acc, pred_tick, _, raw_q, _ = load(resolve_recording_path(name))
    if args.replay is None:
        keep = np.array([p["diagnostics"]["model"] == "waist_imudino_stream_wristankle" for p in preds])
        g = fk(unity_to_smpl(quat_xyzw_to_mat(raw_q)))
    else:
        from analysis.thigh_pitch_compare import stream_replay
        keep = np.ones(len(preds), bool)
        g = fk(stream_replay(header, quats, acc, pred_tick, args.replay))
    p = pitch(g)
    ori = bone_orientation(quats, LABELS).numpy().astype(float)
    accs = np.stack([identity_heading(LABELS).apply(a).numpy() for a in acc])
    # sensor shank pitch (bone axis -y of calibrated shank), and shank world vertical accel
    sp = {s: np.degrees(np.arccos(np.clip(ori[:, i] @ np.array([0, -1.0, 0]) @ DOWN, -1, 1))) for s, i in (("L", 2), ("R", 3))}
    lead = pred_tick - 1  # frame represented
    print(f"== {name[-15:]}  ({keep.sum()} stream preds)")
    for side, other, fa in (("L", "R", "L_forearm"), ("R", "L", "R_forearm")):
        reach = (p[fa] > 35) & keep
        # episodes
        idx = np.flatnonzero(reach)
        if not len(idx):
            print(f"  {side} reach: none"); continue
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 15) + 1)
        for e in splits:
            if len(e) < 10 or not len(accs[max(lead[e][0] - 30, 0):lead[e][-1] + 30]):
                continue
            f = lead[e]
            th = p[f"{other}_thigh"][e]
            sh_s = sp[other][f]
            vacc = accs[max(f[0] - 30, 0):f[-1] + 30, 2 if other == "L" else 3, 1]
            print(f"  {side}-reach t={preds[e[0]]['t']:.1f}-{preds[e[-1]]['t']:.1f}s  forearm max {p[fa][e].max():.0f}  "
                  f"{other}-thigh max {th.max():.0f} (med {np.median(th):.0f})  {other}-shank sensor pitch {sh_s.min():.0f}-{sh_s.max():.0f}  "
                  f"{other}-shank vert acc range {vacc.min():.1f}..{vacc.max():.1f}")
