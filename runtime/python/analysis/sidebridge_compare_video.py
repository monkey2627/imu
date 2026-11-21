"""Side-by-side SMPL mesh video of the side-bridge support rule (analysis/sidebridge_variants).

Columns: ① production output before the side-bridge rule (live_replay with the rule switched off)
| ② + legs straight & stacked, top arm along the thigh | ③ = production (production/sidebridge_contact.py): + support forearm on the floor, head in line.
The rule variants post-process ① with the calibrated sensor orientations of each output's frame.
Top row: side view; bottom row: 3/4 view; each body's lowest vertex on the floor.

    python -m analysis.sidebridge_compare_video [recording] [--workers 8]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from analysis.compare_video import render
from analysis.sidebridge_variants.sidebridge_contact import SideBridgeContact
from production.arm_prior import PARENTS, _globals
from production.live_replay import replay_live
from production.paths import paths

LABEL_JOINT = {"lw": 18, "rw": 19, "ls": 4, "rs": 5, "waist": 0}


def measured_from(local, labels):
    """Sensed-bone globals.  Valid because this recording ran with sensed-bone anchoring on."""
    g = np.asarray(_globals(np.asarray(local, np.float64)))
    return np.stack([g[LABEL_JOINT[label]] for label in labels])


def run_rule(rule, base, times, labels, dt, rebases):
    """As live: the rule is reset at the start-pose rebase (leg_chain.reset + activate)."""
    resets = sorted(t for t, _a, _p in rebases)
    out, diags = [], []
    for local, t in zip(base, times):
        while resets and t >= resets[0]:
            rule.reset("rebase")
            resets.pop(0)
        local, diag = rule.apply(local, dt, measured_from(local, labels), labels)
        out.append(local)
        diags.append(diag)
    return np.stack(out), diags


def metrics(local, rule):
    g = np.asarray(_globals(np.asarray(local, np.float64)))
    p = rule._positions(g)
    floor = min(rule._lowest(g, p, rule.floor_region), rule._lowest(g, p, rule.forearm_region))
    cfg = rule.cfg
    forearm_gap = rule._lowest(g, p, rule.forearm_region) - rule._lowest(g, p, rule.floor_region)
    def angle(a, b):
        return math.degrees(math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    elbow = 180 - angle(p[cfg["sh"]] - p[cfg["elbow"]], p[cfg["elbow"] + 2] - p[cfg["elbow"]])
    knees = [180 - angle(p[h] - p[k], p[k + 3] - p[k]) for h, k in ((1, 4), (2, 5))]
    return (f"pelvis {100 * (p[0, 1] - floor):3.0f} cm above floor",
            f"forearm vs foot/hip {100 * forearm_gap:+3.0f} cm  elbow {elbow:3.0f}",
            f"knee L {knees[0]:3.0f} R {knees[1]:3.0f} deg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", default="sidebridge_left_20261122_033550.jsonl")
    parser.add_argument("--side", default="left")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    import production.frame_chain as frame_chain
    saved, frame_chain.SIDEBRIDGE_ACTIONS = frame_chain.SIDEBRIDGE_ACTIONS, {}   # ① = output before the rule
    try:
        result = replay_live(paths.recordings / args.recording)
    finally:
        frame_chain.SIDEBRIDGE_ACTIONS = saved
    labels, base, times = result.labels, result.final_local.astype(np.float64), result.times
    dt = float(np.median(np.diff(times)))
    columns = [("prod", "① 接入前的生产输出", base)]
    for key, title, kwargs in (("legs", "② ＋双腿伸直叠放、上侧手臂贴大腿", dict(arm=False, head=False)),
                               ("arm", "③ 当前生产：②＋支撑前臂着地、头与身体一线", {})):
        columns.append((key, title, run_rule(SideBridgeContact(args.side, **kwargs), base, times, labels, dt,
                                             result.rebases)[0]))
    probe = SideBridgeContact(args.side)
    text = {key: [metrics(local, probe) for local in track] for key, _t, track in columns}
    for key, _t, track in columns:
        rows = np.array([[float(x.split()[1]) for x in [m[0]]] for m in text[key]])
        print(f"{key}: pelvis above floor cm  median {np.median(rows):.0f}  p10 {np.percentile(rows, 10):.0f}"
              f"  p90 {np.percentile(rows, 90):.0f}", flush=True)
    out = args.out or paths.recordings / f"sidebridge_compare_{Path(args.recording).stem}.mp4"
    render(out, columns, times, text,
           "侧桥（双腿伸直）：起始侧卧→抬髋成一直线→落下。规则用校准后的传感器朝向；渲染时最低点贴地。",
           workers=args.workers, yaw_axis=(0.0, 0.0, 1.0))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
