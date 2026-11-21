"""Comparison video for any action: what live showed vs what the CURRENT live code produces on the same input.

    python -m analysis.live_vs_current_video [recording ...] [--workers 8] [--stats-only]

① the live output stored in the recording (the code that ran then);
② production/live_replay.replay_live(simulate=True): the current run.py logic -- start-pose fit at
   the recorded calibration commands, model rebase, hip-thrust lie-flat, reach alignment -- with
   inference at the same moments live inferred.  No separate offline code path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.compare_video import angle_between, fk, render
from production.live_replay import _unity_quats_to_smpl, replay_live
from production.lyingcat_contact import LyingCatContact
from production.paths import paths

DEFAULT_RECORDINGS = ("birddog_20261119_235023.jsonl", "lyingcat_20261119_234928.jsonl",
                      "crunch_20261119_234704.jsonl", "hipthrust_20261119_234600.jsonl",
                      "deadbug_20261119_234510.jsonl", "sidebridge_left_20261119_232645.jsonl")
DOWN = np.array([0.0, -1.0, 0.0])


def _measure(local, offsets):
    """Per side: forearm direction vs the head direction when the forearm is level (a reach, else None),
    knee flexion, whether the thigh is raised (within 25 deg of level), and the wrist-head distance (m;
    crunch: hands behind the head)."""
    g, p = fk(local, offsets)
    head = g[3] @ np.array([0.0, 1.0, 0.0])
    head = np.array([head[0], 0.0, head[2]]) / max(np.hypot(head[0], head[2]), 1e-9)
    out = {}
    for side, arm, axis, hip, knee in (("L", 18, (1, 0, 0), 1, 4), ("R", 19, (-1, 0, 0), 2, 5)):
        d = g[arm] @ np.asarray(axis, float)
        reach = None
        if abs(d[1]) < 0.5:       # forearm within 30 deg of level
            h = np.array([d[0], 0.0, d[2]]) / np.hypot(d[0], d[2])
            reach = float(np.degrees(np.arctan2(np.cross(head, h)[1], head @ h)))
        thigh = g[hip] @ DOWN
        out[side] = (reach, angle_between(thigh, g[knee] @ DOWN), abs(thigh[1]) < np.sin(np.radians(25)),
                     float(np.linalg.norm(p[arm + 2] - p[15])))
    return out


def _text(m):
    reach = "  ".join(f"{s} {'  --' if m[s][0] is None else f'{m[s][0]:+4.0f}'}" for s in ("L", "R"))
    return ("forearm reach dir vs head (deg): " + reach,
            f"knee flex L {m['L'][1]:3.0f}  R {m['R'][1]:3.0f}   wrist-head L {m['L'][3] * 100:3.0f}"
            f"  R {m['R'][3] * 100:3.0f} cm")


def stats(measures, mask=None):
    """Per column: |reach direction| median over level-forearm frames; knee flexion median while the
    thigh is raised (the leg reach).  `mask` limits to the frames after the start-pose rebase."""
    out = {}
    for key, rows in measures.items():
        rows = [m for i, m in enumerate(rows) if mask is None or mask[i]]
        row = {}
        for s in ("L", "R"):
            reach = [abs(m[s][0]) for m in rows if m[s][0] is not None]
            knee = [m[s][1] for m in rows if m[s][2]]
            hand = [m[s][3] * 100 for m in rows]
            row[s] = (f"reach n={len(reach)} |dir| med {np.median(reach):.0f}" if reach else "reach n=0",
                      f"raised-leg n={len(knee)} knee med {np.median(knee):.0f}" if knee else "raised-leg n=0",
                      f"wrist-head cm med {np.median(hand):.0f} p90 {np.percentile(hand, 90):.0f}" if hand else "")
        out[key] = row
    return out


def build(recording, workers, render_video=True, until=None):
    path = paths.recordings / recording
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    live = np.stack([_unity_quats_to_smpl(r["unityLocalQuaternions"]) for r in rows if r.get("type") == "prediction"])
    current = replay_live(path, simulate=True)
    columns = [("live", "① 实时录制的输出（当时的代码）", live),
               ("current", "② 当前实时代码（起始姿势修正等）", current.final_local)]
    offsets = LyingCatContact().offsets
    measures = {key: [_measure(pose, offsets) for pose in local] for key, _t, local in columns}
    text = {key: [_text(m) for m in rows_] for key, rows_ in measures.items()}
    rebase = current.rebases[0][0] if current.rebases else None
    subtitle = (f"{recording} · 当前代码起始姿势修正并重建上下文：" + "，".join(f"{t}s {a}" for t, a, _p in current.rebases)
                if current.rebases else f"{recording} · 当前代码：起始姿势未通过，未修正")
    out = paths.recordings / "action_eval_20261119" / f"live_vs_current_{Path(recording).stem}.mp4"
    if render_video:
        out = render(out, columns, current.times, text, subtitle, frame_key="current", workers=workers)
    after = current.times >= rebase if rebase is not None else np.ones(len(current.times), bool)
    if until is not None:
        after &= current.times < until
    return out, measures, after, current


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="*", default=list(DEFAULT_RECORDINGS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-only", action="store_true", help="print metrics, no video")
    parser.add_argument("--until", type=float, help="stats only before this recording time (s)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    for recording in args.recordings:
        print(recording, flush=True)
        out, measures, after, current = build(recording, args.workers, render_video=not args.stats_only,
                                          until=args.until)
        print(f"  rebases {current.rebases}", flush=True)
        for key, row in stats(measures, after).items():
            print(f"  {key} (after rebase): {row}", flush=True)
        print(out, flush=True)


if __name__ == "__main__":
    main()
